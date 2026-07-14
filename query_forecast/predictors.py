from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def causal_pad_1d(x: torch.Tensor, kernel_size: int, dilation: int) -> torch.Tensor:
    left = (kernel_size - 1) * dilation
    return F.pad(x, (left, 0))


class ResidualCausalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = causal_pad_1d(x, self.kernel_size, self.dilation)
        y = self.conv(y)
        y = y.transpose(1, 2)
        y = self.norm(y)
        y = F.gelu(y)
        y = y.transpose(1, 2)
        return residual + y


class TinyTCNPredictor(nn.Module):
    def __init__(self, head_dim: int, channels: int = 32, num_layers: int = 4, kernel_size: int = 2):
        super().__init__()
        width = channels
        self.head_dim = head_dim
        self.channels = width
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.in_proj = nn.Conv1d(head_dim, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            ResidualCausalConvBlock(width, kernel_size=kernel_size, dilation=2**idx) for idx in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, head_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        x = history.transpose(1, 2)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x[:, :, -1]
        x = self.out_norm(x)
        return self.out_proj(x)


class TemporalLinearPredictor(nn.Module):
    """A dimension-shared temporal filter that predicts a future query delta.

    It has only ``history_length + 1`` trainable parameters and costs roughly
    ``history_length * head_dim`` MACs for one head.  Sharing the temporal
    coefficients across query dimensions makes this a useful lower-cost point
    between hand-written drift rules and a convolutional predictor.
    """

    def __init__(self, history_length: int):
        super().__init__()
        self.history_length = history_length
        self.temporal = nn.Linear(history_length, 1)
        nn.init.zeros_(self.temporal.weight)
        nn.init.zeros_(self.temporal.bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.shape[1] != self.history_length:
            raise ValueError(f"Expected history length {self.history_length}, got {history.shape[1]}.")
        return self.temporal(history.transpose(1, 2)).squeeze(-1)


@dataclass
class TCNTrainingConfig:
    epochs: int = 4
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_examples: int = 12000
    num_workers: int = 0
    device: str = "cpu"
    seed: int = 7


class ReservoirBuffer:
    def __init__(self, max_examples: int, seed: int):
        self.max_examples = max_examples
        self.seen = 0
        self.rng = torch.Generator().manual_seed(seed)
        self.histories: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []

    def add(self, history: torch.Tensor, target: torch.Tensor) -> None:
        self.seen += 1
        history = history.detach().to(dtype=torch.float16, device="cpu")
        target = target.detach().to(dtype=torch.float16, device="cpu")
        if len(self.histories) < self.max_examples:
            self.histories.append(history)
            self.targets.append(target)
            return
        idx = int(torch.randint(0, self.seen, (1,), generator=self.rng).item())
        if idx < self.max_examples:
            self.histories[idx] = history
            self.targets[idx] = target

    def __len__(self) -> int:
        return len(self.histories)

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.histories:
            raise RuntimeError("Reservoir buffer is empty.")
        histories = torch.stack(self.histories).to(dtype=torch.float32)
        targets = torch.stack(self.targets).to(dtype=torch.float32)
        return histories, targets


def train_predictor_model(
    model: nn.Module,
    histories: torch.Tensor,
    targets: torch.Tensor,
    config: TCNTrainingConfig,
    validation_data: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> nn.Module:
    device = torch.device(config.device)
    model = model.to(device)
    dataset = TensorDataset(histories, targets)
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
        generator=loader_generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    for _ in range(config.epochs):
        model.train()
        for batch_histories, batch_targets in loader:
            batch_histories = batch_histories.to(device)
            batch_targets = batch_targets.to(device)
            pred = model(batch_histories)
            loss = F.smooth_l1_loss(pred, batch_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if validation_data is not None:
            validation_histories, validation_targets = validation_data
            model.eval()
            with torch.no_grad():
                validation_pred = model(validation_histories.to(device))
                validation_loss = float(
                    F.smooth_l1_loss(validation_pred, validation_targets.to(device)).item()
                )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())

    if validation_data is not None:
        model.load_state_dict(best_state)
    return model.eval()


def predictor_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def estimate_predictor_macs(model: nn.Module, history_length: int, head_dim: int) -> int:
    """Approximate multiply-accumulates for one head and one forecast."""
    if isinstance(model, TemporalLinearPredictor):
        return history_length * head_dim
    if isinstance(model, TinyTCNPredictor):
        width = model.channels
        in_projection = history_length * head_dim * width
        convolution_blocks = history_length * model.num_layers * width * width * model.kernel_size
        output_projection = width * head_dim
        return in_projection + convolution_blocks + output_projection
    raise TypeError(f"Unsupported predictor type: {type(model).__name__}")


# Backward-compatible name for callers outside this repository.
train_tcn_model = train_predictor_model
