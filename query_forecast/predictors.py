from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    def __init__(self, head_dim: int, channels: int | None = None, num_layers: int = 3, kernel_size: int = 3):
        super().__init__()
        width = channels or head_dim
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


@dataclass
class TCNTrainingConfig:
    epochs: int = 4
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_examples: int = 12000
    num_workers: int = 0
    device: str = "cpu"


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


def train_tcn_model(
    model: TinyTCNPredictor,
    histories: torch.Tensor,
    targets: torch.Tensor,
    config: TCNTrainingConfig,
) -> TinyTCNPredictor:
    device = torch.device(config.device)
    model = model.to(device)
    dataset = TensorDataset(histories, targets)
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
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
    return model.eval()
