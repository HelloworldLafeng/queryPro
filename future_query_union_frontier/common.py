from __future__ import annotations

import json
import random
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class ExperimentSample:
    sample_id: str
    dataset_name: str
    source_family: str
    task_type: str
    language: str
    context: str
    prompt_input: str


def iter_reasoning_jsonl(root: Path, datasets: list[str]) -> Iterable[ExperimentSample]:
    for dataset_name in datasets:
        path = root / dataset_name / "samples.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                item = json.loads(line)
                yield ExperimentSample(
                    sample_id=item.get("sample_id") or f"{dataset_name}-{index}",
                    dataset_name=item.get("dataset_name", dataset_name),
                    source_family="reasoning",
                    task_type=item.get("task_type", "reasoning"),
                    language=item.get("language", "en"),
                    context=item.get("context", ""),
                    prompt_input=item["prompt_input"],
                )


def reservoir_sample(samples: Iterable[ExperimentSample], count: int, seed: int) -> list[ExperimentSample]:
    rng = random.Random(seed)
    result: list[ExperimentSample] = []
    for seen, sample in enumerate(samples, start=1):
        if len(result) < count:
            result.append(sample)
        else:
            position = rng.randint(1, seen)
            if position <= count:
                result[position - 1] = sample
    return result


def sample_allocated(root: Path, allocation: dict[str, int], seed: int) -> list[ExperimentSample]:
    result = []
    for offset, (dataset_name, count) in enumerate(allocation.items()):
        result.extend(reservoir_sample(iter_reasoning_jsonl(root, [dataset_name]), count, seed + offset))
    return sorted(result, key=lambda sample: (sample.dataset_name, sample.sample_id))


def build_prompt(sample: ExperimentSample) -> str:
    if sample.task_type == "code_generation":
        return f"Please reason about the algorithm, then write the final code.\n\n[Problem]\n{sample.prompt_input}\n\n[Reasoning and Code]\n"
    return f"Please solve the problem step by step.\n\n[Problem]\n{sample.prompt_input}\n\n[Detailed Reasoning]\n"


def generation_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return prompt


def encode_prompt(tokenizer, prompt: str, max_tokens: int) -> torch.Tensor:
    return tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)["input_ids"]


def sample_next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    return torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return states
    batch, heads, length, dim = states.shape
    return states[:, :, None].expand(batch, heads, repeats, length, dim).reshape(batch, heads * repeats, length, dim)


@dataclass
class SelectionSpec:
    layers: list[int]
    heads: list[int]
    kv_heads: list[int]
    kv_groups: int


@dataclass
class CapturedLayer:
    pre_query: torch.Tensor
    post_query: torch.Tensor
    post_key: torch.Tensor
    positions: torch.Tensor
    attention: torch.Tensor | None


class CaptureRuntime:
    def __init__(self, selection: SelectionSpec):
        self.selection = selection
        self.collect_attention = False
        self.current: dict[int, CapturedLayer] = {}

    def begin(self, collect_attention: bool) -> None:
        self.collect_attention = collect_attention
        self.current = {}

    def end(self) -> dict[int, CapturedLayer]:
        result, self.current = self.current, {}
        return result


def _attention(module, q, k, v, mask, return_weights: bool):
    k = repeat_kv(k, module.num_key_value_groups)
    v = repeat_kv(v, module.num_key_value_groups)
    mask = mask[:, :, :, : k.shape[-2]] if mask is not None else None
    if return_weights:
        weights = torch.matmul(q, k.transpose(2, 3)) * module.scaling
        if mask is not None:
            weights = weights + mask
        weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(weights, v).transpose(1, 2).contiguous(), weights
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, scale=module.scaling, is_causal=False)
    return output.transpose(1, 2).contiguous(), None


def patch_qwen3(model, runtime: CaptureRuntime) -> None:
    for layer in model.model.layers:
        attention_module = layer.self_attn

        def wrapped(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, cache_position=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            pre_q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            pre_k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            post_q, post_k = apply_rotary(pre_q, pre_k, cos, sin)
            positions = cache_position if cache_position is not None else kwargs.get("position_ids")
            if positions is None:
                raise RuntimeError("Qwen3 did not provide cache_position or position_ids.")
            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                full_k, value = past_key_values.update(post_k, value, self.layer_idx, cache_kwargs)
            else:
                full_k = post_k
            selected = self.layer_idx in runtime.selection.layers
            if selected:
                head_ids = torch.tensor(runtime.selection.heads, device=pre_q.device)
                kv_ids = torch.tensor(runtime.selection.kv_heads, device=pre_k.device)
                runtime.current[self.layer_idx] = CapturedLayer(
                    pre_query=pre_q.index_select(1, head_ids).detach().to(torch.float16).squeeze(0),
                    post_query=post_q.index_select(1, head_ids).detach().to(torch.float16).squeeze(0),
                    post_key=post_k.index_select(1, kv_ids).detach().to(torch.float16).squeeze(0),
                    positions=positions.reshape(-1).detach().clone(),
                    attention=None,
                )
            output, weights = _attention(self, post_q, full_k, value, attention_mask, selected and runtime.collect_attention)
            if selected and weights is not None:
                head_ids = torch.tensor(runtime.selection.heads, device=weights.device)
                runtime.current[self.layer_idx].attention = weights.index_select(1, head_ids).detach().to(torch.float16).squeeze(0)
            return self.o_proj(output.reshape(*input_shape, -1).contiguous()), weights

        attention_module.forward = types.MethodType(wrapped, attention_module)


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def select_layers(count: int, spec: str) -> list[int]:
    if spec == "all":
        return list(range(count))
    if spec == "representative":
        return sorted({0, count // 6, 2 * count // 6, 3 * count // 6, 4 * count // 6, 5 * count // 6, count - 1})
    result = sorted(set(parse_ints(spec)))
    if not result or result[-1] >= count:
        raise ValueError("Invalid layer selection.")
    return result


def make_selection(model, layer_spec: str, head_stride: int) -> SelectionSpec:
    config = model.config
    groups = config.num_attention_heads // config.num_key_value_heads
    heads = list(range(0, config.num_attention_heads, head_stride))
    return SelectionSpec(select_layers(config.num_hidden_layers, layer_spec), heads, sorted({head // groups for head in heads}), groups)


def gather_keys(chunks: list[torch.Tensor], kv_local: int, prefix_len: int) -> torch.Tensor:
    result, consumed = [], 0
    for chunk in chunks:
        take = min(chunk.shape[1], prefix_len - consumed)
        if take <= 0:
            break
        result.append(chunk[kv_local, :take].float())
        consumed += take
    return torch.cat(result)


def rotary_for_position(model, position: int, device: torch.device):
    position_ids = torch.tensor([[position]], device=device)
    dummy = torch.zeros((1, 1, model.config.hidden_size), device=device, dtype=model.dtype)
    cos, sin = model.model.rotary_emb(dummy, position_ids)
    return cos[0, 0].float(), sin[0, 0].float()


def rotate_query(query: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return query.float() * cos + rotate_half(query.float()) * sin
