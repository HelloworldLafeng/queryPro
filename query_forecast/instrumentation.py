from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def inverse_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) - (rotate_half(x) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@dataclass
class SelectionSpec:
    layers: list[int]
    heads: list[int]
    kv_heads: list[int]
    num_key_value_groups: int


@dataclass
class CapturedLayerStep:
    layer_idx: int
    pre_query: torch.Tensor
    post_query: torch.Tensor
    pre_key: torch.Tensor
    post_key: torch.Tensor
    positions: torch.Tensor
    attn_weights: torch.Tensor | None


class AttentionCaptureRuntime:
    def __init__(self, selection: SelectionSpec):
        self.selection = selection
        self.collect_attn_weights = False
        self.current_step: dict[int, CapturedLayerStep] = {}

    def begin_step(self, collect_attn_weights: bool) -> None:
        self.collect_attn_weights = collect_attn_weights
        self.current_step = {}

    def end_step(self) -> dict[int, CapturedLayerStep]:
        step = self.current_step
        self.current_step = {}
        return step

    def record(self, capture: CapturedLayerStep) -> None:
        self.current_step[capture.layer_idx] = capture


def _manual_attention(
    module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states_full = repeat_kv(key_states, module.num_key_value_groups)
    value_states_full = repeat_kv(value_states, module.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states_full.transpose(2, 3)) * module.scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states_full.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states_full)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _fast_attention(
    module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, None]:
    key_states_full = repeat_kv(key_states, module.num_key_value_groups)
    value_states_full = repeat_kv(value_states, module.num_key_value_groups)
    attn_mask = None
    if attention_mask is not None:
        attn_mask = attention_mask[:, :, :, : key_states_full.shape[-2]]
    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states_full,
        value_states_full,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=module.scaling,
        is_causal=False,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def patch_qwen3_attention(model, runtime: AttentionCaptureRuntime) -> None:
    for layer in model.model.layers:
        attn = layer.self_attn
        original_forward = attn.forward

        def wrapped_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_values=None,
            cache_position: torch.LongTensor | None = None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            post_query, post_key = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            positions = cache_position
            if positions is None:
                positions = kwargs.get("position_ids")
            if positions is None:
                raise RuntimeError(
                    "Qwen3 attention did not provide cache_position or position_ids; "
                    "pin a supported Transformers version or update the capture adapter."
                )
            positions = positions.reshape(-1)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                full_key_states, value_states = past_key_values.update(post_key, value_states, self.layer_idx, cache_kwargs)
            else:
                full_key_states = post_key

            if self.layer_idx in runtime.selection.layers:
                head_idx = torch.tensor(runtime.selection.heads, device=query_states.device)
                kv_idx = torch.tensor(runtime.selection.kv_heads, device=key_states.device)
                runtime.record(
                    CapturedLayerStep(
                        layer_idx=self.layer_idx,
                        pre_query=query_states.index_select(1, head_idx).detach().to(dtype=torch.float16).squeeze(0),
                        post_query=post_query.index_select(1, head_idx).detach().to(dtype=torch.float16).squeeze(0),
                        pre_key=key_states.index_select(1, kv_idx).detach().to(dtype=torch.float16).squeeze(0),
                        post_key=post_key.index_select(1, kv_idx).detach().to(dtype=torch.float16).squeeze(0),
                        positions=positions.detach().clone(),
                        attn_weights=None,
                    )
                )

            if runtime.collect_attn_weights:
                attn_output, attn_weights = _manual_attention(self, post_query, full_key_states, value_states, attention_mask)
            else:
                attn_output, attn_weights = _fast_attention(self, post_query, full_key_states, value_states, attention_mask)

            if self.layer_idx in runtime.selection.layers and attn_weights is not None:
                capture = runtime.current_step[self.layer_idx]
                head_idx = torch.tensor(runtime.selection.heads, device=attn_weights.device)
                capture.attn_weights = attn_weights.index_select(1, head_idx).detach().to(dtype=torch.float16).squeeze(0)
                runtime.current_step[self.layer_idx] = capture

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        attn.forward = types.MethodType(wrapped_forward, attn)
        attn._query_forecast_original_forward = original_forward
