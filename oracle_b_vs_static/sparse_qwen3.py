"""Correctness-first Qwen3 sparse-attention adapter for Oracle-B evaluation.

The adapter keeps the dense KV cache intact and changes only which pages take
part in the attention computation.  Page selection is per layer and query head;
GQA query heads are scored only against their corresponding KV head.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from future_query_union_frontier.common import apply_rotary, repeat_kv


def _page_max(scores: torch.Tensor, page_size: int) -> torch.Tensor:
    padding = (-scores.numel()) % page_size
    return F.pad(scores, (0, padding), value=float("-inf")).reshape(-1, page_size).max(dim=-1).values


def _jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


@dataclass
class StepDiagnostics:
    draft_position: int
    static_oracle_recalls: list[float] = field(default_factory=list)
    attention_recoveries: list[float] = field(default_factory=list)
    static_pages: dict[tuple[int, int], tuple[int, ...]] = field(default_factory=dict)
    oracle_pages: dict[tuple[int, int], tuple[int, ...]] = field(default_factory=dict)

    @property
    def mean_static_oracle_recall(self) -> float:
        return sum(self.static_oracle_recalls) / len(self.static_oracle_recalls) if self.static_oracle_recalls else float("nan")

    @property
    def mean_attention_recovery(self) -> float:
        return sum(self.attention_recoveries) / len(self.attention_recoveries) if self.attention_recoveries else float("nan")


class SparseKVController:
    def __init__(self, page_size: int, budget_ratio: float):
        self.page_size = page_size
        self.budget_ratio = budget_ratio
        self.mode = "dense"
        self.method = ""
        self.round_start_tokens = 0
        self.budget_pages = 0
        self.prior_queries: dict[int, torch.Tensor] = {}
        self.static_pages: dict[tuple[int, int], tuple[int, ...]] = {}
        self.current_step: StepDiagnostics | None = None
        self.capture_positions: set[int] = set()
        self.captured_queries: dict[int, dict[int, torch.Tensor]] = {}

    def begin_dense(self, capture_positions: set[int] | None = None) -> None:
        self.mode = "dense"
        self.capture_positions = capture_positions or set()
        self.captured_queries = {}
        self.current_step = None

    def begin_sparse(self, method: str, round_start_tokens: int, prior_queries: dict[int, torch.Tensor]) -> None:
        if method not in {"static", "oracle_b"}:
            raise ValueError(f"unknown sparse method: {method}")
        self.mode = "sparse"
        self.method = method
        self.round_start_tokens = round_start_tokens
        num_pages = math.ceil(round_start_tokens / self.page_size)
        self.budget_pages = max(1, min(num_pages, math.ceil(num_pages * self.budget_ratio)))
        self.prior_queries = prior_queries
        self.static_pages = {}

    def start_step(self, draft_position: int) -> None:
        self.current_step = StepDiagnostics(draft_position=draft_position)

    def finish_step(self) -> StepDiagnostics:
        if self.current_step is None:
            raise RuntimeError("no active sparse step")
        result, self.current_step = self.current_step, None
        return result

    def capture_dense_queries(self, layer_idx: int, post_q: torch.Tensor, positions: torch.Tensor) -> None:
        if not self.capture_positions:
            return
        flat_positions = positions.reshape(-1).tolist()
        for local_idx, absolute_pos in enumerate(flat_positions):
            if int(absolute_pos) in self.capture_positions:
                self.captured_queries.setdefault(int(absolute_pos), {})[layer_idx] = (
                    post_q[0, :, local_idx].detach().to(torch.float16).cpu()
                )

    def _select_pages(self, query: torch.Tensor, keys: torch.Tensor, num_pages: int) -> tuple[int, ...]:
        scores = torch.matmul(keys.float(), query.float())
        page_scores = _page_max(scores, self.page_size)[:num_pages]
        return tuple(sorted(torch.topk(page_scores, k=self.budget_pages).indices.tolist()))

    def sparse_attention(
        self,
        module,
        post_q: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if post_q.shape[2] != 1 or post_q.shape[0] != 1:
            raise RuntimeError("sparse drafting expects batch_size=1 and one query token per forward")
        if self.current_step is None:
            raise RuntimeError("start_step must be called before sparse attention")
        if full_k.shape[2] < self.round_start_tokens:
            raise RuntimeError("KV cache is shorter than the declared round-start prefix")

        repeated_k = repeat_kv(full_k, module.num_key_value_groups)
        repeated_v = repeat_kv(full_v, module.num_key_value_groups)
        q_heads = post_q.shape[1]
        num_pages = math.ceil(full_k.shape[2] / self.page_size)
        if self.budget_pages > num_pages:
            raise RuntimeError("page budget exceeds available pages")
        keep = torch.zeros((q_heads, full_k.shape[2]), dtype=torch.bool, device=post_q.device)
        prior = self.prior_queries.get(module.layer_idx)
        if prior is None or prior.shape[0] != q_heads:
            raise RuntimeError(f"missing/mismatched endpoint query for layer {module.layer_idx}")
        prior = prior.to(post_q.device)

        for head in range(q_heads):
            key = (module.layer_idx, head)
            keys_for_head = repeated_k[0, head]
            if key not in self.static_pages:
                self.static_pages[key] = self._select_pages(prior[head], keys_for_head, num_pages)
            static_pages = self.static_pages[key]
            oracle_pages = self._select_pages(post_q[0, head, 0], keys_for_head, num_pages)
            selected_pages = oracle_pages if self.method == "oracle_b" else static_pages
            for page in selected_pages:
                begin = page * self.page_size
                keep[head, begin : min(begin + self.page_size, full_k.shape[2])] = True

            static_set, oracle_set = set(static_pages), set(oracle_pages)
            self.current_step.static_oracle_recalls.append(len(static_set & oracle_set) / len(oracle_set))
            self.current_step.static_pages[key] = static_pages
            self.current_step.oracle_pages[key] = oracle_pages

        scores = torch.matmul(post_q, repeated_k.transpose(2, 3)) * module.scaling
        mask = attention_mask[:, :, :, : scores.shape[-1]] if attention_mask is not None else None
        if mask is not None:
            scores = scores + mask
        full_weights = torch.softmax(scores, dim=-1, dtype=torch.float32)

        for head in range(q_heads):
            selected_mass = float(full_weights[0, head, 0, keep[head]].sum().item())
            page_mass = F.pad(
                full_weights[0, head, 0],
                (0, (-full_weights.shape[-1]) % self.page_size),
            ).reshape(-1, self.page_size).sum(dim=-1)[:num_pages]
            oracle_mass = float(torch.topk(page_mass, k=self.budget_pages).values.sum().item())
            self.current_step.attention_recoveries.append(selected_mass / oracle_mass if oracle_mass > 0 else 0.0)

        sparse_scores = scores.masked_fill(~keep[None, :, None, :], torch.finfo(scores.dtype).min)
        sparse_weights = torch.softmax(sparse_scores, dim=-1, dtype=torch.float32).to(post_q.dtype)
        output = torch.matmul(sparse_weights, repeated_v).transpose(1, 2).contiguous()
        return output, sparse_weights


def adjacent_oracle_overlap(left: StepDiagnostics, right: StepDiagnostics) -> float:
    common = sorted(set(left.oracle_pages) & set(right.oracle_pages))
    values = [_jaccard(left.oracle_pages[key], right.oracle_pages[key]) for key in common]
    return sum(values) / len(values) if values else float("nan")


def endpoint_overlap(first: StepDiagnostics, last: StepDiagnostics) -> float:
    return adjacent_oracle_overlap(first, last)


def patch_qwen3_for_sparse_drafting(model, controller: SparseKVController) -> None:
    """Patch Qwen3 attention while preserving its projections, RoPE and cache."""
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
                raise RuntimeError("Qwen3 did not provide cache_position or position_ids")
            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                full_k, full_v = past_key_values.update(post_k, value, self.layer_idx, cache_kwargs)
            else:
                full_k, full_v = post_k, value

            if controller.mode == "sparse":
                output, weights = controller.sparse_attention(self, post_q, full_k, full_v, attention_mask)
            else:
                controller.capture_dense_queries(self.layer_idx, post_q, positions)
                observe_dense = getattr(controller, "observe_dense_attention", None)
                if observe_dense is not None:
                    observe_dense(self, post_q, full_k, attention_mask, positions)
                repeated_k = repeat_kv(full_k, self.num_key_value_groups)
                repeated_v = repeat_kv(full_v, self.num_key_value_groups)
                mask = attention_mask[:, :, :, : repeated_k.shape[-2]] if attention_mask is not None else None
                output = F.scaled_dot_product_attention(
                    post_q,
                    repeated_k,
                    repeated_v,
                    attn_mask=mask,
                    dropout_p=0.0,
                    scale=self.scaling,
                    is_causal=False,
                ).transpose(1, 2).contiguous()
                weights = None
            return self.o_proj(output.reshape(*input_shape, -1).contiguous()), weights

        attention_module.forward = types.MethodType(wrapped, attention_module)
