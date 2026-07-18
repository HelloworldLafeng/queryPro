"""Token-level sparse KV controller and dense future-query oracle probe."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ORACLE_B_DIR = REPOSITORY_ROOT / "oracle_b_vs_static"
for path in (REPOSITORY_ROOT, ORACLE_B_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sparse_qwen3 import SparseKVController  # noqa: E402
from future_query_union_frontier.common import repeat_kv  # noqa: E402


@dataclass
class ProbeLayer:
    query: torch.Tensor
    oracle_scores: torch.Tensor
    oracle_tokens: tuple[int, ...]
    attention_mass: torch.Tensor


@dataclass
class SparseLayerStep:
    query: torch.Tensor
    update_scores: torch.Tensor
    selected_tokens: tuple[int, ...]
    oracle_tokens: tuple[int, ...]
    selected_oracle_recall: float
    attention_recovery: float
    update_replacements: int = 0
    update_candidate_tokens: int = 0
    update_candidate_recall: float = 0.0
    update_entrant_precision: float = 0.0
    update_oracle_overlap: float = 0.0
    update_true_entrants: int = 0
    update_qk_similarity_evaluations: int = 0


@dataclass
class SparseStep:
    draft_position: int
    layers: dict[int, SparseLayerStep] = field(default_factory=dict)


class TokenIncrementalController(SparseKVController):
    """Uses one token-position set per layer, shared by all query heads."""

    def __init__(self, budget_ratio: float):
        # The inherited page size is unused by overridden sparse_attention.
        super().__init__(page_size=1, budget_ratio=budget_ratio)
        self.round_history_tokens = 0
        self.token_budget = 0
        self.current_sets: dict[int, tuple[int, ...]] = {}
        self.best_static_sets: dict[int, tuple[int, ...]] = {}
        self.oracle_update_limit = 0
        self.pending_candidates: dict[int, tuple[int, ...]] = {}
        self.pending_candidate_qk_evaluations: dict[int, int] = {}
        self.current_sparse_step: SparseStep | None = None
        self.probe_active = False
        self.probe_step = -1
        self.probe_current: dict[int, ProbeLayer] = {}
        self.probe_records: list[dict[int, ProbeLayer]] = []
        self.probe_attention_sum: dict[int, torch.Tensor] = {}
        self.probe_endpoint_sets: dict[int, tuple[int, ...]] = {}
        self.probe_endpoint_scores: dict[int, torch.Tensor] = {}

    @staticmethod
    def _topk(scores: torch.Tensor, count: int) -> tuple[int, ...]:
        count = min(count, scores.numel())
        return tuple(sorted(torch.topk(scores, k=count).indices.tolist()))

    @staticmethod
    def _aggregate_qk(post_q: torch.Tensor, repeated_k: torch.Tensor) -> torch.Tensor:
        # [batch=1, q_heads, q_len=1, dim] x [1, q_heads, kv_len, dim]
        per_head = torch.matmul(post_q, repeated_k.transpose(2, 3))[0, :, 0]
        return per_head.float().mean(dim=0)

    @staticmethod
    def _aggregate_attention(module, post_q, repeated_k, attention_mask) -> torch.Tensor:
        scores = torch.matmul(post_q, repeated_k.transpose(2, 3)) * module.scaling
        mask = attention_mask[:, :, :, : scores.shape[-1]] if attention_mask is not None else None
        if mask is not None:
            scores = scores + mask
        return torch.softmax(scores, dim=-1, dtype=torch.float32)[0, :, 0].mean(dim=0)

    def begin_dense(self, capture_positions: set[int] | None = None) -> None:
        super().begin_dense(capture_positions)
        self.probe_active = False

    def begin_probe(self, round_history_tokens: int, prior_queries: dict[int, torch.Tensor]) -> None:
        super().begin_dense()
        self.probe_active = True
        self.round_history_tokens = round_history_tokens
        self.token_budget = max(1, min(round_history_tokens, math.ceil(round_history_tokens * self.budget_ratio)))
        self.probe_prior_queries = prior_queries
        self.probe_step = -1
        self.probe_current = {}
        self.probe_records = []
        self.probe_attention_sum = {}
        self.probe_endpoint_sets = {}
        self.probe_endpoint_scores = {}

    def start_probe_step(self, position: int) -> None:
        if not self.probe_active:
            raise RuntimeError("probe is not active")
        self.probe_step = position
        self.probe_current = {}

    def finish_probe_step(self) -> None:
        if not self.probe_current:
            raise RuntimeError("dense probe captured no layers")
        self.probe_records.append(self.probe_current)
        self.probe_current = {}

    def observe_dense_attention(self, module, post_q, full_k, attention_mask, positions) -> None:
        if not self.probe_active:
            return
        if post_q.shape[0] != 1 or post_q.shape[2] != 1:
            raise RuntimeError("token oracle probe expects one query token")
        repeated_k = repeat_kv(full_k, module.num_key_value_groups)
        all_scores = self._aggregate_qk(post_q, repeated_k)
        # The query token itself is computed in the current forward and is not
        # part of the pre-existing historical KV cache for this decision.
        history_scores = all_scores[:-1]
        attention = self._aggregate_attention(module, post_q, repeated_k, attention_mask)[:-1]
        oracle = self._topk(history_scores, self.token_budget)
        layer = module.layer_idx
        self.probe_current[layer] = ProbeLayer(
            query=post_q[0, :, 0].detach().to(torch.float16).cpu(),
            oracle_scores=history_scores.detach().to(torch.float32).cpu(),
            oracle_tokens=oracle,
            attention_mass=attention.detach().to(torch.float32).cpu(),
        )
        if layer not in self.probe_attention_sum:
            self.probe_attention_sum[layer] = torch.zeros(self.round_history_tokens, dtype=torch.float64)
            prior = self.probe_prior_queries[layer].to(device=post_q.device, dtype=repeated_k.dtype)
            prior_scores = self._aggregate_qk(prior[None, :, None], repeated_k[:, :, : self.round_history_tokens])
            self.probe_endpoint_scores[layer] = prior_scores.detach().to(torch.float32).cpu()
            self.probe_endpoint_sets[layer] = self._topk(prior_scores, self.token_budget)
        available = min(self.round_history_tokens, attention.numel())
        self.probe_attention_sum[layer][:available] += attention[:available].detach().to(torch.float64).cpu()

    def finish_probe(self) -> tuple[
        list[dict[int, ProbeLayer]],
        dict[int, tuple[int, ...]],
        dict[int, torch.Tensor],
        dict[str, float],
    ]:
        if not self.probe_active or not self.probe_records:
            raise RuntimeError("probe has no complete records")
        best_sets, endpoint_mass, best_mass, per_query_mass = {}, 0.0, 0.0, 0.0
        for layer, cumulative in self.probe_attention_sum.items():
            best = self._topk(cumulative, self.token_budget)
            best_sets[layer] = best
            endpoint_mass += float(cumulative[list(self.probe_endpoint_sets[layer])].sum().item())
            best_mass += float(cumulative[list(best)].sum().item())
            for record in self.probe_records:
                attention = record[layer].attention_mass[: self.round_history_tokens]
                per_query_mass += float(torch.topk(attention, k=self.token_budget).values.sum().item())
        if best_mass + 1e-9 < endpoint_mass:
            raise RuntimeError("Best Static token coverage is below endpoint coverage")
        self.probe_active = False
        metrics = {
            "probe_endpoint_coverage_vs_per_query_attention_oracle": endpoint_mass / per_query_mass if per_query_mass else 0.0,
            "probe_best_static_coverage_vs_per_query_attention_oracle": best_mass / per_query_mass if per_query_mass else 0.0,
        }
        return self.probe_records, best_sets, self.probe_endpoint_scores, metrics

    def begin_token_round(
        self,
        method: str,
        round_history_tokens: int,
        prior_queries: dict[int, torch.Tensor],
        best_static_sets: dict[int, tuple[int, ...]] | None = None,
        update_ratio: float = 0.0,
        absolute_updates: int = 0,
    ) -> None:
        self.mode = "sparse"
        self.method = method
        self.round_history_tokens = round_history_tokens
        self.token_budget = max(1, min(round_history_tokens, math.ceil(round_history_tokens * self.budget_ratio)))
        self.prior_queries = prior_queries
        self.current_sets = dict(best_static_sets or {}) if method == "best_static_oracle" else {}
        self.best_static_sets = dict(best_static_sets or {})
        self.oracle_update_limit = (
            min(self.token_budget, absolute_updates)
            if absolute_updates > 0
            else min(self.token_budget, max(1, math.ceil(self.token_budget * update_ratio)))
            if update_ratio > 0
            else 0
        )
        self.pending_candidates = {}
        self.pending_candidate_qk_evaluations = {}

    def set_current_set(self, layer: int, tokens: tuple[int, ...]) -> None:
        if len(tokens) != self.token_budget or len(set(tokens)) != self.token_budget:
            raise ValueError("incremental selection must contain exactly K unique tokens")
        self.current_sets[layer] = tuple(sorted(tokens))

    def set_pending_candidates(
        self, layer: int, tokens: tuple[int, ...], qk_similarity_evaluations: int
    ) -> None:
        self.pending_candidates[layer] = tuple(sorted(set(tokens)))
        self.pending_candidate_qk_evaluations[layer] = qk_similarity_evaluations

    def start_token_step(self, draft_position: int) -> None:
        self.current_sparse_step = SparseStep(draft_position=draft_position)

    def finish_token_step(self) -> SparseStep:
        if self.current_sparse_step is None or not self.current_sparse_step.layers:
            raise RuntimeError("sparse token step captured no layers")
        result, self.current_sparse_step = self.current_sparse_step, None
        return result

    def sparse_attention(self, module, post_q, full_k, full_v, attention_mask):
        if self.current_sparse_step is None or post_q.shape[0] != 1 or post_q.shape[2] != 1:
            raise RuntimeError("token sparse drafting expects one active query token")
        repeated_k = repeat_kv(full_k, module.num_key_value_groups)
        repeated_v = repeat_kv(full_v, module.num_key_value_groups)
        all_scores = self._aggregate_qk(post_q, repeated_k)
        history_scores = all_scores[:-1]
        layer = module.layer_idx
        oracle = self._topk(history_scores, self.token_budget)
        if layer not in self.current_sets:
            prior = self.prior_queries[layer].to(device=post_q.device, dtype=repeated_k.dtype)
            prior_scores = self._aggregate_qk(prior[None, :, None], repeated_k[:, :, :-1])
            self.current_sets[layer] = self._topk(prior_scores, self.token_budget)
        update_replacements = 0
        update_candidate_tokens = 0
        update_candidate_recall = 0.0
        update_entrant_precision = 0.0
        update_oracle_overlap = 0.0
        update_true_entrants = 0
        update_qk_similarity_evaluations = 0
        if (
            self.method in {"oracle_incremental", "candidate_oracle"}
            and self.current_sparse_step.draft_position > 1
        ):
            current = set(self.current_sets[layer])
            oracle_set = set(oracle)
            true_entrants = oracle_set - current
            candidates = (
                set(range(history_scores.numel())) - current
                if self.method == "oracle_incremental"
                else set(self.pending_candidates.get(layer, ())) - current
            )
            eligible = true_entrants & candidates
            additions = sorted(
                eligible, key=lambda token: float(history_scores[token]), reverse=True
            )[: self.oracle_update_limit]
            removable = sorted(
                current - oracle_set, key=lambda token: float(history_scores[token])
            )
            evictions = removable[: len(additions)]
            updated = (current - set(evictions)) | set(additions)
            if len(updated) != self.token_budget:
                raise RuntimeError("Oracle incremental update changed token budget")
            self.current_sets[layer] = tuple(sorted(updated))
            update_replacements = len(additions)
            update_candidate_tokens = len(candidates)
            update_candidate_recall = len(candidates & true_entrants) / len(true_entrants) if true_entrants else 1.0
            update_entrant_precision = 1.0 if additions else 1.0
            update_oracle_overlap = len(updated & oracle_set) / self.token_budget
            update_true_entrants = len(true_entrants)
            if self.method == "candidate_oracle":
                update_qk_similarity_evaluations = self.pending_candidate_qk_evaluations.get(layer, 0)
        selected = oracle if self.method == "oracle_b" else self.current_sets[layer]
        if any(token >= history_scores.numel() for token in selected):
            raise RuntimeError("selection contains a token that is not yet causal history")

        keep = torch.zeros(full_k.shape[2], dtype=torch.bool, device=post_q.device)
        keep[list(selected)] = True
        keep[-1] = True  # current token is not charged to the historical-cache budget
        scores = torch.matmul(post_q, repeated_k.transpose(2, 3)) * module.scaling
        mask = attention_mask[:, :, :, : scores.shape[-1]] if attention_mask is not None else None
        if mask is not None:
            scores = scores + mask
        full_weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
        aggregate_mass = full_weights[0, :, 0, :-1].mean(dim=0)
        selected_mass = float(aggregate_mass[list(selected)].sum().item())
        oracle_mass = float(torch.topk(aggregate_mass, k=self.token_budget).values.sum().item())
        sparse_scores = scores.masked_fill(~keep[None, None, None, :], torch.finfo(scores.dtype).min)
        sparse_weights = torch.softmax(sparse_scores, dim=-1, dtype=torch.float32).to(post_q.dtype)
        output = torch.matmul(sparse_weights, repeated_v).transpose(1, 2).contiguous()
        self.current_sparse_step.layers[layer] = SparseLayerStep(
            query=post_q[0, :, 0].detach().to(torch.float16).cpu(),
            update_scores=all_scores.detach().to(torch.float32).cpu(),
            selected_tokens=tuple(selected),
            oracle_tokens=oracle,
            selected_oracle_recall=len(set(selected) & set(oracle)) / self.token_budget,
            attention_recovery=selected_mass / oracle_mass if oracle_mass else 0.0,
            update_replacements=update_replacements,
            update_candidate_tokens=update_candidate_tokens,
            update_candidate_recall=update_candidate_recall,
            update_entrant_precision=update_entrant_precision,
            update_oracle_overlap=update_oracle_overlap,
            update_true_entrants=update_true_entrants,
            update_qk_similarity_evaluations=update_qk_similarity_evaluations,
        )
        return output, sparse_weights
