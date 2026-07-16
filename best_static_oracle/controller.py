"""Best-static attention oracle built on the Oracle-B sparse controller."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ORACLE_B_DIR = REPOSITORY_ROOT / "oracle_b_vs_static"
for path in (REPOSITORY_ROOT, ORACLE_B_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sparse_qwen3 import SparseKVController  # noqa: E402
from future_query_union_frontier.common import repeat_kv  # noqa: E402


class BestStaticOracleController(SparseKVController):
    """Adds a dense future-query probe and externally supplied fixed pages."""

    def __init__(self, page_size: int, budget_ratio: float):
        super().__init__(page_size, budget_ratio)
        self.probe_active = False
        self.probe_num_pages = 0
        self.probe_prior_queries: dict[int, torch.Tensor] = {}
        self.probe_page_mass: dict[tuple[int, int], torch.Tensor] = {}
        self.probe_per_query_oracle_mass: dict[tuple[int, int], float] = {}
        self.probe_endpoint_pages: dict[tuple[int, int], tuple[int, ...]] = {}
        self.probe_endpoint_mass: dict[tuple[int, int], float] = {}

    def begin_dense(self, capture_positions: set[int] | None = None) -> None:
        super().begin_dense(capture_positions)
        self.probe_active = False

    def begin_best_static_probe(
        self,
        round_start_tokens: int,
        prior_queries: dict[int, torch.Tensor],
    ) -> None:
        super().begin_dense()
        self.probe_active = True
        self.round_start_tokens = round_start_tokens
        self.probe_num_pages = math.ceil(round_start_tokens / self.page_size)
        self.budget_pages = max(
            1,
            min(self.probe_num_pages, math.ceil(self.probe_num_pages * self.budget_ratio)),
        )
        self.probe_prior_queries = prior_queries
        self.probe_page_mass = {}
        self.probe_per_query_oracle_mass = {}
        self.probe_endpoint_pages = {}
        self.probe_endpoint_mass = {}

    def observe_dense_attention(
        self,
        module,
        post_q: torch.Tensor,
        full_k: torch.Tensor,
        attention_mask: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> None:
        if not self.probe_active:
            return
        if post_q.shape[0] != 1 or post_q.shape[2] != 1:
            raise RuntimeError("Best Static Oracle probe expects batch_size=1 and one dense query at a time")
        repeated_k = repeat_kv(full_k, module.num_key_value_groups)
        scores = torch.matmul(post_q, repeated_k.transpose(2, 3)) * module.scaling
        mask = attention_mask[:, :, :, : scores.shape[-1]] if attention_mask is not None else None
        if mask is not None:
            scores = scores + mask
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32)[0, :, 0]
        padding = (-weights.shape[-1]) % self.page_size
        page_mass = F.pad(weights, (0, padding)).reshape(weights.shape[0], -1, self.page_size).sum(dim=-1)
        page_mass = page_mass[:, : self.probe_num_pages]
        prior = self.probe_prior_queries.get(module.layer_idx)
        if prior is None or prior.shape[0] != post_q.shape[1]:
            raise RuntimeError(f"missing/mismatched prior query for layer {module.layer_idx}")
        prior = prior.to(post_q.device)

        for head in range(post_q.shape[1]):
            key = (module.layer_idx, head)
            mass = page_mass[head].detach().to(torch.float64).cpu()
            if key not in self.probe_page_mass:
                self.probe_page_mass[key] = torch.zeros(self.probe_num_pages, dtype=torch.float64)
                self.probe_endpoint_pages[key] = self._select_pages(
                    prior[head], repeated_k[0, head, : self.round_start_tokens], self.probe_num_pages
                )
                self.probe_endpoint_mass[key] = 0.0
                self.probe_per_query_oracle_mass[key] = 0.0
            self.probe_page_mass[key] += mass
            endpoint = torch.tensor(self.probe_endpoint_pages[key], dtype=torch.long)
            self.probe_endpoint_mass[key] += float(mass[endpoint].sum().item())
            self.probe_per_query_oracle_mass[key] += float(
                torch.topk(mass, k=self.budget_pages).values.sum().item()
            )

    def finish_best_static_probe(self) -> tuple[dict[tuple[int, int], tuple[int, ...]], dict[str, float]]:
        if not self.probe_active or not self.probe_page_mass:
            raise RuntimeError("Best Static Oracle probe has no captured attention")
        pages: dict[tuple[int, int], tuple[int, ...]] = {}
        best_mass, endpoint_mass, per_query_mass, eligible_mass = [], [], [], []
        for key, cumulative_mass in self.probe_page_mass.items():
            selected = tuple(sorted(torch.topk(cumulative_mass, k=self.budget_pages).indices.tolist()))
            pages[key] = selected
            selected_mass = float(cumulative_mass[list(selected)].sum().item())
            if selected_mass + 1e-10 < self.probe_endpoint_mass[key]:
                raise RuntimeError("Best Static coverage is below endpoint coverage; oracle construction is invalid")
            best_mass.append(selected_mass)
            endpoint_mass.append(self.probe_endpoint_mass[key])
            per_query_mass.append(self.probe_per_query_oracle_mass[key])
            eligible_mass.append(float(cumulative_mass.sum().item()))
        self.probe_active = False

        def ratio_sum(numerators: list[float], denominators: list[float]) -> float:
            denominator = sum(denominators)
            return sum(numerators) / denominator if denominator > 0 else 0.0

        metrics = {
            "probe_endpoint_coverage_vs_per_query_attention_oracle": ratio_sum(endpoint_mass, per_query_mass),
            "probe_best_static_coverage_vs_per_query_attention_oracle": ratio_sum(best_mass, per_query_mass),
            "probe_endpoint_fraction_of_eligible_attention": ratio_sum(endpoint_mass, eligible_mass),
            "probe_best_static_fraction_of_eligible_attention": ratio_sum(best_mass, eligible_mass),
            "probe_best_static_absolute_coverage_gain": ratio_sum(best_mass, eligible_mass)
            - ratio_sum(endpoint_mass, eligible_mass),
        }
        return pages, metrics

    def begin_sparse_with_pages(
        self,
        method: str,
        round_start_tokens: int,
        prior_queries: dict[int, torch.Tensor],
        best_static_pages: dict[tuple[int, int], tuple[int, ...]] | None = None,
    ) -> None:
        if method == "best_static_oracle":
            if not best_static_pages:
                raise ValueError("best_static_oracle requires precomputed fixed pages")
            super().begin_sparse("static", round_start_tokens, prior_queries)
            self.static_pages = dict(best_static_pages)
            self.method = method
            return
        super().begin_sparse(method, round_start_tokens, prior_queries)
