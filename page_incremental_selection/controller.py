"""Page-level Oracle Incremental controller matching the existing baselines."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ORACLE_B_DIR = REPOSITORY_ROOT / "oracle_b_vs_static"
for path in (REPOSITORY_ROOT, ORACLE_B_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from future_query_union_frontier.common import repeat_kv  # noqa: E402
from sparse_qwen3 import SparseKVController, StepDiagnostics  # noqa: E402


@dataclass
class PageIncrementalStep(StepDiagnostics):
    """Per-position diagnostics aggregated across layer/query-head selectors."""

    update_replacements: list[int] = field(default_factory=list)
    update_limits: list[int] = field(default_factory=list)
    true_entrant_counts: list[int] = field(default_factory=list)

    @property
    def mean_update_replacements(self) -> float:
        return (
            sum(self.update_replacements) / len(self.update_replacements)
            if self.update_replacements
            else 0.0
        )

    @property
    def mean_update_limit(self) -> float:
        return sum(self.update_limits) / len(self.update_limits) if self.update_limits else 0.0

    @property
    def mean_true_entrants(self) -> float:
        return (
            sum(self.true_entrant_counts) / len(self.true_entrant_counts)
            if self.true_entrant_counts
            else 0.0
        )


def _page_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    page_size: int,
    num_pages: int,
) -> torch.Tensor:
    """Return the existing baseline's max-token post-RoPE QK page score."""

    token_scores = torch.matmul(keys.float(), query.float())
    padding = (-token_scores.numel()) % page_size
    return F.pad(token_scores, (0, padding), value=float("-inf")).reshape(
        -1, page_size
    ).max(dim=-1).values[:num_pages]


class PageIncrementalController(SparseKVController):
    """Refresh at most ceil(r * B) pages for every layer/query head and step."""

    def __init__(self, page_size: int, budget_ratio: float):
        super().__init__(page_size, budget_ratio)
        self.update_ratio = 0.0
        self.update_limit_pages = 0

    def begin_incremental(
        self,
        update_ratio: float,
        round_start_tokens: int,
        prior_queries: dict[int, torch.Tensor],
    ) -> None:
        if not (0.0 < update_ratio <= 1.0):
            raise ValueError("update_ratio must be in (0, 1]")
        # This exactly reproduces endpoint-mean initialization, page budget,
        # and fixed-set construction from the existing static baseline.
        super().begin_sparse("static", round_start_tokens, prior_queries)
        self.method = "page_incremental"
        self.update_ratio = update_ratio
        self.update_limit_pages = min(
            self.budget_pages,
            max(1, math.ceil(self.budget_pages * update_ratio)),
        )

    def start_step(self, draft_position: int) -> None:
        self.current_step = PageIncrementalStep(draft_position=draft_position)

    def sparse_attention(
        self,
        module,
        post_q: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if post_q.shape[0] != 1 or post_q.shape[2] != 1:
            raise RuntimeError("page incremental drafting requires batch_size=1 and one query token")
        if not isinstance(self.current_step, PageIncrementalStep):
            raise RuntimeError("start_step must be called before sparse attention")
        if full_k.shape[2] < self.round_start_tokens:
            raise RuntimeError("KV cache is shorter than the declared round-start prefix")

        repeated_k = repeat_kv(full_k, module.num_key_value_groups)
        repeated_v = repeat_kv(full_v, module.num_key_value_groups)
        q_heads = post_q.shape[1]
        num_pages = math.ceil(full_k.shape[2] / self.page_size)
        if self.budget_pages > num_pages:
            raise RuntimeError("page budget exceeds currently available pages")
        prior = self.prior_queries.get(module.layer_idx)
        if prior is None or prior.shape[0] != q_heads:
            raise RuntimeError(f"missing/mismatched endpoint query for layer {module.layer_idx}")
        prior = prior.to(post_q.device)
        keep = torch.zeros((q_heads, full_k.shape[2]), dtype=torch.bool, device=post_q.device)

        for head in range(q_heads):
            selector = (module.layer_idx, head)
            keys_for_head = repeated_k[0, head]
            if selector not in self.static_pages:
                self.static_pages[selector] = self._select_pages(
                    prior[head], keys_for_head, num_pages
                )
            current = set(self.static_pages[selector])
            if len(current) != self.budget_pages:
                raise RuntimeError("endpoint initialization did not produce exactly B unique pages")
            page_scores = _page_scores(
                post_q[0, head, 0], keys_for_head, self.page_size, num_pages
            )
            oracle = tuple(
                sorted(torch.topk(page_scores, k=self.budget_pages).indices.tolist())
            )
            oracle_set = set(oracle)
            replacements = 0
            true_entrants = oracle_set - current

            # Position 1 uses the unmodified endpoint-mean set. From position 2
            # onward, the current real sparse Query supplies the Oracle target
            # before this layer's sparse attention is evaluated.
            if self.current_step.draft_position > 1:
                additions = sorted(
                    true_entrants,
                    key=lambda page: float(page_scores[page]),
                    reverse=True,
                )[: self.update_limit_pages]
                removable = sorted(
                    current - oracle_set,
                    key=lambda page: float(page_scores[page]),
                )
                evictions = removable[: len(additions)]
                current = (current - set(evictions)) | set(additions)
                if len(current) != self.budget_pages:
                    raise RuntimeError("incremental refresh changed the 10% page budget")
                replacements = len(additions)
                self.static_pages[selector] = tuple(sorted(current))

            selected = tuple(sorted(current))
            if any(page >= num_pages for page in selected):
                raise RuntimeError("selection contains a page that is not yet causally available")
            for page in selected:
                begin = page * self.page_size
                keep[head, begin : min(begin + self.page_size, full_k.shape[2])] = True

            self.current_step.static_oracle_recalls.append(
                len(set(selected) & oracle_set) / len(oracle_set)
            )
            self.current_step.static_pages[selector] = selected
            self.current_step.oracle_pages[selector] = oracle
            self.current_step.update_replacements.append(replacements)
            self.current_step.update_limits.append(
                self.update_limit_pages if self.current_step.draft_position > 1 else 0
            )
            self.current_step.true_entrant_counts.append(len(true_entrants))

        scores = torch.matmul(post_q, repeated_k.transpose(2, 3)) * module.scaling
        mask = attention_mask[:, :, :, : scores.shape[-1]] if attention_mask is not None else None
        if mask is not None:
            scores = scores + mask
        full_weights = torch.softmax(scores, dim=-1, dtype=torch.float32)

        for head in range(q_heads):
            selected_mass = float(full_weights[0, head, 0, keep[head]].sum().item())
            padding = (-full_weights.shape[-1]) % self.page_size
            page_mass = F.pad(full_weights[0, head, 0], (0, padding)).reshape(
                -1, self.page_size
            ).sum(dim=-1)[:num_pages]
            oracle_mass = float(torch.topk(page_mass, k=self.budget_pages).values.sum().item())
            self.current_step.attention_recoveries.append(
                selected_mass / oracle_mass if oracle_mass > 0 else 0.0
            )

        sparse_scores = scores.masked_fill(
            ~keep[None, :, None, :], torch.finfo(scores.dtype).min
        )
        sparse_weights = torch.softmax(sparse_scores, dim=-1, dtype=torch.float32).to(
            post_q.dtype
        )
        output = torch.matmul(sparse_weights, repeated_v).transpose(1, 2).contiguous()
        return output, sparse_weights
