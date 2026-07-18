from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ORACLE_B_DIR = REPOSITORY_ROOT / "oracle_b_vs_static"
for path in (REPOSITORY_ROOT, ORACLE_B_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_experiment as base  # type: ignore  # noqa: E402
from controller import SparseStep, TokenIncrementalController  # noqa: E402
from sparse_qwen3 import patch_qwen3_for_sparse_drafting  # noqa: E402


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str
    update_ratio: float = 0.0
    absolute_updates: int = 0
    candidate_factor: float = 0.0
    eviction: str = "oracle"


@dataclass
class UpdateDiagnostics:
    actual_replacements: float = 0.0
    candidate_tokens: float = 0.0
    candidate_recall: float = 0.0
    entrant_precision: float = 0.0
    updated_oracle_overlap: float = 0.0
    qk_similarity_evaluations: float = 0.0
    true_entrant_tokens: float = 0.0
    pending_layers: dict[int, "PendingLayerUpdate"] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class PendingLayerUpdate:
    before: tuple[int, ...]
    updated: tuple[int, ...]
    additions: tuple[int, ...]
    candidates: tuple[int, ...]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Token-level incremental sparse-KV oracle upper-bound experiment.")
    parser.add_argument("--model-path", type=Path, default=Path(r"D:\preExperiments\model\Qwen3-4B"))
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\preExperiments\LongBench\data"))
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--suite", choices=("upper_bound", "candidate", "heuristic", "all"), default="upper_bound")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument(
        "--dataset-allocation",
        default="qasper:10,multifieldqa_en:10,hotpotqa:10,2wikimqa:10,musique:10",
    )
    parser.add_argument("--sample-seed", type=int, default=7)
    parser.add_argument("--sample-policy", choices=("longest", "random"), default="longest")
    parser.add_argument("--min-input-tokens", type=int, default=1024)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--draft-length", type=int, default=8)
    parser.add_argument("--budget-ratio", type=float, default=0.1)
    parser.add_argument("--update-ratios", default="0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--absolute-update-counts", default="")
    parser.add_argument("--candidate-factors", default="0.5,1,2,4")
    parser.add_argument("--heuristic-update-ratio", type=float, default=0.05)
    parser.add_argument("--heuristic-candidate-factor", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--correctness-check-tokens", type=int, default=128)
    parser.add_argument("--dense-check-atol", type=float, default=0.05)
    return parser.parse_args()


def build_methods(args) -> list[MethodSpec]:
    methods = [
        MethodSpec("static_token_10pct", "static"),
        MethodSpec("best_static_token_oracle", "best_static_oracle"),
        MethodSpec("oracle_b_token", "oracle_b"),
    ]
    if args.suite in {"upper_bound", "all"}:
        for ratio in parse_floats(args.update_ratios):
            methods.append(MethodSpec(f"oracle_incremental_r{ratio:g}", "oracle_incremental", update_ratio=ratio))
        for count in parse_ints(args.absolute_update_counts):
            methods.append(MethodSpec(f"oracle_incremental_m{count}", "oracle_incremental", absolute_updates=count))
    if args.suite in {"candidate", "all"}:
        for factor in parse_floats(args.candidate_factors):
            methods.append(
                MethodSpec(
                    f"candidate_oracle_f{factor:g}_r{args.heuristic_update_ratio:g}",
                    "candidate_oracle",
                    update_ratio=args.heuristic_update_ratio,
                    candidate_factor=factor,
                )
            )
    if args.suite in {"heuristic", "all"}:
        for kind, eviction in (
            ("random", "verification"),
            ("score_verify", "verification"),
            ("score_current", "current"),
            ("score_hybrid", "current"),
        ):
            methods.append(
                MethodSpec(
                    f"{kind}_f{args.heuristic_candidate_factor:g}_r{args.heuristic_update_ratio:g}",
                    kind,
                    update_ratio=args.heuristic_update_ratio,
                    candidate_factor=args.heuristic_candidate_factor,
                    eviction=eviction,
                )
            )
    return methods


def dense_probe(model, controller, state, dense_cache, prior_queries, horizon, eos_token_id):
    cache = base.cloned_cropped_cache(dense_cache, int(state.shape[1] - 1))
    controller.begin_probe(int(state.shape[1] - 1), prior_queries)
    current = state[:, -1:]
    tokens = []
    for position in range(1, horizon + 1):
        controller.start_probe_step(position)
        output = model(input_ids=current, past_key_values=cache, use_cache=True)
        controller.finish_probe_step()
        cache = output.past_key_values
        token = int(output.logits[0, -1].argmax().item())
        tokens.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
        current = torch.tensor([[token]], device=state.device, dtype=state.dtype)
    records, best_sets, endpoint_scores, metrics = controller.finish_probe()
    return records, best_sets, endpoint_scores, metrics


def padded_scores(scores: torch.Tensor, length: int) -> torch.Tensor:
    if scores.numel() >= length:
        return scores[:length].float()
    floor = float(scores.min().item()) - max(float(scores.std().item()), 1.0)
    return torch.cat((scores.float(), torch.full((length - scores.numel(),), floor)))


def rank_normalize(scores: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(order.numel(), dtype=torch.float32)
    return 1.0 - ranks / max(order.numel() - 1, 1)


def build_candidate_pool(
    selected: tuple[int, ...],
    verify_scores: torch.Tensor,
    current_scores: torch.Tensor,
    target_count: int,
    rng: random.Random,
) -> tuple[int, ...]:
    universe = current_scores.numel()
    selected_set = set(selected)
    target_count = min(target_count, universe - len(selected_set))
    if target_count <= 0:
        return ()
    verify = padded_scores(verify_scores, universe)
    pool: list[int] = []
    seen = set(selected_set)

    def add(values) -> None:
        for value in values:
            value = int(value)
            if value not in seen and 0 <= value < universe and len(pool) < target_count:
                seen.add(value)
                pool.append(value)

    # Boundary/high-priority verification and current-query tokens.
    quota = max(1, target_count // 3)
    add(torch.argsort(verify, descending=True).tolist()[:quota])
    if len(pool) < target_count:
        add(torch.argsort(current_scores, descending=True).tolist()[:quota])
    # Recent tokens and positions neighboring the selected set.
    if len(pool) < target_count:
        recent = range(max(0, universe - max(1, target_count // 4)), universe)
        add(recent)
    if len(pool) < target_count:
        neighbors = []
        for token in selected:
            neighbors.extend((token - 1, token + 1))
        add(neighbors)
    if len(pool) < target_count:
        remaining = [token for token in range(universe) if token not in seen]
        rng.shuffle(remaining)
        add(remaining)
    return tuple(pool)


def choose_update_count(spec: MethodSpec, token_budget: int) -> int:
    if spec.absolute_updates > 0:
        return min(token_budget, spec.absolute_updates)
    return min(token_budget, max(1, math.ceil(token_budget * spec.update_ratio)))


def update_one_layer(
    spec: MethodSpec,
    current: tuple[int, ...],
    current_scores: torch.Tensor,
    verify_scores: torch.Tensor,
    token_budget: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], UpdateDiagnostics, PendingLayerUpdate]:
    update_limit = choose_update_count(spec, token_budget)
    current_set = set(current)
    candidate_target = math.ceil(token_budget * spec.candidate_factor) if spec.candidate_factor > 0 else 0
    candidates = (
        build_candidate_pool(current, verify_scores, current_scores, candidate_target, rng)
        if candidate_target > 0
        else tuple(token for token in range(current_scores.numel()) if token not in current_set)
    )
    candidate_set = set(candidates)

    if spec.kind == "random":
        choices = list(candidate_set)
        rng.shuffle(choices)
        additions = choices[:update_limit]
    else:
        verify = padded_scores(verify_scores, current_scores.numel())
        if spec.kind == "score_verify":
            entrant_scores = rank_normalize(verify)
        elif spec.kind == "score_current":
            entrant_scores = rank_normalize(current_scores)
        elif spec.kind == "score_hybrid":
            entrant_scores = 0.5 * rank_normalize(verify) + 0.5 * rank_normalize(current_scores)
        else:
            raise ValueError(f"unsupported update method: {spec.kind}")
        additions = sorted(candidate_set, key=lambda token: float(entrant_scores[token]), reverse=True)[:update_limit]

    additions = [token for token in additions if token not in current_set]
    remove_count = len(additions)
    if spec.eviction == "current":
        removable = sorted(current_set, key=lambda token: float(current_scores[token]))
    else:
        verify = padded_scores(verify_scores, current_scores.numel())
        removable = sorted(current_set, key=lambda token: float(verify[token]))
    evictions = removable[:remove_count]
    updated = (current_set - set(evictions)) | set(additions)
    if len(updated) != token_budget:
        raise RuntimeError("incremental update changed the 10% token budget")
    updated_tuple = tuple(sorted(updated))
    pending = PendingLayerUpdate(
        before=tuple(sorted(current_set)),
        updated=updated_tuple,
        additions=tuple(additions),
        candidates=tuple(candidates),
    )
    return updated_tuple, UpdateDiagnostics(
        actual_replacements=len(additions),
        candidate_tokens=len(candidates),
        # Candidate construction ranks the current-query score over the full
        # causal history, even if the entrant rule later evaluates only C_j.
        qk_similarity_evaluations=current_scores.numel(),
    ), pending


def update_all_layers(spec, controller, step: SparseStep, verify_scores, rng):
    diagnostics = []
    pending_layers = {}
    for layer, sparse_layer in step.layers.items():
        updated, item, pending = update_one_layer(
            spec,
            sparse_layer.selected_tokens,
            sparse_layer.update_scores,
            verify_scores[layer],
            controller.token_budget,
            rng,
        )
        controller.set_current_set(layer, updated)
        diagnostics.append(item)
        pending_layers[layer] = pending
    return UpdateDiagnostics(
        actual_replacements=statistics.mean(item.actual_replacements for item in diagnostics),
        candidate_tokens=statistics.mean(item.candidate_tokens for item in diagnostics),
        candidate_recall=statistics.mean(item.candidate_recall for item in diagnostics),
        entrant_precision=statistics.mean(item.entrant_precision for item in diagnostics),
        updated_oracle_overlap=statistics.mean(item.updated_oracle_overlap for item in diagnostics),
        qk_similarity_evaluations=statistics.mean(item.qk_similarity_evaluations for item in diagnostics),
        true_entrant_tokens=statistics.mean(item.true_entrant_tokens for item in diagnostics),
        pending_layers=pending_layers,
    )


def resolve_update_diagnostics(update: UpdateDiagnostics, next_step: SparseStep) -> None:
    """Label a causal update only after the next real sparse Query exists."""
    if not update.pending_layers:
        return
    recalls, precisions, overlaps, entrant_counts = [], [], [], []
    for layer, pending in update.pending_layers.items():
        oracle = set(next_step.layers[layer].oracle_tokens)
        true_entrants = oracle - set(pending.before)
        candidates = set(pending.candidates)
        additions = set(pending.additions)
        recalls.append(len(candidates & true_entrants) / len(true_entrants) if true_entrants else 1.0)
        precisions.append(len(additions & true_entrants) / len(additions) if additions else 1.0)
        overlaps.append(len(set(pending.updated) & oracle) / len(pending.updated))
        entrant_counts.append(len(true_entrants))
    update.candidate_recall = statistics.mean(recalls)
    update.entrant_precision = statistics.mean(precisions)
    update.updated_oracle_overlap = statistics.mean(overlaps)
    update.true_entrant_tokens = statistics.mean(entrant_counts)
    update.pending_layers = {}


def internal_oracle_update_metrics(step: SparseStep) -> UpdateDiagnostics:
    layers = list(step.layers.values())
    return UpdateDiagnostics(
        actual_replacements=statistics.mean(layer.update_replacements for layer in layers),
        candidate_tokens=statistics.mean(layer.update_candidate_tokens for layer in layers),
        candidate_recall=statistics.mean(layer.update_candidate_recall for layer in layers),
        entrant_precision=statistics.mean(layer.update_entrant_precision for layer in layers),
        updated_oracle_overlap=statistics.mean(layer.update_oracle_overlap for layer in layers),
        qk_similarity_evaluations=statistics.mean(
            layer.update_qk_similarity_evaluations for layer in layers
        ),
        true_entrant_tokens=statistics.mean(layer.update_true_entrants for layer in layers),
    )


def set_candidate_pools(spec, controller, step, verify_scores, rng):
    target = math.ceil(controller.token_budget * spec.candidate_factor)
    for layer, sparse_layer in step.layers.items():
        candidates = build_candidate_pool(
            sparse_layer.selected_tokens,
            verify_scores[layer],
            sparse_layer.update_scores,
            target,
            rng,
        )
        controller.set_pending_candidates(
            layer, candidates, qk_similarity_evaluations=sparse_layer.update_scores.numel()
        )


def draft_round(
    model, controller, state, dense_cache, prior_queries, spec,
    best_sets, endpoint_scores, target, eos, rng,
):
    cache = base.cloned_cropped_cache(dense_cache, int(state.shape[1] - 1))
    controller.begin_token_round(
        spec.kind,
        int(state.shape[1] - 1),
        prior_queries,
        best_sets,
        update_ratio=spec.update_ratio,
        absolute_updates=spec.absolute_updates,
    )
    current = state[:, -1:]
    tokens, steps, updates = [], [], []
    verify_scores = endpoint_scores
    for position in range(1, target + 1):
        controller.start_token_step(position)
        output = model(input_ids=current, past_key_values=cache, use_cache=True)
        step = controller.finish_token_step()
        cache = output.past_key_values
        token = int(output.logits[0, -1].argmax().item())
        tokens.append(token)
        steps.append(step)
        updates.append(UpdateDiagnostics())
        if position > 1 and spec.kind in {"oracle_incremental", "candidate_oracle"}:
            updates[position - 2] = internal_oracle_update_metrics(step)
        elif position > 1 and spec.kind in {
            "random", "score_verify", "score_current", "score_hybrid",
        }:
            resolve_update_diagnostics(updates[position - 2], step)
        if eos is not None and token == eos:
            break
        if position < target and spec.kind == "candidate_oracle":
            set_candidate_pools(spec, controller, step, verify_scores, rng)
        elif position < target and spec.kind in {
            "random", "score_verify", "score_current", "score_hybrid",
        }:
            updates[-1] = update_all_layers(
                spec, controller, step, verify_scores, rng
            )
        current = torch.tensor([[token]], device=state.device, dtype=state.dtype)
    return tokens, steps, updates


def mean_layer_metric(step: SparseStep, name: str) -> float:
    return statistics.mean(float(getattr(layer, name)) for layer in step.layers.values())


def run_method(args, sample, model, controller, tokenizer, spec):
    device = torch.device(args.device)
    state = base.clone_to_device(sample.input_ids, device)
    prior_queries, dense_cache = base.initialize_dense_state(model, controller, state)
    generated, rows = [], []
    eos = tokenizer.eos_token_id
    round_id = 0
    while len(generated) < args.max_new_tokens:
        round_id += 1
        remaining = args.max_new_tokens - len(generated)
        target = min(args.draft_length, remaining)
        probe_records, best_sets, endpoint_scores, probe_metrics = dense_probe(
            model, controller, state, dense_cache, prior_queries, target, eos
        )
        target = min(target, len(probe_records))
        rng = random.Random(f"{args.sample_seed}:{sample.dataset_name}:{sample.sample_id}:{spec.name}:{round_id}")
        draft_tokens, steps, updates = draft_round(
            model, controller, state, dense_cache, prior_queries, spec,
            best_sets, endpoint_scores, target, eos, rng,
        )
        accepted, correction, first_queries, terminal_draft_queries, verification_cache = base.verify_round(
            model, controller, state, dense_cache, draft_tokens
        )
        commit = list(draft_tokens[:accepted])
        accepted_eos = accepted == len(draft_tokens) and eos is not None and draft_tokens[-1] == eos
        append_verifier = not accepted_eos and len(commit) < remaining
        if append_verifier:
            commit.append(correction)
        committed_before_verifier = int(state.shape[1] + accepted)
        verification_cache.crop(committed_before_verifier)
        if append_verifier:
            controller.begin_dense({committed_before_verifier})
            correction_tensor = torch.tensor([[correction]], device=device, dtype=state.dtype)
            output = model(input_ids=correction_tensor, past_key_values=verification_cache, use_cache=True)
            dense_cache = output.past_key_values
            terminal_queries = controller.captured_queries.get(committed_before_verifier)
            if terminal_queries is None:
                raise RuntimeError("failed to capture verifier/bonus query")
        else:
            dense_cache = verification_cache
            terminal_queries = terminal_draft_queries
        prior_queries = {
            layer: (0.5 * (first_queries[layer].float() + terminal_queries[layer].float())).to(torch.float16)
            for layer in first_queries
        }
        round_start = int(state.shape[1])
        state = torch.cat((state, torch.tensor([commit], device=device, dtype=state.dtype)), dim=1)
        generated.extend(commit)
        row = {
            "sample_id": sample.sample_id,
            "dataset_name": sample.dataset_name,
            "method": spec.name,
            "method_kind": spec.kind,
            "round_id": round_id,
            "input_length": sample.input_length,
            "round_start_length": round_start,
            "historical_token_budget": controller.token_budget,
            "update_ratio": spec.update_ratio,
            "absolute_update_limit": spec.absolute_updates,
            "candidate_factor": spec.candidate_factor,
            "draft_length": len(draft_tokens),
            "accepted_length": accepted,
            "all_accepted": int(accepted == args.draft_length),
            "zero_accepted": int(accepted == 0),
            "generated_tokens_so_far": len(generated),
            **probe_metrics,
        }
        for position in range(1, args.draft_length + 1):
            if position <= len(steps):
                row[f"selection_recall_pos_{position}"] = mean_layer_metric(steps[position - 1], "selected_oracle_recall")
                row[f"attention_recovery_pos_{position}"] = mean_layer_metric(steps[position - 1], "attention_recovery")
                update = updates[position - 1]
                row[f"replacements_after_pos_{position}"] = update.actual_replacements
                row[f"candidate_tokens_after_pos_{position}"] = update.candidate_tokens
                row[f"candidate_recall_after_pos_{position}"] = update.candidate_recall
                row[f"entrant_precision_after_pos_{position}"] = update.entrant_precision
                row[f"updated_overlap_after_pos_{position}"] = update.updated_oracle_overlap
                row[f"qk_evaluations_after_pos_{position}"] = update.qk_similarity_evaluations
                row[f"true_entrants_after_pos_{position}"] = update.true_entrant_tokens
            else:
                for prefix in (
                    "selection_recall_pos", "attention_recovery_pos", "replacements_after_pos",
                    "candidate_tokens_after_pos", "candidate_recall_after_pos",
                    "entrant_precision_after_pos", "updated_overlap_after_pos",
                    "qk_evaluations_after_pos",
                    "true_entrants_after_pos",
                ):
                    row[f"{prefix}_{position}"] = ""
        rows.append(row)
        if eos is not None and eos in commit:
            break
    return base.MethodResult(method=spec.name, generated_tokens=generated, rounds=rows)


def round_fields(gamma: int) -> list[str]:
    fields = [
        "sample_id", "dataset_name", "method", "method_kind", "round_id", "input_length",
        "round_start_length", "historical_token_budget", "update_ratio", "absolute_update_limit",
        "candidate_factor", "draft_length", "accepted_length", "all_accepted", "zero_accepted",
        "generated_tokens_so_far", "probe_endpoint_coverage_vs_per_query_attention_oracle",
        "probe_best_static_coverage_vs_per_query_attention_oracle",
    ]
    for position in range(1, gamma + 1):
        fields += [
            f"selection_recall_pos_{position}", f"attention_recovery_pos_{position}",
            f"replacements_after_pos_{position}", f"candidate_tokens_after_pos_{position}",
            f"candidate_recall_after_pos_{position}", f"entrant_precision_after_pos_{position}",
            f"updated_overlap_after_pos_{position}", f"qk_evaluations_after_pos_{position}",
            f"true_entrants_after_pos_{position}",
        ]
    return fields


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def summarize(args, methods, samples, all_results):
    rows = [row for _, result in all_results for row in result.rounds]
    write_csv(args.results_dir / "token_incremental_per_round.csv", rows, round_fields(args.draft_length))
    sample_rows, lookup = [], {}
    for sample, result in all_results:
        accepted = [int(row["accepted_length"]) for row in result.rounds]
        item = {
            "sample_id": sample.sample_id,
            "dataset_name": sample.dataset_name,
            "method": result.method,
            "input_length": sample.input_length,
            "rounds": len(accepted),
            "generated_tokens": len(result.generated_tokens),
            "mean_accepted_length": safe_mean(accepted),
            "full_acceptance_rate": sum(value == args.draft_length for value in accepted) / len(accepted),
            "zero_acceptance_rate": sum(value == 0 for value in accepted) / len(accepted),
        }
        lookup[(sample.dataset_name, sample.sample_id, result.method)] = item
        sample_rows.append(item)
    static_name, oracle_name = "static_token_10pct", "oracle_b_token"
    for item in sample_rows:
        key = (item["dataset_name"], item["sample_id"])
        static = lookup[key + (static_name,)]["mean_accepted_length"]
        oracle = lookup[key + (oracle_name,)]["mean_accepted_length"]
        item["delta_vs_static"] = item["mean_accepted_length"] - static
        item["oracle_b_gain_recovery"] = (
            (item["mean_accepted_length"] - static) / (oracle - static) if oracle > static else 0.0
        )
    write_csv(args.results_dir / "token_incremental_per_sample.csv", sample_rows, list(sample_rows[0]))

    position_rows, summary_rows = [], []
    for spec in methods:
        method_rows = [row for row in rows if row["method"] == spec.name]
        accepted = [int(row["accepted_length"]) for row in method_rows]
        macro = safe_mean(
            item["mean_accepted_length"] for item in sample_rows if item["method"] == spec.name
        )
        replacements = [
            float(row[f"replacements_after_pos_{position}"])
            for row in method_rows for position in range(1, args.draft_length)
            if row[f"replacements_after_pos_{position}"] != ""
        ]
        replacements_per_round = [
            sum(
                float(row[f"replacements_after_pos_{position}"])
                for position in range(1, args.draft_length)
                if row[f"replacements_after_pos_{position}"] != ""
            )
            for row in method_rows
        ]
        candidate_recalls = [
            float(row[f"candidate_recall_after_pos_{position}"])
            for row in method_rows for position in range(1, args.draft_length)
            if row[f"candidate_tokens_after_pos_{position}"] not in ("", 0, 0.0)
            and row[f"true_entrants_after_pos_{position}"] not in ("", 0, 0.0)
        ]
        candidate_sizes = [
            float(row[f"candidate_tokens_after_pos_{position}"])
            for row in method_rows for position in range(1, args.draft_length)
            if row[f"candidate_tokens_after_pos_{position}"] not in ("", 0, 0.0)
        ]
        qk_evaluations = [
            float(row[f"qk_evaluations_after_pos_{position}"])
            for row in method_rows for position in range(1, args.draft_length)
            if row[f"qk_evaluations_after_pos_{position}"] != ""
        ]
        summary_rows.append({
            "method": spec.name,
            "method_kind": spec.kind,
            "update_ratio": spec.update_ratio,
            "absolute_update_limit": spec.absolute_updates,
            "candidate_factor": spec.candidate_factor,
            "mean_accepted_length": safe_mean(accepted),
            "macro_by_sample_mean_accepted_length": macro,
            "full_acceptance_rate": sum(value == args.draft_length for value in accepted) / len(accepted),
            "zero_acceptance_rate": sum(value == 0 for value in accepted) / len(accepted),
            "total_rounds": len(accepted),
            "mean_actual_replacements_per_step": safe_mean(replacements),
            "mean_total_replacements_per_round": safe_mean(replacements_per_round),
            "mean_historical_token_budget": safe_mean(float(row["historical_token_budget"]) for row in method_rows),
            "mean_candidate_recall": safe_mean(candidate_recalls),
            "mean_candidate_tokens_per_step": safe_mean(candidate_sizes),
            "mean_qk_similarity_evaluations_per_step": safe_mean(qk_evaluations),
            "predictor_parameters": 0,
            "mlp_calls_per_step": 0,
        })
        for position in range(1, args.draft_length + 1):
            eligible = [
                row for row in method_rows
                if int(row["draft_length"]) >= position and int(row["accepted_length"]) >= position - 1
            ]
            count = sum(int(row["accepted_length"]) >= position for row in eligible)
            position_rows.append({
                "method": spec.name,
                "draft_position": position,
                "eligible_rounds": len(eligible),
                "accepted_rounds": count,
                "conditional_acceptance_rate": count / len(eligible) if eligible else 0.0,
            })
    static_macro = next(row["macro_by_sample_mean_accepted_length"] for row in summary_rows if row["method"] == static_name)
    oracle_macro = next(row["macro_by_sample_mean_accepted_length"] for row in summary_rows if row["method"] == oracle_name)
    for row in summary_rows:
        row["oracle_b_gain_recovery"] = (
            (row["macro_by_sample_mean_accepted_length"] - static_macro) / (oracle_macro - static_macro)
            if oracle_macro > static_macro else 0.0
        )
    write_csv(args.results_dir / "token_incremental_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(args.results_dir / "position_acceptance_rate.csv", position_rows, list(position_rows[0]))
    distribution_rows = []
    for accepted_length in range(args.draft_length + 1):
        item = {"accepted_length": accepted_length}
        for spec in methods:
            values = [int(row["accepted_length"]) for row in rows if row["method"] == spec.name]
            count = sum(value == accepted_length for value in values)
            item[f"{spec.name}_rounds"] = count
            item[f"{spec.name}_ratio"] = count / len(values) if values else 0.0
        distribution_rows.append(item)
    write_csv(
        args.results_dir / "acceptance_length_distribution.csv",
        distribution_rows,
        list(distribution_rows[0]),
    )
    payload = {
        "suite": args.suite,
        "num_samples": len(samples),
        "static_macro_mean_accepted_length": static_macro,
        "oracle_b_macro_mean_accepted_length": oracle_macro,
        "methods": summary_rows,
    }
    (args.results_dir / "token_incremental_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(args.results_dir, summary_rows, position_rows)
    write_plots(args.results_dir, summary_rows, position_rows, rows, args.draft_length)


def write_report(results_dir, summary_rows, position_rows):
    lines = [
        "# Token-level Incremental KV Selection", "", "## Acceptance and update upper bounds", "",
        "| Method | Mean accepted | Full-8 rate | Mean replacements | Candidate recall | Oracle-B gain recovery |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['macro_by_sample_mean_accepted_length']:.4f} | "
            f"{row['full_acceptance_rate']:.4f} | {row['mean_actual_replacements_per_step']:.2f} | "
            f"{row['mean_candidate_recall']:.4f} | {row['oracle_b_gain_recovery']:.2%} |"
        )
    lines += ["", "## Conditional acceptance", ""]
    for row in summary_rows:
        values = [
            item["conditional_acceptance_rate"] for item in position_rows if item["method"] == row["method"]
        ]
        lines.append(f"- `{row['method']}`: " + ", ".join(f"P{idx + 1}={value:.4f}" for idx, value in enumerate(values)))
    lines += [
        "", "## Gate for the predictor stage", "",
        "Proceed to Token Entrant Predictor training only if a small Oracle Incremental budget recovers a substantial "
        "fraction of the Static-to-Oracle-B acceptance gap and the causal candidate pool retains sufficient entrant recall.",
    ]
    (results_dir / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(results_dir, summary_rows, position_rows, round_rows, gamma):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (results_dir / "plots_skipped.txt").write_text("matplotlib is not installed\n", encoding="utf-8")
        return
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 5))
    for method in sorted({row["method"] for row in position_rows}):
        data = [row for row in position_rows if row["method"] == method]
        plt.plot(
            [row["draft_position"] for row in data],
            [row["conditional_acceptance_rate"] for row in data],
            marker="o",
            label=method,
        )
    plt.xlabel("Draft position")
    plt.ylabel("Conditional acceptance rate")
    plt.xticks(range(1, gamma + 1))
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plot_dir / "conditional_acceptance_by_position.png", dpi=160)
    plt.close()

    oracle_updates = [row for row in summary_rows if row["method_kind"] == "oracle_incremental"]
    if oracle_updates:
        oracle_updates.sort(key=lambda row: row["mean_actual_replacements_per_step"])
        plt.figure(figsize=(7, 5))
        plt.plot(
            [row["mean_actual_replacements_per_step"] for row in oracle_updates],
            [row["macro_by_sample_mean_accepted_length"] for row in oracle_updates],
            marker="o",
        )
        plt.xlabel("Mean token replacements per step")
        plt.ylabel("Mean accepted length (sample macro)")
        plt.tight_layout()
        plt.savefig(plot_dir / "update_amount_vs_accepted_length.png", dpi=160)
        plt.close()

    points = [row for row in summary_rows if row["mean_candidate_recall"] > 0]
    if points:
        plt.figure(figsize=(7, 5))
        for row in points:
            plt.scatter(row["mean_candidate_recall"], row["macro_by_sample_mean_accepted_length"])
            plt.annotate(row["method"], (row["mean_candidate_recall"], row["macro_by_sample_mean_accepted_length"]), fontsize=6)
        plt.xlabel("Entrant candidate recall")
        plt.ylabel("Mean accepted length")
        plt.tight_layout()
        plt.savefig(plot_dir / "entrant_recall_vs_accepted_length.png", dpi=160)
        plt.close()

    plt.figure(figsize=(7, 5))
    for method in sorted({row["method"] for row in round_rows}):
        data = [row for row in round_rows if row["method"] == method]
        overlaps, accepted = [], []
        for row in data:
            values = [
                float(row[f"selection_recall_pos_{position}"])
                for position in range(1, gamma + 1)
                if row[f"selection_recall_pos_{position}"] != ""
            ]
            if values:
                overlaps.append(statistics.mean(values))
                accepted.append(int(row["accepted_length"]))
        plt.scatter(overlaps, accepted, s=7, alpha=0.25, label=method)
    plt.xlabel("Mean selection overlap with Oracle B")
    plt.ylabel("Accepted length")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(plot_dir / "selection_overlap_vs_accepted_length.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    methods = build_methods(args)
    allocation = base.parse_allocation(args.dataset_allocation)
    if sum(allocation.values()) != args.num_samples:
        raise ValueError("dataset-allocation must sum to num-samples")
    if args.draft_length != 8:
        raise ValueError("this pre-experiment fixes draft-length at 8")
    base.set_deterministic_seed(args.sample_seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    samples = base.prepare_samples(args, tokenizer, allocation)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, attn_implementation="sdpa", dtype=dtype, low_cpu_mem_usage=True
    ).to(args.device).eval()
    controller = TokenIncrementalController(args.budget_ratio)
    check_length = min(args.correctness_check_tokens, samples[0].input_length)
    check_ids = samples[0].input_ids[:, -check_length:].to(args.device)
    with torch.inference_mode():
        reference = model(input_ids=check_ids, use_cache=False).logits[:, -1].float().cpu()
    patch_qwen3_for_sparse_drafting(model, controller)
    with torch.inference_mode():
        controller.begin_dense()
        patched = model(input_ids=check_ids, use_cache=False).logits[:, -1].float().cpu()
    max_error = float((reference - patched).abs().max().item())
    top1_equal = bool(reference.argmax().item() == patched.argmax().item())
    if max_error > args.dense_check_atol or not top1_equal:
        raise RuntimeError(f"dense patch check failed: max_error={max_error}, top1_equal={top1_equal}")
    config = vars(args).copy()
    config.update({
        "model_path": str(args.model_path), "data_root": str(args.data_root),
        "results_dir": str(args.results_dir), "methods": [spec.__dict__ for spec in methods],
        "selection_unit": "historical token", "selection_granularity": "one token-position set per layer shared by query heads",
        "current_query_token": "always retained outside historical KV-cache budget",
        "token_budget_rounding": "ceil(0.1 * round-start historical tokens), fixed within a draft round",
        "oracle_selection": "mean post-RoPE QK across correctly GQA-mapped query heads",
        "incremental_oracle_target": "current real sparse Query after it is formed and before sparse attention",
        "heuristic_diagnostic_target": "next real sparse Query; causal updates are labeled one step later",
        "qk_cost_proxy": "counts causal current-query full-history scoring used to build candidate pools; Oracle-label scans are excluded",
        "implementation_stage": "stage_1_oracle_and_candidate_upper_bound_gate",
        "learned_predictor_enabled": False,
        "oracle_probe_cost_in_primary_comparison": False,
        "dense_patch_check": {"max_abs_logit_error": max_error, "top1_equal": top1_equal},
        "sample_ids": [sample.sample_id for sample in samples],
    })
    (args.results_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    all_results = []
    with torch.inference_mode():
        for sample in samples:
            outputs = []
            for spec in methods:
                base.set_deterministic_seed(args.sample_seed)
                result = run_method(args, sample, model, controller, tokenizer, spec)
                all_results.append((sample, result))
                outputs.append(result.generated_tokens)
            if any(output != outputs[0] for output in outputs[1:]):
                raise RuntimeError(f"dense-equivalence failure on {sample.sample_id}")
    summarize(args, methods, samples, all_results)

if __name__ == "__main__":
    main()
