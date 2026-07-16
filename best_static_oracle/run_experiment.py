from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ORACLE_B_DIR = REPOSITORY_ROOT / "oracle_b_vs_static"
for path in (REPOSITORY_ROOT, ORACLE_B_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_experiment as base  # type: ignore  # noqa: E402
from controller import BestStaticOracleController  # noqa: E402
from sparse_qwen3 import (  # noqa: E402
    StepDiagnostics,
    adjacent_oracle_overlap,
    endpoint_overlap,
    patch_qwen3_for_sparse_drafting,
)


METHODS = ("static", "best_static_oracle", "oracle_b")
LABELS = {
    "static": "Static endpoint mean",
    "best_static_oracle": "Best Static Oracle",
    "oracle_b": "Oracle B",
}
PROBE_FIELDS = (
    "probe_endpoint_coverage_vs_per_query_attention_oracle",
    "probe_best_static_coverage_vs_per_query_attention_oracle",
    "probe_endpoint_fraction_of_eligible_attention",
    "probe_best_static_fraction_of_eligible_attention",
    "probe_best_static_absolute_coverage_gain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare endpoint static, Best Static Oracle, and Oracle B.")
    parser.add_argument("--model-path", type=Path, default=Path(r"D:\preExperiments\model\Qwen3-4B"))
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\preExperiments\LongBench"))
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
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
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--budget-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--correctness-check-tokens", type=int, default=128)
    parser.add_argument("--dense-check-atol", type=float, default=0.05)
    return parser.parse_args()


def probe_best_static_pages(
    model,
    controller: BestStaticOracleController,
    state: torch.Tensor,
    dense_cache,
    prior_queries: dict[int, torch.Tensor],
    horizon: int,
    eos_token_id: int | None,
):
    cache = base.cloned_cropped_cache(dense_cache, int(state.shape[1] - 1))
    controller.begin_best_static_probe(int(state.shape[1]), prior_queries)
    current = state[:, -1:]
    probe_tokens = []
    for _ in range(horizon):
        output = model(input_ids=current, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        token = int(output.logits[0, -1].argmax().item())
        probe_tokens.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
        current = torch.tensor([[token]], device=state.device, dtype=state.dtype)
    pages, metrics = controller.finish_best_static_probe()
    metrics["probe_query_count"] = len(probe_tokens)
    return pages, metrics


def draft_round(
    model,
    controller: BestStaticOracleController,
    state: torch.Tensor,
    dense_cache,
    method: str,
    prior_queries: dict[int, torch.Tensor],
    draft_target: int,
    eos_token_id: int | None,
    best_static_pages=None,
) -> tuple[list[int], list[StepDiagnostics]]:
    cache = base.cloned_cropped_cache(dense_cache, int(state.shape[1] - 1))
    controller.begin_sparse_with_pages(
        method, int(state.shape[1]), prior_queries, best_static_pages
    )
    current = state[:, -1:]
    tokens, diagnostics = [], []
    for position in range(1, draft_target + 1):
        controller.start_step(position)
        output = model(input_ids=current, past_key_values=cache, use_cache=True)
        diagnostics.append(controller.finish_step())
        cache = output.past_key_values
        token = int(output.logits[0, -1].argmax().item())
        tokens.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
        current = torch.tensor([[token]], device=state.device, dtype=state.dtype)
    return tokens, diagnostics


def diagnostic_fields(diagnostics: list[StepDiagnostics], gamma: int) -> dict:
    row = {}
    for position in range(1, gamma + 1):
        diag = diagnostics[position - 1] if position <= len(diagnostics) else None
        row[f"fixed_reference_recall_of_query_pages_pos_{position}"] = (
            diag.mean_static_oracle_recall if diag else ""
        )
        row[f"attention_recovery_pos_{position}"] = diag.mean_attention_recovery if diag else ""
    for position in range(1, gamma):
        row[f"oracle_overlap_pos_{position}_{position + 1}"] = (
            adjacent_oracle_overlap(diagnostics[position - 1], diagnostics[position])
            if position < len(diagnostics)
            else ""
        )
    row["oracle_overlap_pos_1_last"] = (
        endpoint_overlap(diagnostics[0], diagnostics[-1]) if len(diagnostics) >= 2 else ""
    )
    return row


def run_method(args, sample, model, controller, tokenizer, method: str):
    device = torch.device(args.device)
    state = base.clone_to_device(sample.input_ids, device)
    prior_queries, dense_cache = base.initialize_dense_state(model, controller, state)
    generated, rounds = [], []
    eos = tokenizer.eos_token_id
    round_id = 0

    while len(generated) < args.max_new_tokens:
        round_id += 1
        remaining = args.max_new_tokens - len(generated)
        target = min(args.draft_length, remaining)
        pages, probe_metrics = None, {field: "" for field in PROBE_FIELDS}
        probe_metrics["probe_query_count"] = ""
        if method == "best_static_oracle":
            pages, probe_metrics = probe_best_static_pages(
                model, controller, state, dense_cache, prior_queries, target, eos
            )
        draft_tokens, diagnostics = draft_round(
            model,
            controller,
            state,
            dense_cache,
            method,
            prior_queries,
            target,
            eos,
            pages,
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
            output = model(
                input_ids=correction_tensor,
                past_key_values=verification_cache,
                use_cache=True,
            )
            dense_cache = output.past_key_values
            terminal_queries = controller.captured_queries.get(committed_before_verifier)
            if terminal_queries is None:
                raise RuntimeError("failed to capture verifier/bonus endpoint query")
        else:
            dense_cache = verification_cache
            terminal_queries = terminal_draft_queries
        prior_queries = {
            layer: (0.5 * (first_queries[layer].float() + terminal_queries[layer].float())).to(torch.float16)
            for layer in first_queries
        }

        round_start_length = int(state.shape[1])
        state = torch.cat((state, torch.tensor([commit], device=device, dtype=state.dtype)), dim=1)
        generated.extend(commit)
        row = {
            "sample_id": sample.sample_id,
            "dataset_name": sample.dataset_name,
            "method": method,
            "input_length": sample.input_length,
            "round_id": round_id,
            "round_start_length": round_start_length,
            "round_start_pages": math.ceil(round_start_length / args.page_size),
            "budget_pages": controller.budget_pages,
            "draft_length": len(draft_tokens),
            "accepted_length": accepted,
            "all_accepted": int(accepted == args.draft_length),
            "zero_accepted": int(accepted == 0),
            "generated_tokens_so_far": len(generated),
            **probe_metrics,
            **diagnostic_fields(diagnostics, args.draft_length),
        }
        rounds.append(row)
        if eos is not None and eos in commit:
            break
    return base.MethodResult(method=method, generated_tokens=generated, rounds=rounds)


def round_fields(gamma: int) -> list[str]:
    fields = [
        "sample_id", "dataset_name", "method", "input_length", "round_id",
        "round_start_length", "round_start_pages", "budget_pages", "draft_length",
        "accepted_length", "all_accepted", "zero_accepted", "generated_tokens_so_far",
        "probe_query_count", *PROBE_FIELDS,
    ]
    fields += [
        f"fixed_reference_recall_of_query_pages_pos_{position}"
        for position in range(1, gamma + 1)
    ]
    fields += [f"oracle_overlap_pos_{position}_{position + 1}" for position in range(1, gamma)]
    fields += ["oracle_overlap_pos_1_last"]
    fields += [f"attention_recovery_pos_{position}" for position in range(1, gamma + 1)]
    return fields


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def summarize(args, samples, all_results) -> None:
    result_dir = args.results_dir
    round_rows = [row for _, result in all_results for row in result.rounds]
    write_csv(result_dir / "best_static_oracle_per_round.csv", round_rows, round_fields(args.draft_length))

    sample_rows, lookup = [], {}
    for sample, result in all_results:
        accepted = [int(row["accepted_length"]) for row in result.rounds]
        item = {
            "sample_id": sample.sample_id,
            "dataset_name": sample.dataset_name,
            "method": result.method,
            "input_length": sample.input_length,
            "initial_kv_pages": math.ceil(sample.input_length / args.page_size),
            "speculative_rounds": len(accepted),
            "total_generated_tokens": len(result.generated_tokens),
            "total_accepted_draft_tokens": sum(accepted),
            "mean_accepted_length": safe_mean(accepted),
            "full_accept_rounds": sum(value == args.draft_length for value in accepted),
            "zero_accept_rounds": sum(value == 0 for value in accepted),
        }
        lookup[(sample.dataset_name, sample.sample_id, result.method)] = item
    for sample in samples:
        key = (sample.dataset_name, sample.sample_id)
        static = lookup[key + ("static",)]["mean_accepted_length"]
        best = lookup[key + ("best_static_oracle",)]["mean_accepted_length"]
        oracle = lookup[key + ("oracle_b",)]["mean_accepted_length"]
        for method in METHODS:
            item = dict(lookup[key + (method,)])
            item["delta_vs_static"] = item["mean_accepted_length"] - static
            item["gap_to_oracle_b"] = oracle - item["mean_accepted_length"]
            item["fraction_of_oracle_b_gain_closed"] = (
                (item["mean_accepted_length"] - static) / (oracle - static)
                if oracle > static
                else 0.0
            )
            sample_rows.append(item)
    write_csv(result_dir / "best_static_oracle_per_sample.csv", sample_rows, list(sample_rows[0]))

    distribution, positions = [], []
    for length in range(args.draft_length + 1):
        item = {"accepted_length": length}
        for method in METHODS:
            values = [int(row["accepted_length"]) for row in round_rows if row["method"] == method]
            count = sum(value == length for value in values)
            item[f"{method}_rounds"] = count
            item[f"{method}_ratio"] = count / len(values) if values else 0.0
        distribution.append(item)
    write_csv(result_dir / "acceptance_length_distribution.csv", distribution, list(distribution[0]))
    for method in METHODS:
        rows = [row for row in round_rows if row["method"] == method]
        for position in range(1, args.draft_length + 1):
            eligible = [
                row for row in rows
                if int(row["draft_length"]) >= position and int(row["accepted_length"]) >= position - 1
            ]
            count = sum(int(row["accepted_length"]) >= position for row in eligible)
            positions.append({
                "method": method,
                "draft_position": position,
                "eligible_rounds": len(eligible),
                "accepted_rounds": count,
                "conditional_acceptance_rate": count / len(eligible) if eligible else 0.0,
            })
    write_csv(result_dir / "position_acceptance_rate.csv", positions, list(positions[0]))

    overall = {}
    for method in METHODS:
        rows = [row for row in round_rows if row["method"] == method]
        accepted = [int(row["accepted_length"]) for row in rows]
        overall[method] = {
            "mean_accepted_length": safe_mean(accepted),
            "full_acceptance_rate": sum(value == args.draft_length for value in accepted) / len(accepted),
            "zero_acceptance_rate": sum(value == 0 for value in accepted) / len(accepted),
            "total_rounds": len(rows),
            "mean_attention_recovery": safe_mean(
                float(row[f"attention_recovery_pos_{position}"])
                for row in rows for position in range(1, args.draft_length + 1)
                if row[f"attention_recovery_pos_{position}"] != ""
            ),
        }
        overall[method]["macro_by_sample_mean_accepted_length"] = safe_mean(
            float(row["mean_accepted_length"])
            for row in sample_rows if row["method"] == method
        )
    static, best, oracle = (
        overall["static"]["mean_accepted_length"],
        overall["best_static_oracle"]["mean_accepted_length"],
        overall["oracle_b"]["mean_accepted_length"],
    )
    best_rows = [row for row in round_rows if row["method"] == "best_static_oracle"]
    macro_static = overall["static"]["macro_by_sample_mean_accepted_length"]
    macro_best = overall["best_static_oracle"]["macro_by_sample_mean_accepted_length"]
    macro_oracle = overall["oracle_b"]["macro_by_sample_mean_accepted_length"]
    paired = []
    for sample in samples:
        key = (sample.dataset_name, sample.sample_id)
        paired.append({
            "best_minus_static": lookup[key + ("best_static_oracle",)]["mean_accepted_length"]
            - lookup[key + ("static",)]["mean_accepted_length"],
            "oracle_minus_best": lookup[key + ("oracle_b",)]["mean_accepted_length"]
            - lookup[key + ("best_static_oracle",)]["mean_accepted_length"],
        })
    payload = {
        "methods": overall,
        "best_static_absolute_gain_vs_static": best - static,
        "oracle_b_absolute_gain_vs_static": oracle - static,
        "best_static_gap_to_oracle_b": oracle - best,
        "best_static_fraction_of_oracle_b_gain_closed": (best - static) / (oracle - static) if oracle > static else 0.0,
        "paired_macro": {
            "static_mean_accepted_length": macro_static,
            "best_static_mean_accepted_length": macro_best,
            "oracle_b_mean_accepted_length": macro_oracle,
            "best_static_gain_vs_static": macro_best - macro_static,
            "best_static_gap_to_oracle_b": macro_oracle - macro_best,
            "best_static_fraction_of_oracle_b_gain_closed": (
                (macro_best - macro_static) / (macro_oracle - macro_static)
                if macro_oracle > macro_static else 0.0
            ),
            "best_static_better_than_static_samples": sum(item["best_minus_static"] > 1e-12 for item in paired),
            "best_static_equal_to_static_samples": sum(abs(item["best_minus_static"]) <= 1e-12 for item in paired),
            "best_static_worse_than_static_samples": sum(item["best_minus_static"] < -1e-12 for item in paired),
            "oracle_b_better_than_best_static_samples": sum(item["oracle_minus_best"] > 1e-12 for item in paired),
            "oracle_b_equal_to_best_static_samples": sum(abs(item["oracle_minus_best"]) <= 1e-12 for item in paired),
            "oracle_b_worse_than_best_static_samples": sum(item["oracle_minus_best"] < -1e-12 for item in paired),
        },
        "best_static_probe": {
            field: safe_mean(float(row[field]) for row in best_rows if row[field] != "")
            for field in PROBE_FIELDS
        },
        "num_samples": len(samples),
        "input_tokens": {
            "mean": safe_mean(sample.input_length for sample in samples),
            "median": statistics.median(sample.input_length for sample in samples),
            "min": min(sample.input_length for sample in samples),
            "max": max(sample.input_length for sample in samples),
        },
    }
    (result_dir / "best_static_oracle_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_report(result_dir, payload, positions)


def write_report(result_dir: Path, payload: dict, positions: list[dict]) -> None:
    lines = [
        "# Best Static Oracle Experiment", "", "## Acceptance", "",
        "| Method | Mean accepted | Full-8 rate | Zero rate | Rounds | Attention recovery |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        item = payload["methods"][method]
        lines.append(
            f"| {LABELS[method]} | {item['mean_accepted_length']:.4f} | "
            f"{item['full_acceptance_rate']:.4f} | {item['zero_acceptance_rate']:.4f} | "
            f"{item['total_rounds']} | {item['mean_attention_recovery']:.4f} |"
        )
    lines += [
        "",
        f"Best Static gain over endpoint static: `{payload['best_static_absolute_gain_vs_static']:.4f}`.",
        f"Best Static gap to Oracle B: `{payload['best_static_gap_to_oracle_b']:.4f}`.",
        f"Fraction of Oracle-B acceptance gain closed by Best Static (round-micro): "
        f"`{payload['best_static_fraction_of_oracle_b_gain_closed']:.2%}`.",
        f"Fraction closed using paired sample-macro means: "
        f"`{payload['paired_macro']['best_static_fraction_of_oracle_b_gain_closed']:.2%}`.",
        "", "## Conditional acceptance", "",
        "| Method | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 | Pos 8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        values = [row["conditional_acceptance_rate"] for row in positions if row["method"] == method]
        lines.append(f"| {LABELS[method]} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    probe = payload["best_static_probe"]
    lines += [
        "", "## Dense future-query probe", "",
        f"- Endpoint coverage / per-query attention oracle: "
        f"`{probe['probe_endpoint_coverage_vs_per_query_attention_oracle']:.4f}`",
        f"- Best Static coverage / per-query attention oracle: "
        f"`{probe['probe_best_static_coverage_vs_per_query_attention_oracle']:.4f}`",
        "", "## Decision", "",
        "If Best Static closes most of Oracle B's acceptance gain, a predicted shared future-query KV set is sufficient. "
        "If a substantial acceptance gap remains, query-specific refresh or incremental routing is necessary.",
    ]
    (result_dir / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    allocation = base.parse_allocation(args.dataset_allocation)
    if sum(allocation.values()) != args.num_samples:
        raise ValueError("dataset-allocation must sum to num-samples")
    if not (0.0 < args.budget_ratio <= 1.0):
        raise ValueError("budget-ratio must be in (0, 1]")
    base.set_deterministic_seed(args.sample_seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    samples = base.prepare_samples(args, tokenizer, allocation)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    controller = BestStaticOracleController(args.page_size, args.budget_ratio)

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
        "model_path": str(args.model_path),
        "data_root": str(args.data_root),
        "results_dir": str(args.results_dir),
        "methods": list(METHODS),
        "best_static_definition": "Top-B pages by summed dense attention mass over up to 8 future causal decision queries",
        "selection_universe": "pages present at round start; a selected partial final page may receive later tokens",
        "selection_granularity": "per layer and query head, using only its corresponding GQA KV head",
        "budget_rounding": "max(1, ceil(0.1 * round_start_pages))",
        "page_score_for_endpoint_and_oracle_b": "max token post-RoPE QK within page",
        "forced_pages": "none",
        "oracle_probe_and_selection_time_in_primary_comparison": False,
        "dense_patch_check": {"tokens": check_length, "max_abs_logit_error": max_error, "top1_equal": top1_equal},
        "sample_ids": [sample.sample_id for sample in samples],
    })
    (args.results_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_results = []
    with torch.inference_mode():
        for sample in samples:
            outputs = []
            for method in METHODS:
                base.set_deterministic_seed(args.sample_seed)
                result = run_method(args, sample, model, controller, tokenizer, method)
                all_results.append((sample, result))
                outputs.append(result.generated_tokens)
            if any(output != outputs[0] for output in outputs[1:]):
                raise RuntimeError(f"dense-equivalence failure on {sample.sample_id}")
    summarize(args, samples, all_results)

if __name__ == "__main__":
    main()
