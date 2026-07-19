from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ORACLE_B_DIR = REPOSITORY_ROOT / "oracle_b_vs_static"
for path in (REPOSITORY_ROOT, ORACLE_B_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_experiment as base  # type: ignore  # noqa: E402
from controller import PageIncrementalController, PageIncrementalStep  # noqa: E402
from sparse_qwen3 import adjacent_oracle_overlap, endpoint_overlap, patch_qwen3_for_sparse_drafting  # noqa: E402


UPDATE_RATIOS = (0.01, 0.05, 0.10, 0.20)
REFERENCE_METHODS = ("static", "best_static_oracle", "oracle_b")
REFERENCE_LABELS = {
    "static": "Static endpoint mean (reused)",
    "best_static_oracle": "Best Static Oracle (reused)",
    "oracle_b": "Oracle B (reused)",
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    update_ratio: float


@dataclass
class ReferenceResults:
    root: Path
    config: dict
    summary: dict
    per_sample: dict[tuple[str, str, str], dict]
    positions: list[dict]
    distribution: list[dict]


def methods() -> list[MethodSpec]:
    return [MethodSpec(f"page_incremental_r{ratio:g}", ratio) for ratio in UPDATE_RATIOS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Page-level 1/5/10/20% Oracle Incremental KV-selection sweep."
    )
    parser.add_argument("--model-path", type=Path, default=Path(r"D:\preExperiments\model\Qwen3-4B"))
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\preExperiments\LongBench\data"))
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument(
        "--reference-results-dir",
        type=Path,
        default=REPOSITORY_ROOT / "best_static_oracle" / "results" / "formal_50",
    )
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


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_reference_results(root: Path) -> ReferenceResults:
    required = {
        "experiment_config.json",
        "best_static_oracle_summary.json",
        "best_static_oracle_per_sample.csv",
        "position_acceptance_rate.csv",
        "acceptance_length_distribution.csv",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(f"reference result directory is incomplete: {missing}")
    config = json.loads((root / "experiment_config.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "best_static_oracle_summary.json").read_text(encoding="utf-8"))
    per_sample_rows = read_csv(root / "best_static_oracle_per_sample.csv")
    per_sample = {
        (row["dataset_name"], row["sample_id"], row["method"]): row
        for row in per_sample_rows
    }
    return ReferenceResults(
        root=root,
        config=config,
        summary=summary,
        per_sample=per_sample,
        positions=read_csv(root / "position_acceptance_rate.csv"),
        distribution=read_csv(root / "acceptance_length_distribution.csv"),
    )


def validate_reference(args, samples, reference: ReferenceResults) -> None:
    expected = {
        "num_samples": args.num_samples,
        "dataset_allocation": args.dataset_allocation,
        "sample_seed": args.sample_seed,
        "sample_policy": args.sample_policy,
        "min_input_tokens": args.min_input_tokens,
        "max_context_tokens": args.max_context_tokens,
        "max_new_tokens": args.max_new_tokens,
        "draft_length": args.draft_length,
        "page_size": args.page_size,
        "budget_ratio": args.budget_ratio,
        "dtype": args.dtype,
    }
    mismatches = {
        key: {"reference": reference.config.get(key), "current": value}
        for key, value in expected.items()
        if reference.config.get(key) != value
    }
    sample_ids = [sample.sample_id for sample in samples]
    if reference.config.get("sample_ids") != sample_ids:
        mismatches["sample_ids"] = "reference and current prepared samples differ"
    missing_rows = [
        (sample.dataset_name, sample.sample_id, method)
        for sample in samples
        for method in REFERENCE_METHODS
        if (sample.dataset_name, sample.sample_id, method) not in reference.per_sample
    ]
    if missing_rows:
        mismatches["per_sample_rows"] = f"missing {len(missing_rows)} reference rows"
    if mismatches:
        raise ValueError(
            "reference results are not comparable with this run:\n"
            + json.dumps(mismatches, indent=2)
        )


def selection_overlap(left: PageIncrementalStep, right: PageIncrementalStep) -> float:
    selectors = sorted(set(left.static_pages) & set(right.static_pages))
    values = []
    for selector in selectors:
        a, b = set(left.static_pages[selector]), set(right.static_pages[selector])
        values.append(len(a & b) / len(a | b) if a or b else 1.0)
    return statistics.mean(values) if values else float("nan")


def draft_round(
    model,
    controller: PageIncrementalController,
    state: torch.Tensor,
    dense_cache,
    prior_queries: dict[int, torch.Tensor],
    spec: MethodSpec,
    target: int,
    eos_token_id: int | None,
) -> tuple[list[int], list[PageIncrementalStep]]:
    cache = base.cloned_cropped_cache(dense_cache, int(state.shape[1] - 1))
    controller.begin_incremental(spec.update_ratio, int(state.shape[1]), prior_queries)
    current = state[:, -1:]
    tokens, diagnostics = [], []
    for position in range(1, target + 1):
        controller.start_step(position)
        output = model(input_ids=current, past_key_values=cache, use_cache=True)
        step = controller.finish_step()
        if not isinstance(step, PageIncrementalStep):
            raise RuntimeError("page incremental controller returned unexpected diagnostics")
        diagnostics.append(step)
        cache = output.past_key_values
        token = int(output.logits[0, -1].argmax().item())
        tokens.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
        current = torch.tensor([[token]], device=state.device, dtype=state.dtype)
    return tokens, diagnostics


def diagnostic_fields(diagnostics: list[PageIncrementalStep], gamma: int) -> dict:
    row = {}
    for position in range(1, gamma + 1):
        step = diagnostics[position - 1] if position <= len(diagnostics) else None
        row[f"selection_recall_of_oracle_b_pos_{position}"] = (
            step.mean_static_oracle_recall if step else ""
        )
        row[f"attention_recovery_pos_{position}"] = (
            step.mean_attention_recovery if step else ""
        )
        row[f"replacements_pos_{position}"] = step.mean_update_replacements if step else ""
        row[f"update_limit_pos_{position}"] = step.mean_update_limit if step else ""
        row[f"true_entrants_pos_{position}"] = step.mean_true_entrants if step else ""
    for position in range(1, gamma):
        if position < len(diagnostics):
            left, right = diagnostics[position - 1], diagnostics[position]
            row[f"oracle_overlap_pos_{position}_{position + 1}"] = adjacent_oracle_overlap(
                left, right
            )
            row[f"selection_overlap_pos_{position}_{position + 1}"] = selection_overlap(
                left, right
            )
        else:
            row[f"oracle_overlap_pos_{position}_{position + 1}"] = ""
            row[f"selection_overlap_pos_{position}_{position + 1}"] = ""
    row["oracle_overlap_pos_1_last"] = (
        endpoint_overlap(diagnostics[0], diagnostics[-1]) if len(diagnostics) >= 2 else ""
    )
    row["selection_overlap_pos_1_last"] = (
        selection_overlap(diagnostics[0], diagnostics[-1]) if len(diagnostics) >= 2 else ""
    )
    return row


def run_method(args, sample, model, controller, tokenizer, spec: MethodSpec):
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
        draft_tokens, diagnostics = draft_round(
            model, controller, state, dense_cache, prior_queries, spec, target, eos
        )
        accepted, correction, first_queries, terminal_draft_queries, verification_cache = base.verify_round(
            model, controller, state, dense_cache, draft_tokens
        )
        commit = list(draft_tokens[:accepted])
        accepted_eos = (
            accepted == len(draft_tokens)
            and eos is not None
            and draft_tokens[-1] == eos
        )
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
            layer: (0.5 * (first_queries[layer].float() + terminal_queries[layer].float())).to(
                torch.float16
            )
            for layer in first_queries
        }

        round_start_length = int(state.shape[1])
        state = torch.cat(
            (state, torch.tensor([commit], device=device, dtype=state.dtype)), dim=1
        )
        generated.extend(commit)
        rounds.append({
            "sample_id": sample.sample_id,
            "dataset_name": sample.dataset_name,
            "method": spec.name,
            "update_ratio": spec.update_ratio,
            "input_length": sample.input_length,
            "round_id": round_id,
            "round_start_length": round_start_length,
            "round_start_pages": math.ceil(round_start_length / args.page_size),
            "budget_pages": controller.budget_pages,
            "max_update_pages": controller.update_limit_pages,
            "draft_length": len(draft_tokens),
            "accepted_length": accepted,
            "all_accepted": int(accepted == args.draft_length),
            "zero_accepted": int(accepted == 0),
            "generated_tokens_so_far": len(generated),
            **diagnostic_fields(diagnostics, args.draft_length),
        })
        if eos is not None and eos in commit:
            break
    return base.MethodResult(method=spec.name, generated_tokens=generated, rounds=rounds)


def round_fields(gamma: int) -> list[str]:
    fields = [
        "sample_id", "dataset_name", "method", "update_ratio", "input_length",
        "round_id", "round_start_length", "round_start_pages", "budget_pages",
        "max_update_pages", "draft_length", "accepted_length", "all_accepted",
        "zero_accepted", "generated_tokens_so_far",
    ]
    for position in range(1, gamma + 1):
        fields += [
            f"selection_recall_of_oracle_b_pos_{position}",
            f"attention_recovery_pos_{position}",
            f"replacements_pos_{position}",
            f"update_limit_pos_{position}",
            f"true_entrants_pos_{position}",
        ]
    for position in range(1, gamma):
        fields += [
            f"oracle_overlap_pos_{position}_{position + 1}",
            f"selection_overlap_pos_{position}_{position + 1}",
        ]
    fields += ["oracle_overlap_pos_1_last", "selection_overlap_pos_1_last"]
    return fields


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def compare_counts(differences: list[float]) -> dict[str, int]:
    return {
        "better": sum(value > 1e-12 for value in differences),
        "equal": sum(abs(value) <= 1e-12 for value in differences),
        "worse": sum(value < -1e-12 for value in differences),
    }


def summarize(args, specs, samples, all_results, reference: ReferenceResults) -> None:
    result_dir = args.results_dir
    round_rows = [row for _, result in all_results for row in result.rounds]
    write_csv(
        result_dir / "page_incremental_per_round.csv",
        round_rows,
        round_fields(args.draft_length),
    )

    sample_rows = []
    for sample, result in all_results:
        accepted = [int(row["accepted_length"]) for row in result.rounds]
        key = (sample.dataset_name, sample.sample_id)
        static = float(reference.per_sample[key + ("static",)]["mean_accepted_length"])
        best = float(reference.per_sample[key + ("best_static_oracle",)]["mean_accepted_length"])
        oracle = float(reference.per_sample[key + ("oracle_b",)]["mean_accepted_length"])
        mean_accepted = safe_mean(accepted)
        sample_rows.append({
            "sample_id": sample.sample_id,
            "dataset_name": sample.dataset_name,
            "method": result.method,
            "input_length": sample.input_length,
            "speculative_rounds": len(accepted),
            "total_generated_tokens": len(result.generated_tokens),
            "total_accepted_draft_tokens": sum(accepted),
            "mean_accepted_length": mean_accepted,
            "full_accept_rounds": sum(value == args.draft_length for value in accepted),
            "zero_accept_rounds": sum(value == 0 for value in accepted),
            "reference_static_mean": static,
            "reference_best_static_mean": best,
            "reference_oracle_b_mean": oracle,
            "delta_vs_static": mean_accepted - static,
            "gap_to_oracle_b": oracle - mean_accepted,
            "fraction_of_oracle_b_gain_closed": (
                (mean_accepted - static) / (oracle - static) if oracle > static else 0.0
            ),
        })
    write_csv(result_dir / "page_incremental_per_sample.csv", sample_rows)

    position_rows = []
    for row in reference.positions:
        position_rows.append({
            "method": row["method"],
            "result_source": "reused_reference",
            "draft_position": int(row["draft_position"]),
            "eligible_rounds": int(row["eligible_rounds"]),
            "accepted_rounds": int(row["accepted_rounds"]),
            "conditional_acceptance_rate": float(row["conditional_acceptance_rate"]),
        })
    for spec in specs:
        rows = [row for row in round_rows if row["method"] == spec.name]
        for position in range(1, args.draft_length + 1):
            eligible = [
                row for row in rows
                if int(row["draft_length"]) >= position
                and int(row["accepted_length"]) >= position - 1
            ]
            accepted = sum(int(row["accepted_length"]) >= position for row in eligible)
            position_rows.append({
                "method": spec.name,
                "result_source": "current_run",
                "draft_position": position,
                "eligible_rounds": len(eligible),
                "accepted_rounds": accepted,
                "conditional_acceptance_rate": accepted / len(eligible) if eligible else 0.0,
            })
    write_csv(result_dir / "position_acceptance_rate.csv", position_rows)

    distribution_rows = []
    for method in REFERENCE_METHODS:
        for row in reference.distribution:
            distribution_rows.append({
                "method": method,
                "result_source": "reused_reference",
                "accepted_length": int(row["accepted_length"]),
                "rounds": int(row[f"{method}_rounds"]),
                "ratio": float(row[f"{method}_ratio"]),
            })
    for spec in specs:
        values = [int(row["accepted_length"]) for row in round_rows if row["method"] == spec.name]
        for length in range(args.draft_length + 1):
            count = sum(value == length for value in values)
            distribution_rows.append({
                "method": spec.name,
                "result_source": "current_run",
                "accepted_length": length,
                "rounds": count,
                "ratio": count / len(values) if values else 0.0,
            })
    write_csv(result_dir / "acceptance_length_distribution.csv", distribution_rows)

    summary_rows = []
    for method in REFERENCE_METHODS:
        item = reference.summary["methods"][method]
        summary_rows.append({
            "method": method,
            "label": REFERENCE_LABELS[method],
            "result_source": "reused_reference",
            "update_ratio": "",
            "mean_accepted_length": item["mean_accepted_length"],
            "macro_by_sample_mean_accepted_length": item["macro_by_sample_mean_accepted_length"],
            "full_acceptance_rate": item["full_acceptance_rate"],
            "zero_acceptance_rate": item["zero_acceptance_rate"],
            "total_rounds": item["total_rounds"],
            "mean_max_update_pages": "",
            "mean_actual_replacements_per_step": "",
            "mean_total_replacements_per_round": "",
            "mean_budget_pages": "",
            "mean_selection_recall_of_oracle_b": "",
            "mean_attention_recovery": item["mean_attention_recovery"],
            "macro_gain_recovery": (
                0.0 if method == "static"
                else 1.0 if method == "oracle_b"
                else reference.summary["paired_macro"]["best_static_fraction_of_oracle_b_gain_closed"]
            ),
            "better_than_static_samples": "",
            "equal_to_static_samples": "",
            "worse_than_static_samples": "",
            "better_than_best_static_samples": "",
            "equal_to_best_static_samples": "",
            "worse_than_best_static_samples": "",
            "better_than_oracle_b_samples": "",
            "equal_to_oracle_b_samples": "",
            "worse_than_oracle_b_samples": "",
        })

    static_macro = reference.summary["methods"]["static"]["macro_by_sample_mean_accepted_length"]
    oracle_macro = reference.summary["methods"]["oracle_b"]["macro_by_sample_mean_accepted_length"]
    for spec in specs:
        rows = [row for row in round_rows if row["method"] == spec.name]
        accepted = [int(row["accepted_length"]) for row in rows]
        method_samples = [row for row in sample_rows if row["method"] == spec.name]
        macro = safe_mean(float(row["mean_accepted_length"]) for row in method_samples)
        replacements = [
            float(row[f"replacements_pos_{position}"])
            for row in rows
            for position in range(2, args.draft_length + 1)
            if row[f"replacements_pos_{position}"] != ""
        ]
        replacements_per_round = [
            sum(
                float(row[f"replacements_pos_{position}"])
                for position in range(2, args.draft_length + 1)
                if row[f"replacements_pos_{position}"] != ""
            )
            for row in rows
        ]
        recalls = [
            float(row[f"selection_recall_of_oracle_b_pos_{position}"])
            for row in rows
            for position in range(1, args.draft_length + 1)
            if row[f"selection_recall_of_oracle_b_pos_{position}"] != ""
        ]
        recoveries = [
            float(row[f"attention_recovery_pos_{position}"])
            for row in rows
            for position in range(1, args.draft_length + 1)
            if row[f"attention_recovery_pos_{position}"] != ""
        ]
        differences = [float(row["delta_vs_static"]) for row in method_samples]
        counts = compare_counts(differences)
        best_counts = compare_counts([
            float(row["mean_accepted_length"]) - float(row["reference_best_static_mean"])
            for row in method_samples
        ])
        oracle_counts = compare_counts([
            float(row["mean_accepted_length"]) - float(row["reference_oracle_b_mean"])
            for row in method_samples
        ])
        summary_rows.append({
            "method": spec.name,
            "label": f"Page Incremental {spec.update_ratio:.0%}",
            "result_source": "current_run",
            "update_ratio": spec.update_ratio,
            "mean_accepted_length": safe_mean(accepted),
            "macro_by_sample_mean_accepted_length": macro,
            "full_acceptance_rate": sum(value == args.draft_length for value in accepted) / len(accepted),
            "zero_acceptance_rate": sum(value == 0 for value in accepted) / len(accepted),
            "total_rounds": len(rows),
            "mean_max_update_pages": safe_mean(float(row["max_update_pages"]) for row in rows),
            "mean_actual_replacements_per_step": safe_mean(replacements),
            "mean_total_replacements_per_round": safe_mean(replacements_per_round),
            "mean_budget_pages": safe_mean(float(row["budget_pages"]) for row in rows),
            "mean_selection_recall_of_oracle_b": safe_mean(recalls),
            "mean_attention_recovery": safe_mean(recoveries),
            "macro_gain_recovery": (
                (macro - static_macro) / (oracle_macro - static_macro)
                if oracle_macro > static_macro else 0.0
            ),
            "better_than_static_samples": counts["better"],
            "equal_to_static_samples": counts["equal"],
            "worse_than_static_samples": counts["worse"],
            "better_than_best_static_samples": best_counts["better"],
            "equal_to_best_static_samples": best_counts["equal"],
            "worse_than_best_static_samples": best_counts["worse"],
            "better_than_oracle_b_samples": oracle_counts["better"],
            "equal_to_oracle_b_samples": oracle_counts["equal"],
            "worse_than_oracle_b_samples": oracle_counts["worse"],
        })
    write_csv(result_dir / "page_incremental_summary.csv", summary_rows)

    payload = {
        "reference_results_dir": str(reference.root),
        "reference_methods_were_rerun": False,
        "num_samples": len(samples),
        "methods": summary_rows,
    }
    (result_dir / "page_incremental_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_report(result_dir, summary_rows, position_rows)
    write_plots(result_dir, summary_rows, position_rows)


def write_report(result_dir: Path, summary_rows: list[dict], position_rows: list[dict]) -> None:
    lines = [
        "# Page-level Incremental KV Selection", "",
        "The three reference rows are loaded from the validated Best Static Oracle result directory; they were not rerun.",
        "", "## Acceptance", "",
        "| Method | Source | Mean accepted | Full-8 rate | Max update pages | Actual replacements | Macro Oracle-B gain recovery |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        max_pages = row["mean_max_update_pages"]
        replacements = row["mean_actual_replacements_per_step"]
        if max_pages == "":
            max_pages_text, replacements_text = "—", "—"
        else:
            max_pages_text = f"{float(max_pages):.2f}"
            replacements_text = f"{float(replacements):.2f}"
        lines.append(
            f"| {row['label']} | {row['result_source']} | "
            f"{float(row['mean_accepted_length']):.4f} | "
            f"{float(row['full_acceptance_rate']):.4f} | {max_pages_text} | "
            f"{replacements_text} | {float(row['macro_gain_recovery']):.2%} |"
        )
    lines += [
        "", "## Conditional acceptance", "",
        "| Method | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        values = [
            item["conditional_acceptance_rate"]
            for item in position_rows if item["method"] == row["method"]
        ]
        lines.append(
            f"| {row['label']} | " + " | ".join(f"{float(value):.4f}" for value in values) + " |"
        )
    lines += [
        "", "## Paired sample comparisons", "",
        "| Method | vs Static W/E/L | vs Best Static W/E/L | vs Oracle B W/E/L |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        if row["result_source"] != "current_run":
            continue
        lines.append(
            f"| {row['label']} | "
            f"{row['better_than_static_samples']}/{row['equal_to_static_samples']}/{row['worse_than_static_samples']} | "
            f"{row['better_than_best_static_samples']}/{row['equal_to_best_static_samples']}/{row['worse_than_best_static_samples']} | "
            f"{row['better_than_oracle_b_samples']}/{row['equal_to_oracle_b_samples']}/{row['worse_than_oracle_b_samples']} |"
        )
    lines += [
        "", "## Interpretation", "",
        "Use mean accepted length and late-position conditional acceptance as the primary evidence. "
        "Selection recall and attention recovery are diagnostics and must not replace acceptance-quality comparisons.",
    ]
    (result_dir / "experiment_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_plots(result_dir: Path, summary_rows: list[dict], position_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (result_dir / "plots_skipped.txt").write_text(
            "matplotlib is not installed\n", encoding="utf-8"
        )
        return
    plot_dir = result_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(9, 5))
    for row in summary_rows:
        values = [item for item in position_rows if item["method"] == row["method"]]
        plt.plot(
            [int(item["draft_position"]) for item in values],
            [float(item["conditional_acceptance_rate"]) for item in values],
            marker="o", label=row["label"],
        )
    plt.xlabel("Draft position")
    plt.ylabel("Conditional acceptance rate")
    plt.xticks(range(1, 9))
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plot_dir / "conditional_acceptance_by_position.png", dpi=160)
    plt.close()

    incremental = [row for row in summary_rows if row["result_source"] == "current_run"]
    plt.figure(figsize=(7, 5))
    plt.plot(
        [100 * float(row["update_ratio"]) for row in incremental],
        [float(row["macro_by_sample_mean_accepted_length"]) for row in incremental],
        marker="o", label="Page Incremental",
    )
    for method in ("static", "best_static_oracle", "oracle_b"):
        row = next(item for item in summary_rows if item["method"] == method)
        plt.axhline(
            float(row["macro_by_sample_mean_accepted_length"]),
            linestyle="--", linewidth=1, label=row["label"],
        )
    plt.xlabel("Maximum pages replaced per step (% of selected pages)")
    plt.ylabel("Mean accepted length (sample macro)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plot_dir / "update_ratio_vs_accepted_length.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.results_dir.resolve() == args.reference_results_dir.resolve():
        raise ValueError("results-dir must not overwrite the reused reference result directory")
    specs = methods()
    allocation = base.parse_allocation(args.dataset_allocation)
    if sum(allocation.values()) != args.num_samples:
        raise ValueError("dataset-allocation must sum to num-samples")
    if args.draft_length != 8:
        raise ValueError("this comparison fixes draft-length at 8")
    if args.page_size != 16:
        raise ValueError("page-size must remain 16 to match the existing references")
    if not math.isclose(args.budget_ratio, 0.1):
        raise ValueError("budget-ratio must remain 0.1 to match the existing references")
    reference = load_reference_results(args.reference_results_dir)
    base.set_deterministic_seed(args.sample_seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    samples = base.prepare_samples(args, tokenizer, allocation)
    validate_reference(args, samples, reference)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    controller = PageIncrementalController(args.page_size, args.budget_ratio)

    check_length = min(args.correctness_check_tokens, samples[0].input_length)
    check_ids = samples[0].input_ids[:, -check_length:].to(args.device)
    with torch.inference_mode():
        reference_logits = model(input_ids=check_ids, use_cache=False).logits[:, -1].float().cpu()
    patch_qwen3_for_sparse_drafting(model, controller)
    with torch.inference_mode():
        controller.begin_dense()
        patched_logits = model(input_ids=check_ids, use_cache=False).logits[:, -1].float().cpu()
    max_error = float((reference_logits - patched_logits).abs().max().item())
    top1_equal = bool(reference_logits.argmax().item() == patched_logits.argmax().item())
    if max_error > args.dense_check_atol or not top1_equal:
        raise RuntimeError(f"dense patch check failed: max_error={max_error}, top1_equal={top1_equal}")

    config = vars(args).copy()
    config.update({
        "model_path": str(args.model_path),
        "data_root": str(args.data_root),
        "results_dir": str(args.results_dir),
        "reference_results_dir": str(args.reference_results_dir),
        "methods": [spec.__dict__ for spec in specs],
        "reference_methods": list(REFERENCE_METHODS),
        "reference_methods_were_rerun": False,
        "page_score": "max token post-RoPE QK within page",
        "selection_granularity": "per layer and query head, using the corresponding GQA KV head",
        "budget_rounding": "max(1, ceil(0.1 * round_start_pages))",
        "update_rounding": "max(1, ceil(update_ratio * selected_page_budget))",
        "initial_selection": "previous verification first/terminal post-RoPE Query mean",
        "incremental_oracle_target": "current real sparse Query, before this layer's attention",
        "forced_pages": "none",
        "oracle_selection_time_in_primary_comparison": False,
        "dense_patch_check": {
            "tokens": check_length,
            "max_abs_logit_error": max_error,
            "top1_equal": top1_equal,
        },
        "sample_ids": [sample.sample_id for sample in samples],
    })
    (args.results_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    all_results = []
    with torch.inference_mode():
        for sample in samples:
            outputs = []
            for spec in specs:
                base.set_deterministic_seed(args.sample_seed)
                result = run_method(args, sample, model, controller, tokenizer, spec)
                all_results.append((sample, result))
                outputs.append(result.generated_tokens)
            if any(output != outputs[0] for output in outputs[1:]):
                raise RuntimeError(f"dense-equivalence failure on {sample.sample_id}")
    summarize(args, specs, samples, all_results, reference)


if __name__ == "__main__":
    main()
