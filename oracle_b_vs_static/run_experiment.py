from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import build_prompt, encode_prompt, generation_prompt, iter_input_samples
from sparse_qwen3 import (
    SparseKVController,
    StepDiagnostics,
    adjacent_oracle_overlap,
    endpoint_overlap,
    patch_qwen3_for_sparse_drafting,
)


METHODS = ("static", "oracle_b")


@dataclass
class PreparedSample:
    sample_id: str
    dataset_name: str
    input_ids: torch.Tensor
    input_length: int


@dataclass
class MethodResult:
    method: str
    generated_tokens: list[int]
    rounds: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare static 10% KV pages with per-token true-query Oracle B.")
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


def parse_allocation(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in value.split(","):
        name, count = item.split(":", maxsplit=1)
        result[name.strip()] = int(count)
    return result


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_samples(args, tokenizer, allocation: dict[str, int]) -> list[PreparedSample]:
    prepared: list[PreparedSample] = []
    rng = random.Random(args.sample_seed)
    for dataset_offset, (dataset_name, count) in enumerate(allocation.items()):
        candidates: list[tuple[float, PreparedSample]] = []
        for sample in iter_input_samples(args.data_root, dataset_name):
            prompt = generation_prompt(tokenizer, build_prompt(sample))
            ids = encode_prompt(tokenizer, prompt, args.max_context_tokens)
            length = int(ids.shape[1])
            if length < args.min_input_tokens:
                continue
            tie_breaker = random.Random(f"{args.sample_seed}:{dataset_offset}:{sample.sample_id}").random()
            candidates.append(
                (
                    tie_breaker,
                    PreparedSample(
                        sample_id=str(sample.sample_id),
                        dataset_name=sample.dataset_name,
                        input_ids=ids,
                        input_length=length,
                    ),
                )
            )
        if args.sample_policy == "longest":
            candidates.sort(key=lambda item: (-item[1].input_length, item[0]))
        else:
            rng.shuffle(candidates)
        if len(candidates) < count:
            raise ValueError(
                f"{dataset_name} has only {len(candidates)} samples with at least "
                f"{args.min_input_tokens} tokens, but {count} were requested"
            )
        prepared.extend(item[1] for item in candidates[:count])
    return sorted(prepared, key=lambda item: (item.dataset_name, item.sample_id))


def clone_to_device(ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    return ids.detach().clone().to(device)


def initialize_dense_state(model, controller, prompt_ids: torch.Tensor):
    final_position = int(prompt_ids.shape[1] - 1)
    controller.begin_dense({final_position})
    output = model(input_ids=prompt_ids, use_cache=True)
    captures = controller.captured_queries
    if final_position not in captures:
        raise RuntimeError("failed to capture the final prompt query for first-round initialization")
    if output.past_key_values is None or not hasattr(output.past_key_values, "crop"):
        raise RuntimeError("this experiment requires a Hugging Face Cache object with crop() support")
    return captures[final_position], output.past_key_values


def cloned_cropped_cache(cache, length: int):
    result = copy.deepcopy(cache)
    result.crop(length)
    return result


def draft_round(
    model,
    controller: SparseKVController,
    state: torch.Tensor,
    dense_cache,
    method: str,
    prior_queries: dict[int, torch.Tensor],
    draft_target: int,
    eos_token_id: int | None,
) -> tuple[list[int], list[StepDiagnostics]]:
    cache = cloned_cropped_cache(dense_cache, int(state.shape[1] - 1))
    controller.begin_sparse(method, int(state.shape[1]), prior_queries)
    current_input = state[:, -1:]
    draft_tokens: list[int] = []
    diagnostics: list[StepDiagnostics] = []
    for position in range(1, draft_target + 1):
        controller.start_step(position)
        output = model(input_ids=current_input, use_cache=True, past_key_values=cache)
        diagnostics.append(controller.finish_step())
        cache = output.past_key_values
        token = int(output.logits[0, -1].argmax().item())
        draft_tokens.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
        current_input = torch.tensor([[token]], device=state.device, dtype=state.dtype)
    return draft_tokens, diagnostics


def verify_round(
    model,
    controller: SparseKVController,
    state: torch.Tensor,
    dense_cache,
    draft_tokens: list[int],
) -> tuple[int, int, dict[int, torch.Tensor], dict[int, torch.Tensor], object]:
    draft = torch.tensor([draft_tokens], device=state.device, dtype=state.dtype)
    state_length = int(state.shape[1])
    first_query_position = state_length
    terminal_query_position = state_length + len(draft_tokens) - 1
    verification_input = torch.cat((state[:, -1:], draft), dim=1)
    verification_cache = cloned_cropped_cache(dense_cache, state_length - 1)
    controller.begin_dense({first_query_position, terminal_query_position})
    output = model(
        input_ids=verification_input,
        past_key_values=verification_cache,
        use_cache=True,
    )
    captures = controller.captured_queries
    verification_cache = output.past_key_values
    verifier_tokens = output.logits[0].argmax(dim=-1).tolist()
    accepted = 0
    while accepted < len(draft_tokens) and draft_tokens[accepted] == verifier_tokens[accepted]:
        accepted += 1
    correction_or_bonus = int(verifier_tokens[accepted])

    first = captures.get(first_query_position)
    terminal = captures.get(terminal_query_position)
    if first is None or terminal is None:
        raise RuntimeError("verification did not capture endpoint queries")
    return accepted, correction_or_bonus, first, terminal, verification_cache


def _safe_mean(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return mean(clean) if clean else 0.0


def diagnostic_fields(diagnostics: list[StepDiagnostics], gamma: int) -> dict[str, float | str]:
    row: dict[str, float | str] = {}
    for position in range(1, gamma + 1):
        diag = diagnostics[position - 1] if position <= len(diagnostics) else None
        row[f"static_oracle_recall_pos_{position}"] = diag.mean_static_oracle_recall if diag else ""
        row[f"attention_recovery_pos_{position}"] = diag.mean_attention_recovery if diag else ""
    for position in range(1, gamma):
        if position < len(diagnostics):
            row[f"oracle_overlap_pos_{position}_{position + 1}"] = adjacent_oracle_overlap(
                diagnostics[position - 1], diagnostics[position]
            )
        else:
            row[f"oracle_overlap_pos_{position}_{position + 1}"] = ""
    row["oracle_overlap_pos_1_last"] = (
        endpoint_overlap(diagnostics[0], diagnostics[-1]) if len(diagnostics) >= 2 else ""
    )
    return row


def run_method(args, sample: PreparedSample, model, controller, tokenizer, method: str) -> MethodResult:
    device = torch.device(args.device)
    state = clone_to_device(sample.input_ids, device)
    prior_queries, dense_cache = initialize_dense_state(model, controller, state)
    generated: list[int] = []
    rounds: list[dict] = []
    round_id = 0
    eos = tokenizer.eos_token_id

    while len(generated) < args.max_new_tokens:
        round_id += 1
        remaining = args.max_new_tokens - len(generated)
        target = min(args.draft_length, remaining)
        draft_tokens, diagnostics = draft_round(
            model, controller, state, dense_cache, method, prior_queries, target, eos
        )
        accepted, correction, first_draft_queries, terminal_draft_queries, verification_cache = verify_round(
            model, controller, state, dense_cache, draft_tokens
        )
        accepted_tokens = draft_tokens[:accepted]
        commit = list(accepted_tokens)
        accepted_eos = accepted == len(draft_tokens) and eos is not None and draft_tokens[-1] == eos
        append_verifier = not accepted_eos and len(commit) < remaining
        if append_verifier:
            commit.append(correction)

        committed_before_verifier = int(state.shape[1] + accepted)
        verification_cache.crop(committed_before_verifier)
        if append_verifier:
            controller.begin_dense({committed_before_verifier})
            correction_tensor = torch.tensor([[correction]], device=device, dtype=state.dtype)
            correction_output = model(
                input_ids=correction_tensor,
                past_key_values=verification_cache,
                use_cache=True,
            )
            dense_cache = correction_output.past_key_values
            terminal_queries = controller.captured_queries.get(committed_before_verifier)
            if terminal_queries is None:
                raise RuntimeError("failed to capture the verifier/bonus endpoint query")
        else:
            dense_cache = verification_cache
            terminal_queries = terminal_draft_queries
        next_prior = {
            layer: (0.5 * (first_draft_queries[layer].float() + terminal_queries[layer].float())).to(torch.float16)
            for layer in first_draft_queries
        }
        state = torch.cat(
            (state, torch.tensor([commit], device=device, dtype=state.dtype)), dim=1
        )
        generated.extend(commit)
        prior_queries = next_prior

        row = {
            "sample_id": sample.sample_id,
            "dataset_name": sample.dataset_name,
            "method": method,
            "input_length": sample.input_length,
            "num_kv_pages": math.ceil(sample.input_length / args.page_size),
            "round_id": round_id,
            "round_start_length": int(state.shape[1] - len(commit)),
            "round_start_pages": math.ceil((state.shape[1] - len(commit)) / args.page_size),
            "budget_pages": controller.budget_pages,
            "draft_length": len(draft_tokens),
            "accepted_length": accepted,
            "all_accepted": int(accepted == args.draft_length),
            "zero_accepted": int(accepted == 0),
            "generated_tokens_so_far": len(generated),
        }
        row.update(diagnostic_fields(diagnostics, args.draft_length))
        rounds.append(row)
        if eos is not None and eos in commit:
            break
    return MethodResult(method=method, generated_tokens=generated, rounds=rounds)


def round_fields(gamma: int) -> list[str]:
    fields = [
        "sample_id", "dataset_name", "method", "input_length", "num_kv_pages",
        "round_id", "round_start_length", "round_start_pages", "budget_pages",
        "draft_length", "accepted_length", "all_accepted", "zero_accepted",
        "generated_tokens_so_far",
    ]
    fields.extend(f"static_oracle_recall_pos_{position}" for position in range(1, gamma + 1))
    fields.extend(f"oracle_overlap_pos_{position}_{position + 1}" for position in range(1, gamma))
    fields.append("oracle_overlap_pos_1_last")
    fields.extend(f"attention_recovery_pos_{position}" for position in range(1, gamma + 1))
    return fields


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(args, samples: list[PreparedSample], all_results: list[tuple[PreparedSample, MethodResult]]) -> None:
    results_dir = args.results_dir
    round_rows = [row for _, result in all_results for row in result.rounds]
    write_csv(results_dir / "oracle_b_vs_static_per_round.csv", round_rows, round_fields(args.draft_length))

    per_sample_rows: list[dict] = []
    by_sample_method: dict[tuple[str, str, str], dict] = {}
    for sample, result in all_results:
        accepted = [int(row["accepted_length"]) for row in result.rounds]
        summary = {
            "sample_id": sample.sample_id,
            "dataset_name": sample.dataset_name,
            "method": result.method,
            "input_length": sample.input_length,
            "num_kv_pages": math.ceil(sample.input_length / args.page_size),
            "speculative_rounds": len(result.rounds),
            "total_generated_tokens": len(result.generated_tokens),
            "total_accepted_draft_tokens": sum(accepted),
            "mean_accepted_length": mean(accepted) if accepted else 0.0,
            "full_accept_rounds": sum(value == args.draft_length for value in accepted),
            "zero_accept_rounds": sum(value == 0 for value in accepted),
        }
        by_sample_method[(sample.dataset_name, sample.sample_id, result.method)] = summary

    for sample in samples:
        static = by_sample_method[(sample.dataset_name, sample.sample_id, "static")]
        oracle = by_sample_method[(sample.dataset_name, sample.sample_id, "oracle_b")]
        delta = oracle["mean_accepted_length"] - static["mean_accepted_length"]
        for source in (static, oracle):
            row = dict(source)
            row["delta_mean_accepted_oracle_minus_static"] = delta
            per_sample_rows.append(row)
    sample_fields = list(per_sample_rows[0])
    write_csv(results_dir / "oracle_b_vs_static_per_sample.csv", per_sample_rows, sample_fields)

    distribution_rows: list[dict] = []
    for accepted_length in range(args.draft_length + 1):
        row: dict[str, int | float] = {"accepted_length": accepted_length}
        for method in METHODS:
            values = [int(item["accepted_length"]) for item in round_rows if item["method"] == method]
            count = sum(value == accepted_length for value in values)
            row[f"{method}_rounds"] = count
            row[f"{method}_ratio"] = count / len(values) if values else 0.0
        distribution_rows.append(row)
    write_csv(
        results_dir / "acceptance_length_distribution.csv",
        distribution_rows,
        list(distribution_rows[0]),
    )

    position_rows: list[dict] = []
    for method in METHODS:
        method_rounds = [row for row in round_rows if row["method"] == method]
        for position in range(1, args.draft_length + 1):
            eligible = [
                row for row in method_rounds
                if int(row["draft_length"]) >= position and int(row["accepted_length"]) >= position - 1
            ]
            accepted_count = sum(int(row["accepted_length"]) >= position for row in eligible)
            position_rows.append(
                {
                    "method": method,
                    "draft_position": position,
                    "eligible_rounds": len(eligible),
                    "accepted_rounds": accepted_count,
                    "conditional_acceptance_rate": accepted_count / len(eligible) if eligible else 0.0,
                }
            )
    write_csv(results_dir / "position_acceptance_rate.csv", position_rows, list(position_rows[0]))

    overall: dict[str, dict] = {}
    for method in METHODS:
        rows = [row for row in round_rows if row["method"] == method]
        values = [int(row["accepted_length"]) for row in rows]
        overall[method] = {
            "kv_budget_ratio": args.budget_ratio,
            "draft_length": args.draft_length,
            "mean_accepted_length": mean(values) if values else 0.0,
            "full_acceptance_rate": sum(value == args.draft_length for value in values) / len(values) if values else 0.0,
            "zero_acceptance_rate": sum(value == 0 for value in values) / len(values) if values else 0.0,
            "total_rounds": len(values),
            "total_generated_tokens": sum(
                len(item[1].generated_tokens) for item in all_results if item[1].method == method
            ),
            "mean_attention_recovery": _safe_mean([
                float(row[f"attention_recovery_pos_{position}"])
                for row in rows for position in range(1, args.draft_length + 1)
                if row[f"attention_recovery_pos_{position}"] != ""
            ]),
            "mean_adjacent_oracle_selection_overlap": _safe_mean([
                float(row[f"oracle_overlap_pos_{position}_{position + 1}"])
                for row in rows for position in range(1, args.draft_length)
                if row[f"oracle_overlap_pos_{position}_{position + 1}"] != ""
            ]),
            "mean_first_last_oracle_selection_overlap": _safe_mean([
                float(row["oracle_overlap_pos_1_last"])
                for row in rows if row["oracle_overlap_pos_1_last"] != ""
            ]),
        }
    static_mean = overall["static"]["mean_accepted_length"]
    oracle_mean = overall["oracle_b"]["mean_accepted_length"]
    deltas = [
        by_sample_method[(sample.dataset_name, sample.sample_id, "oracle_b")]["mean_accepted_length"]
        - by_sample_method[(sample.dataset_name, sample.sample_id, "static")]["mean_accepted_length"]
        for sample in samples
    ]
    input_lengths = [sample.input_length for sample in samples]
    page_counts = [math.ceil(length / args.page_size) for length in input_lengths]
    round_start_pages = [int(row["round_start_pages"]) for row in round_rows]
    payload = {
        "methods": overall,
        "mean_accepted_length_absolute_improvement": oracle_mean - static_mean,
        "mean_accepted_length_relative_improvement_percent": (
            (oracle_mean - static_mean) / static_mean * 100.0 if static_mean > 0 else None
        ),
        "paired_sample_outcomes": {
            "oracle_b_better_count": sum(delta > 1e-12 for delta in deltas),
            "same_count": sum(abs(delta) <= 1e-12 for delta in deltas),
            "oracle_b_worse_count": sum(delta < -1e-12 for delta in deltas),
            "oracle_b_better_ratio": sum(delta > 1e-12 for delta in deltas) / len(deltas),
            "same_ratio": sum(abs(delta) <= 1e-12 for delta in deltas) / len(deltas),
            "oracle_b_worse_ratio": sum(delta < -1e-12 for delta in deltas) / len(deltas),
        },
        "input_statistics": {
            "num_samples": len(samples),
            "input_tokens_mean": mean(input_lengths),
            "input_tokens_median": median(input_lengths),
            "input_tokens_min": min(input_lengths),
            "input_tokens_max": max(input_lengths),
            "kv_pages_mean": mean(page_counts),
            "kv_pages_median": median(page_counts),
            "kv_pages_min": min(page_counts),
            "kv_pages_max": max(page_counts),
            "round_start_kv_pages_mean": mean(round_start_pages),
            "round_start_kv_pages_median": median(round_start_pages),
            "round_start_kv_pages_min": min(round_start_pages),
            "round_start_kv_pages_max": max(round_start_pages),
        },
        "diagnostics_by_accepted_length": {},
    }
    for method in METHODS:
        method_payload = {}
        for accepted_length in range(args.draft_length + 1):
            rows = [
                row for row in round_rows
                if row["method"] == method and int(row["accepted_length"]) == accepted_length
            ]
            method_payload[str(accepted_length)] = {
                "rounds": len(rows),
                "mean_first_last_oracle_overlap": _safe_mean([
                    float(row["oracle_overlap_pos_1_last"])
                    for row in rows if row["oracle_overlap_pos_1_last"] != ""
                ]),
                "mean_attention_recovery": _safe_mean([
                    float(row[f"attention_recovery_pos_{position}"])
                    for row in rows for position in range(1, args.draft_length + 1)
                    if row[f"attention_recovery_pos_{position}"] != ""
                ]),
            }
        payload["diagnostics_by_accepted_length"][method] = method_payload
    (results_dir / "oracle_b_vs_static_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_markdown_report(results_dir, payload, position_rows, distribution_rows)


def write_markdown_report(results_dir: Path, payload: dict, position_rows: list[dict], distribution_rows: list[dict]) -> None:
    methods = payload["methods"]
    labels = {"static": "Static endpoint mean", "oracle_b": "Oracle B"}
    lines = [
        "# Oracle B vs Static 10% KV Selection",
        "",
        "## Overall acceptance",
        "",
        "| Method | KV budget | Draft length | Mean accepted length | Full-8 rate | Zero-accept rate | Rounds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        item = methods[method]
        lines.append(
            f"| {labels[method]} | {item['kv_budget_ratio']:.0%} | {item['draft_length']} | "
            f"{item['mean_accepted_length']:.4f} | {item['full_acceptance_rate']:.4f} | "
            f"{item['zero_acceptance_rate']:.4f} | {item['total_rounds']} |"
        )
    relative = payload["mean_accepted_length_relative_improvement_percent"]
    relative_text = "undefined" if relative is None else f"{relative:.2f}%"
    lines.extend(
        [
            "",
            f"Oracle B absolute improvement: `{payload['mean_accepted_length_absolute_improvement']:.4f}`.",
            f"Oracle B relative improvement: `{relative_text}`.",
            f"Mean attention recovery (static / Oracle B): "
            f"`{methods['static']['mean_attention_recovery']:.4f} / {methods['oracle_b']['mean_attention_recovery']:.4f}`.",
            "",
            "## Conditional acceptance by draft position",
            "",
            "| Method | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 | Pos 8 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS:
        values = [
            row["conditional_acceptance_rate"] for row in position_rows if row["method"] == method
        ]
        lines.append(f"| {labels[method]} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines.extend(
        [
            "",
            "## Acceptance-length distribution",
            "",
            "| Accepted length | Static rounds | Static ratio | Oracle B rounds | Oracle B ratio |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in distribution_rows:
        lines.append(
            f"| {row['accepted_length']} | {row['static_rounds']} | {row['static_ratio']:.4f} | "
            f"{row['oracle_b_rounds']} | {row['oracle_b_ratio']:.4f} |"
        )
    stats = payload["input_statistics"]
    lines.extend(
        [
            "",
            "## Input coverage",
            "",
            f"- Samples: `{stats['num_samples']}`",
            f"- Input tokens (mean / median / min / max): `{stats['input_tokens_mean']:.1f} / "
            f"{stats['input_tokens_median']:.1f} / {stats['input_tokens_min']} / {stats['input_tokens_max']}`",
            f"- KV pages (mean / median / min / max): `{stats['kv_pages_mean']:.1f} / "
            f"{stats['kv_pages_median']:.1f} / {stats['kv_pages_min']} / {stats['kv_pages_max']}`",
            f"- Round-start KV pages (mean / median / min / max): `{stats['round_start_kv_pages_mean']:.1f} / "
            f"{stats['round_start_kv_pages_median']:.1f} / {stats['round_start_kv_pages_min']} / "
            f"{stats['round_start_kv_pages_max']}`",
            "",
            "## Interpretation checklist",
            "",
            "Use the decision cases in `RUN.md`. Oracle page-scoring time is deliberately excluded; this report compares acceptance quality only.",
        ]
    )
    (results_dir / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    allocation = parse_allocation(args.dataset_allocation)
    if sum(allocation.values()) != args.num_samples:
        raise ValueError("dataset-allocation must sum to num-samples")
    if not (0.0 < args.budget_ratio <= 1.0):
        raise ValueError("budget-ratio must be in (0, 1]")
    if min(args.draft_length, args.page_size, args.max_new_tokens, args.num_samples) <= 0:
        raise ValueError("draft-length, page-size, max-new-tokens and num-samples must be positive")
    set_deterministic_seed(args.sample_seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    samples = prepare_samples(args, tokenizer, allocation)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation="eager",
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    controller = SparseKVController(args.page_size, args.budget_ratio)
    check_length = min(args.correctness_check_tokens, samples[0].input_length)
    check_ids = samples[0].input_ids[:, -check_length:].to(args.device)
    with torch.inference_mode():
        reference_logits = model(input_ids=check_ids, use_cache=False).logits[:, -1].float().cpu()
    patch_qwen3_for_sparse_drafting(model, controller)
    with torch.inference_mode():
        controller.begin_dense()
        patched_logits = model(input_ids=check_ids, use_cache=False).logits[:, -1].float().cpu()
    dense_check_max_abs = float((reference_logits - patched_logits).abs().max().item())
    dense_check_top1_equal = bool(reference_logits.argmax().item() == patched_logits.argmax().item())
    if dense_check_max_abs > args.dense_check_atol or not dense_check_top1_equal:
        raise RuntimeError(
            "patched dense attention failed equivalence check: "
            f"max_abs={dense_check_max_abs:.6f}, top1_equal={dense_check_top1_equal}"
        )

    config = vars(args).copy()
    config.update(
        {
            "model_path": str(args.model_path),
            "data_root": str(args.data_root),
            "results_dir": str(args.results_dir),
            "decoding": "greedy",
            "batch_size": 1,
            "page_score": "max token QK within page",
            "budget_rounding": "max(1, ceil(budget_ratio * round_start_pages))",
            "selection_granularity": "per layer and query head, using the corresponding GQA KV head",
            "forced_pages": "none (matches the existing page-level proxy)",
            "oracle_selection_time_in_primary_comparison": False,
            "prompt_truncation": "keep first min(256, max_context/4) tokens and the remaining tail tokens",
            "dense_patch_check": {
                "tokens": check_length,
                "max_abs_logit_error": dense_check_max_abs,
                "top1_equal": dense_check_top1_equal,
                "atol": args.dense_check_atol,
            },
            "sample_ids": [sample.sample_id for sample in samples],
        }
    )
    (args.results_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    all_results: list[tuple[PreparedSample, MethodResult]] = []
    with torch.inference_mode():
        for sample in samples:
            method_results = []
            for method in METHODS:
                set_deterministic_seed(args.sample_seed)
                result = run_method(args, sample, model, controller, tokenizer, method)
                all_results.append((sample, result))
                method_results.append(result)
            if method_results[0].generated_tokens != method_results[1].generated_tokens:
                raise RuntimeError(
                    f"dense-equivalence failure on {sample.sample_id}: the two methods committed different tokens"
                )
    summarize(args, samples, all_results)


if __name__ == "__main__":
    main()
