from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    CaptureRuntime,
    build_prompt,
    encode_prompt,
    gather_keys,
    generation_prompt,
    make_selection,
    parse_floats,
    patch_qwen3,
    rotate_query,
    rotary_for_position,
    sample_allocated,
    sample_next_token,
)


@dataclass
class StepRecord:
    step_idx: int
    prefix_len: int
    position: int
    layers: dict[int, dict[str, torch.Tensor]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure future-query oracle union and stale-ranking frontier concentration.")
    parser.add_argument("--model-path", type=Path, default=Path(r"D:\preExperiments\model\Qwen3-4B"))
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\preExperiments\ReasoningData"))
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--dataset-allocation", default="gsm8k:8,math500:8,aime2024:4")
    parser.add_argument("--sample-seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--budget-ratios", default="0.1")
    parser.add_argument("--frontier-ratios", default="0.01,0.02,0.05,0.1")
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--num-decode-steps", type=int, default=256)
    parser.add_argument("--layers", default="representative")
    parser.add_argument("--head-stride", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    return parser.parse_args()


def parse_allocation(value: str) -> dict[str, int]:
    result = {}
    for item in value.split(","):
        name, count = item.split(":")
        result[name.strip()] = int(count)
    return result


def topk(scores: torch.Tensor, count: int) -> torch.Tensor:
    return torch.topk(scores, k=min(count, scores.numel())).indices


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def prior_rankings(previous_first, source, keys, head_local, future_position, model, device):
    terminal_attention = source["attention"][head_local]
    post_mean = 0.5 * (previous_first["post_query"][head_local] + source["post_query"][head_local])
    pre_mean = 0.5 * (previous_first["pre_query"][head_local] + source["pre_query"][head_local])
    cos, sin = rotary_for_position(model, future_position, device)
    corrected = rotate_query(pre_mean.to(device), cos, sin)
    return {
        "previous_terminal_attention": terminal_attention,
        "post_rope_endpoint_mean": torch.matmul(keys, post_mean.to(device).float()),
        "pre_rope_future_corrected_endpoint_mean": torch.matmul(keys, corrected.float()),
    }


def analyze_round(
    sample,
    source_record,
    previous_first_record,
    future_records,
    key_history,
    selection,
    budget_ratios,
    frontier_ratios,
    model,
    device,
    union_writer,
    frontier_writer,
):
    head_to_local = {head: index for index, head in enumerate(selection.heads)}
    kv_to_local = {head: index for index, head in enumerate(selection.kv_heads)}
    future_terminal_position = future_records[-1].position

    for layer_idx in selection.layers:
        for head in selection.heads:
            head_local = head_to_local[head]
            kv_head = head // selection.kv_groups
            kv_local = kv_to_local[kv_head]
            keys = gather_keys(key_history[layer_idx], kv_local, source_record.prefix_len).to(device)
            source_layer = source_record.layers[layer_idx]
            previous_first_layer = previous_first_record.layers[layer_idx]
            future_attention = torch.stack(
                [record.layers[layer_idx]["attention"][head_local, : source_record.prefix_len].float() for record in future_records]
            ).to(device)
            future_importance = future_attention.sum(dim=0)
            ranking_scores = prior_rankings(
                previous_first_layer,
                source_layer,
                keys,
                head_local,
                future_terminal_position,
                model,
                device,
            )

            for budget_ratio in budget_ratios:
                budget_tokens = max(1, min(source_record.prefix_len, math.ceil(source_record.prefix_len * budget_ratio)))
                oracle_sets = [topk(row, budget_tokens) for row in future_attention]
                union_mask = torch.zeros(source_record.prefix_len, dtype=torch.bool, device=device)
                for indices in oracle_sets:
                    union_mask[indices] = True
                union_indices = union_mask.nonzero(as_tuple=False).flatten()
                terminal_indices = oracle_sets[-1]
                mean_terminal_recall = sum(
                    float(torch.isin(terminal_indices, indices).float().mean().item()) for indices in oracle_sets
                ) / len(oracle_sets)
                terminal_mass = sum(float(row[terminal_indices].sum().item()) for row in future_attention)
                oracle_mass = sum(float(row[indices].sum().item()) for row, indices in zip(future_attention, oracle_sets))
                union_writer.writerow(
                    {
                        "sample_id": sample.sample_id,
                        "dataset_name": sample.dataset_name,
                        "round_start_step": source_record.step_idx,
                        "layer": layer_idx,
                        "head": head,
                        "horizon": len(future_records),
                        "prefix_tokens": source_record.prefix_len,
                        "budget_ratio": budget_ratio,
                        "budget_tokens": budget_tokens,
                        "union_tokens": int(union_indices.numel()),
                        "union_over_terminal_budget": union_indices.numel() / budget_tokens,
                        "union_fraction_of_prefix": union_indices.numel() / source_record.prefix_len,
                        "mean_terminal_set_recall_across_queries": mean_terminal_recall,
                        "terminal_selection_attention_recovery_across_round": safe_ratio(terminal_mass, oracle_mass),
                        "mean_oracle_attention_mass": oracle_mass / len(oracle_sets),
                    }
                )

                for ranker, scores in ranking_scores.items():
                    scores = scores[: source_record.prefix_len]
                    order = torch.argsort(scores, descending=True)
                    prior_indices = order[:budget_tokens]
                    prior_mask = torch.zeros_like(union_mask)
                    prior_mask[prior_indices] = True
                    missed_mask = union_mask & ~prior_mask
                    missed_count = int(missed_mask.sum().item())
                    missed_importance = float(future_importance[missed_mask].sum().item())
                    prior_mass = sum(float(row[prior_indices].sum().item()) for row in future_attention)

                    for frontier_ratio in frontier_ratios:
                        frontier_tokens = max(1, math.ceil(source_record.prefix_len * frontier_ratio))
                        frontier_indices = order[budget_tokens : budget_tokens + frontier_tokens]
                        frontier_mask = torch.zeros_like(union_mask)
                        frontier_mask[frontier_indices] = True
                        recovered_misses = missed_mask & frontier_mask
                        frontier_writer.writerow(
                            {
                                "sample_id": sample.sample_id,
                                "dataset_name": sample.dataset_name,
                                "round_start_step": source_record.step_idx,
                                "layer": layer_idx,
                                "head": head,
                                "ranker": ranker,
                                "horizon": len(future_records),
                                "prefix_tokens": source_record.prefix_len,
                                "budget_ratio": budget_ratio,
                                "budget_tokens": budget_tokens,
                                "frontier_ratio": frontier_ratio,
                                "frontier_tokens": int(frontier_indices.numel()),
                                "future_union_tokens": int(union_indices.numel()),
                                "prior_union_recall": safe_ratio(int((union_mask & prior_mask).sum().item()), union_indices.numel()),
                                "missed_union_tokens": missed_count,
                                "frontier_miss_recall": (
                                    int(recovered_misses.sum().item()) / missed_count if missed_count > 0 else float("nan")
                                ),
                                "frontier_missed_attention_share_recovered": (
                                    float(future_importance[recovered_misses].sum().item()) / missed_importance
                                    if missed_importance > 0
                                    else float("nan")
                                ),
                                "prior_attention_recovery_across_round": safe_ratio(prior_mass, oracle_mass),
                            }
                        )


def evaluate_sample(args, sample, model, tokenizer, runtime, selection, union_writer, frontier_writer):
    device = torch.device(args.device)
    prompt = generation_prompt(tokenizer, build_prompt(sample))
    prompt_ids = encode_prompt(tokenizer, prompt, args.max_context_tokens).to(device)
    key_history = {layer: [] for layer in selection.layers}
    records: deque[StepRecord] = deque(maxlen=2 * args.horizon + 2)
    budget_ratios = parse_floats(args.budget_ratios)
    frontier_ratios = parse_floats(args.frontier_ratios)

    with torch.inference_mode():
        runtime.begin(False)
        output = model(input_ids=prompt_ids, use_cache=True)
        prefill = runtime.end()
        cache = output.past_key_values
        next_token = sample_next_token(output.logits[:, -1], args.temperature)
        for layer in selection.layers:
            key_history[layer].append(prefill[layer].post_key)

        for step_idx in range(args.num_decode_steps):
            runtime.begin(True)
            output = model(input_ids=next_token, use_cache=True, past_key_values=cache)
            captured = runtime.end()
            cache = output.past_key_values
            sampled = sample_next_token(output.logits[:, -1], args.temperature)
            prefix_len = prompt_ids.shape[1] + step_idx + 1
            layer_records = {}
            position = None
            for layer in selection.layers:
                capture = captured[layer]
                key_history[layer].append(capture.post_key)
                position = int(capture.positions[-1].item()) if position is None else position
                layer_records[layer] = {
                    "pre_query": capture.pre_query[:, 0].float(),
                    "post_query": capture.post_query[:, 0].float(),
                    "attention": capture.attention[:, 0, :prefix_len].float(),
                }
            records.append(StepRecord(step_idx, prefix_len, int(position), layer_records))
            record_list = list(records)
            source_index = len(record_list) - args.horizon - 1
            previous_first_index = source_index - args.horizon
            if previous_first_index >= 0:
                source = record_list[source_index]
                if source.step_idx % args.horizon == 0:
                    future = record_list[source_index + 1 :]
                    if len(future) == args.horizon:
                        analyze_round(
                            sample,
                            source,
                            record_list[previous_first_index],
                            future,
                            key_history,
                            selection,
                            budget_ratios,
                            frontier_ratios,
                            model,
                            device,
                            union_writer,
                            frontier_writer,
                        )
            next_token = sampled
            if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
                break


def summarize(results_dir: Path) -> None:
    union = pd.read_csv(results_dir / "union_metrics.csv")
    frontier = pd.read_csv(results_dir / "frontier_metrics.csv")
    union_metrics = dict(
        num_rows=("sample_id", "size"),
        num_samples=("sample_id", "nunique"),
        union_over_terminal_budget=("union_over_terminal_budget", "mean"),
        union_fraction_of_prefix=("union_fraction_of_prefix", "mean"),
        mean_terminal_set_recall_across_queries=("mean_terminal_set_recall_across_queries", "mean"),
        terminal_selection_attention_recovery_across_round=("terminal_selection_attention_recovery_across_round", "mean"),
    )
    frontier_metrics = dict(
        num_rows=("sample_id", "size"),
        num_samples=("sample_id", "nunique"),
        prior_union_recall=("prior_union_recall", "mean"),
        frontier_miss_recall=("frontier_miss_recall", "mean"),
        frontier_missed_attention_share_recovered=("frontier_missed_attention_share_recovered", "mean"),
        prior_attention_recovery_across_round=("prior_attention_recovery_across_round", "mean"),
    )
    union.groupby(["horizon", "budget_ratio"], as_index=False).agg(**union_metrics).to_csv(
        results_dir / "summary_union.csv", index=False
    )
    union.groupby(["dataset_name", "horizon", "budget_ratio"], as_index=False).agg(**union_metrics).to_csv(
        results_dir / "summary_union_by_dataset.csv", index=False
    )
    frontier_groups = ["ranker", "horizon", "budget_ratio", "frontier_ratio"]
    frontier.groupby(frontier_groups, as_index=False).agg(**frontier_metrics).to_csv(
        results_dir / "summary_frontier.csv", index=False
    )
    frontier.groupby(["dataset_name"] + frontier_groups, as_index=False).agg(**frontier_metrics).to_csv(
        results_dir / "summary_frontier_by_dataset.csv", index=False
    )


def main() -> None:
    args = parse_args()
    allocation = parse_allocation(args.dataset_allocation)
    if sum(allocation.values()) != args.num_samples:
        raise ValueError("dataset allocation must sum to num-samples")
    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, attn_implementation="eager", dtype=dtype, low_cpu_mem_usage=True
    ).to(args.device).eval()
    selection = make_selection(model, args.layers, args.head_stride)
    runtime = CaptureRuntime(selection)
    patch_qwen3(model, runtime)
    samples = sample_allocated(args.data_root, allocation, args.sample_seed)
    config = vars(args).copy()
    config.update({"model_path": str(args.model_path), "data_root": str(args.data_root), "results_dir": str(args.results_dir), "layers_selected": selection.layers, "heads_selected": selection.heads, "sample_ids": [s.sample_id for s in samples]})
    (args.results_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    union_fields = ["sample_id", "dataset_name", "round_start_step", "layer", "head", "horizon", "prefix_tokens", "budget_ratio", "budget_tokens", "union_tokens", "union_over_terminal_budget", "union_fraction_of_prefix", "mean_terminal_set_recall_across_queries", "terminal_selection_attention_recovery_across_round", "mean_oracle_attention_mass"]
    frontier_fields = ["sample_id", "dataset_name", "round_start_step", "layer", "head", "ranker", "horizon", "prefix_tokens", "budget_ratio", "budget_tokens", "frontier_ratio", "frontier_tokens", "future_union_tokens", "prior_union_recall", "missed_union_tokens", "frontier_miss_recall", "frontier_missed_attention_share_recovered", "prior_attention_recovery_across_round"]
    with (args.results_dir / "union_metrics.csv").open("w", newline="", encoding="utf-8") as union_handle, (args.results_dir / "frontier_metrics.csv").open("w", newline="", encoding="utf-8") as frontier_handle:
        union_writer, frontier_writer = csv.DictWriter(union_handle, fieldnames=union_fields), csv.DictWriter(frontier_handle, fieldnames=frontier_fields)
        union_writer.writeheader()
        frontier_writer.writeheader()
        for sample in samples:
            evaluate_sample(args, sample, model, tokenizer, runtime, selection, union_writer, frontier_writer)
            union_handle.flush()
            frontier_handle.flush()
    summarize(args.results_dir)


if __name__ == "__main__":
    main()
