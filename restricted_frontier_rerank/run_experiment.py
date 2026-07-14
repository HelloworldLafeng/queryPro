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
import torch.nn.functional as F
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
    parser = argparse.ArgumentParser(description="Compare restricted actual-query page reranking with full query-aware page selection.")
    parser.add_argument("--model-path", type=Path, default=Path(r"D:\preExperiments\model\Qwen3-4B"))
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\preExperiments\ReasoningData"))
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--dataset-allocation", default="gsm8k:8,math500:8,aime2024:4")
    parser.add_argument("--sample-seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--budget-ratios", default="0.1")
    parser.add_argument("--frontier-page-ratios", default="0.01,0.02,0.05")
    parser.add_argument("--near-mass-threshold", type=float, default=0.985)
    parser.add_argument("--near-page-recall-threshold", type=float, default=0.9)
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


def page_sum(values: torch.Tensor, page_size: int) -> torch.Tensor:
    padding = (-values.numel()) % page_size
    return F.pad(values, (0, padding)).reshape(-1, page_size).sum(dim=-1)


def page_max(values: torch.Tensor, page_size: int) -> torch.Tensor:
    padding = (-values.numel()) % page_size
    return F.pad(values, (0, padding), value=float("-inf")).reshape(-1, page_size).max(dim=-1).values


def select_from_pool(scores: torch.Tensor, pool: torch.Tensor, count: int) -> torch.Tensor:
    if pool.numel() == 0:
        raise ValueError("candidate pool is empty")
    local = torch.topk(scores[pool], k=min(count, pool.numel())).indices
    return pool[local]


def overlap_recall(selected: torch.Tensor, reference: torch.Tensor) -> float:
    """Fraction of the reference set contained in the selected/candidate set."""
    return float(torch.isin(reference, selected).float().mean().item())


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    a_set, b_set = set(a.tolist()), set(b.tolist())
    return len(a_set & b_set) / len(a_set | b_set) if a_set or b_set else 1.0


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def prior_page_rankings(previous_first, source, keys, head_local, terminal_position, page_size, model, device):
    post_mean = 0.5 * (previous_first["post_query"][head_local] + source["post_query"][head_local])
    pre_mean = 0.5 * (previous_first["pre_query"][head_local] + source["pre_query"][head_local])
    cos, sin = rotary_for_position(model, terminal_position, device)
    corrected = rotate_query(pre_mean.to(device), cos, sin)
    return {
        "previous_terminal_attention": page_sum(source["attention"][head_local], page_size),
        "post_rope_endpoint_mean": page_max(torch.matmul(keys, post_mean.to(device).float()), page_size),
        "pre_rope_future_corrected_endpoint_mean": page_max(torch.matmul(keys, corrected.float()), page_size),
    }


def analyze_round(sample, source_record, previous_first_record, future_records, key_history, selection, args, model, device, writer):
    head_to_local = {head: index for index, head in enumerate(selection.heads)}
    kv_to_local = {head: index for index, head in enumerate(selection.kv_heads)}
    budget_ratios = parse_floats(args.budget_ratios)
    frontier_ratios = parse_floats(args.frontier_page_ratios)

    for layer_idx in selection.layers:
        for head in selection.heads:
            head_local = head_to_local[head]
            kv_local = kv_to_local[head // selection.kv_groups]
            keys = gather_keys(key_history[layer_idx], kv_local, source_record.prefix_len).to(device)
            num_pages = math.ceil(source_record.prefix_len / args.page_size)
            prior_scores = prior_page_rankings(
                previous_first_record.layers[layer_idx],
                source_record.layers[layer_idx],
                keys,
                head_local,
                future_records[-1].position,
                args.page_size,
                model,
                device,
            )

            for draft_offset, future_record in enumerate(future_records, start=1):
                future_layer = future_record.layers[layer_idx]
                token_attention = future_layer["attention"][head_local, : source_record.prefix_len].to(device).float()
                page_attention = page_sum(token_attention, args.page_size)
                actual_query = future_layer["post_query"][head_local].to(device).float()
                full_query_scores = page_max(torch.matmul(keys, actual_query), args.page_size)

                for budget_ratio in budget_ratios:
                    budget_pages = max(1, min(num_pages, math.ceil(num_pages * budget_ratio)))
                    oracle_pages = torch.topk(page_attention, k=budget_pages).indices
                    full_query_pages = torch.topk(full_query_scores, k=budget_pages).indices
                    oracle_mass = float(page_attention[oracle_pages].sum().item())
                    full_query_mass = float(page_attention[full_query_pages].sum().item())

                    for ranker, scores in prior_scores.items():
                        order = torch.argsort(scores[:num_pages], descending=True)
                        base_pages = order[:budget_pages]
                        for frontier_ratio in frontier_ratios:
                            frontier_pages_count = max(1, math.ceil(num_pages * frontier_ratio))
                            frontier_pages = order[budget_pages : budget_pages + frontier_pages_count]
                            pool = torch.unique(torch.cat((base_pages, frontier_pages)), sorted=False)
                            restricted_pages = select_from_pool(full_query_scores, pool, budget_pages)
                            upper_bound_pages = select_from_pool(page_attention, pool, budget_pages)
                            restricted_mass = float(page_attention[restricted_pages].sum().item())
                            upper_bound_mass = float(page_attention[upper_bound_pages].sum().item())
                            recovery_vs_full = ratio(restricted_mass, full_query_mass)
                            recall_vs_full = overlap_recall(restricted_pages, full_query_pages)
                            writer.writerow(
                                {
                                    "sample_id": sample.sample_id,
                                    "dataset_name": sample.dataset_name,
                                    "round_start_step": source_record.step_idx,
                                    "draft_offset": draft_offset,
                                    "layer": layer_idx,
                                    "head": head,
                                    "ranker": ranker,
                                    "page_size": args.page_size,
                                    "prefix_pages": num_pages,
                                    "budget_ratio": budget_ratio,
                                    "budget_pages": budget_pages,
                                    "frontier_page_ratio": frontier_ratio,
                                    "frontier_pages": int(frontier_pages.numel()),
                                    "candidate_pool_pages": int(pool.numel()),
                                    "candidate_pool_ratio": pool.numel() / num_pages,
                                    "candidate_recall_of_full_query_pages": overlap_recall(pool, full_query_pages),
                                    "restricted_page_recall_vs_full_query": recall_vs_full,
                                    "restricted_jaccard_vs_full_query": jaccard(restricted_pages, full_query_pages),
                                    "restricted_attention_mass": restricted_mass,
                                    "full_query_attention_mass": full_query_mass,
                                    "oracle_attention_mass": oracle_mass,
                                    "attention_recovery_vs_full_query": recovery_vs_full,
                                    "attention_recovery_vs_oracle": ratio(restricted_mass, oracle_mass),
                                    "full_query_recovery_vs_oracle": ratio(full_query_mass, oracle_mass),
                                    "candidate_oracle_upper_bound_recovery": ratio(upper_bound_mass, oracle_mass),
                                    "near_full_query_mass": int(recovery_vs_full >= args.near_mass_threshold),
                                    "near_full_query_joint": int(
                                        recovery_vs_full >= args.near_mass_threshold
                                        and recall_vs_full >= args.near_page_recall_threshold
                                    ),
                                }
                            )


def evaluate_sample(args, sample, model, tokenizer, runtime, selection, writer):
    device = torch.device(args.device)
    prompt_ids = encode_prompt(tokenizer, generation_prompt(tokenizer, build_prompt(sample)), args.max_context_tokens).to(device)
    key_history = {layer: [] for layer in selection.layers}
    records: deque[StepRecord] = deque(maxlen=2 * args.horizon + 2)
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
            layers, position = {}, None
            for layer in selection.layers:
                capture = captured[layer]
                key_history[layer].append(capture.post_key)
                position = int(capture.positions[-1].item()) if position is None else position
                layers[layer] = {
                    "pre_query": capture.pre_query[:, 0].float(),
                    "post_query": capture.post_query[:, 0].float(),
                    "attention": capture.attention[:, 0, :prefix_len].float(),
                }
            records.append(StepRecord(step_idx, prefix_len, int(position), layers))
            record_list = list(records)
            source_index = len(record_list) - args.horizon - 1
            previous_first_index = source_index - args.horizon
            if previous_first_index >= 0:
                source = record_list[source_index]
                future = record_list[source_index + 1 :]
                if source.step_idx % args.horizon == 0 and len(future) == args.horizon:
                    analyze_round(sample, source, record_list[previous_first_index], future, key_history, selection, args, model, device, writer)
            next_token = sampled
            if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
                break


def summarize(results_dir: Path) -> None:
    frame = pd.read_csv(results_dir / "rerank_metrics.csv")
    metrics = {
        "num_rows": ("sample_id", "size"),
        "num_samples": ("sample_id", "nunique"),
        "candidate_recall_of_full_query_pages": ("candidate_recall_of_full_query_pages", "mean"),
        "restricted_page_recall_vs_full_query": ("restricted_page_recall_vs_full_query", "mean"),
        "attention_recovery_vs_full_query": ("attention_recovery_vs_full_query", "mean"),
        "attention_recovery_vs_oracle": ("attention_recovery_vs_oracle", "mean"),
        "full_query_recovery_vs_oracle": ("full_query_recovery_vs_oracle", "mean"),
        "candidate_oracle_upper_bound_recovery": ("candidate_oracle_upper_bound_recovery", "mean"),
        "near_full_query_mass_rate": ("near_full_query_mass", "mean"),
        "near_full_query_joint_rate": ("near_full_query_joint", "mean"),
    }
    groups = ["ranker", "budget_ratio", "frontier_page_ratio"]
    frame.groupby(groups, as_index=False).agg(**metrics).to_csv(results_dir / "summary_overall.csv", index=False)
    frame.groupby(groups + ["draft_offset"], as_index=False).agg(**metrics).to_csv(results_dir / "summary_by_draft_offset.csv", index=False)
    frame.groupby(groups + ["dataset_name"], as_index=False).agg(**metrics).to_csv(results_dir / "summary_by_dataset.csv", index=False)


def main() -> None:
    args = parse_args()
    allocation = parse_allocation(args.dataset_allocation)
    if sum(allocation.values()) != args.num_samples:
        raise ValueError("dataset allocation must sum to num-samples")
    if args.horizon <= 0 or args.page_size <= 0:
        raise ValueError("horizon and page-size must be positive")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, attn_implementation="eager", dtype=dtype, low_cpu_mem_usage=True).to(args.device).eval()
    selection = make_selection(model, args.layers, args.head_stride)
    runtime = CaptureRuntime(selection)
    patch_qwen3(model, runtime)
    samples = sample_allocated(args.data_root, allocation, args.sample_seed)
    config = vars(args).copy()
    config.update({"model_path": str(args.model_path), "data_root": str(args.data_root), "results_dir": str(args.results_dir), "layers_selected": selection.layers, "heads_selected": selection.heads, "sample_ids": [s.sample_id for s in samples]})
    (args.results_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    fields = ["sample_id", "dataset_name", "round_start_step", "draft_offset", "layer", "head", "ranker", "page_size", "prefix_pages", "budget_ratio", "budget_pages", "frontier_page_ratio", "frontier_pages", "candidate_pool_pages", "candidate_pool_ratio", "candidate_recall_of_full_query_pages", "restricted_page_recall_vs_full_query", "restricted_jaccard_vs_full_query", "restricted_attention_mass", "full_query_attention_mass", "oracle_attention_mass", "attention_recovery_vs_full_query", "attention_recovery_vs_oracle", "full_query_recovery_vs_oracle", "candidate_oracle_upper_bound_recovery", "near_full_query_mass", "near_full_query_joint"]
    with (args.results_dir / "rerank_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            evaluate_sample(args, sample, model, tokenizer, runtime, selection, writer)
            handle.flush()
    summarize(args.results_dir)


if __name__ == "__main__":
    main()
