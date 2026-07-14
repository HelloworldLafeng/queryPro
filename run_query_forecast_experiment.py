from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from query_forecast import (
    AttentionCaptureRuntime,
    SelectionSpec,
    TCNTrainingConfig,
    TinyTCNPredictor,
    build_prompt,
    patch_qwen3_attention,
    sample_experiment_data,
    sample_reasoning_data_allocated,
    train_tcn_model,
)
from query_forecast.predictors import ReservoirBuffer


QUERY_TYPES = ("pre_rope_corrected", "post_rope")
FORECAST_METHODS = ("persistence_query", "linear_drift", "ema_drift", "tiny_tcn")


@dataclass
class StepRecord:
    step_idx: int
    prefix_len: int
    token_id: int
    position_id: int
    per_layer: dict[int, dict[str, torch.Tensor]]


class BufferedMetricsWriter:
    def __init__(self, writer: csv.DictWriter, handle, flush_every: int = 4096):
        self.writer = writer
        self.handle = handle
        self.flush_every = flush_every
        self.buffer: list[dict[str, Any]] = []

    def write(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.writer.writerows(self.buffer)
        self.handle.flush()
        self.buffer.clear()


class SummaryAccumulator:
    def __init__(self) -> None:
        self.sum_attention_recovery = 0.0
        self.sum_token_recall = 0.0
        self.sum_query_cosine = 0.0
        self.sum_jaccard = 0.0
        self.sum_changed = 0.0
        self.count = 0
        self.query_cosine_count = 0

    def update(self, row: dict[str, Any]) -> None:
        self.sum_attention_recovery += float(row["attention_recovery"])
        self.sum_token_recall += float(row["token_recall"])
        self.sum_jaccard += float(row["jaccard_reuse_vs_oracle"])
        self.sum_changed += float(row["selection_changed"])
        self.count += 1
        if row["query_cosine"] == row["query_cosine"]:
            self.sum_query_cosine += float(row["query_cosine"])
            self.query_cosine_count += 1

    def as_dict(self) -> dict[str, float]:
        return {
            "attention_recovery": self.sum_attention_recovery / self.count if self.count else 0.0,
            "token_recall": self.sum_token_recall / self.count if self.count else 0.0,
            "query_cosine": self.sum_query_cosine / self.query_cosine_count if self.query_cosine_count else float("nan"),
            "jaccard_reuse_vs_oracle": self.sum_jaccard / self.count if self.count else 0.0,
            "changed_step_ratio": self.sum_changed / self.count if self.count else 0.0,
            "num_rows": self.count,
        }


class ChangedRatioAccumulator:
    def __init__(self) -> None:
        self.num_steps = 0
        self.num_changed_steps = 0

    def update(self, changed: bool) -> None:
        self.num_steps += 1
        self.num_changed_steps += int(changed)

    def as_dict(self) -> dict[str, float]:
        return {
            "num_steps": self.num_steps,
            "num_changed_steps": self.num_changed_steps,
            "changed_ratio": self.num_changed_steps / self.num_steps if self.num_steps else 0.0,
        }


class SummaryManager:
    def __init__(self) -> None:
        self.all_steps: dict[tuple, SummaryAccumulator] = defaultdict(SummaryAccumulator)
        self.changed_steps: dict[tuple, SummaryAccumulator] = defaultdict(SummaryAccumulator)
        self.by_dataset: dict[tuple, SummaryAccumulator] = defaultdict(SummaryAccumulator)
        self.by_layer: dict[tuple, SummaryAccumulator] = defaultdict(SummaryAccumulator)
        self.by_query_type: dict[tuple, SummaryAccumulator] = defaultdict(SummaryAccumulator)
        self.changed_ratio_by_dataset: dict[tuple, ChangedRatioAccumulator] = defaultdict(ChangedRatioAccumulator)
        self.changed_ratio_by_layer: dict[tuple, ChangedRatioAccumulator] = defaultdict(ChangedRatioAccumulator)

    def update_metric(self, row: dict[str, Any]) -> None:
        base_key = (row["query_type"], row["method"], row["horizon_L"], row["budget_ratio"])
        self.all_steps[base_key].update(row)
        self.by_query_type[("all",) + base_key].update(row)
        self.by_dataset[("all", row["dataset_name"]) + base_key].update(row)
        self.by_layer[("all", row["layer"]) + base_key].update(row)
        if row["selection_changed"]:
            self.changed_steps[base_key].update(row)
            self.by_query_type[("changed",) + base_key].update(row)
            self.by_dataset[("changed", row["dataset_name"]) + base_key].update(row)
            self.by_layer[("changed", row["layer"]) + base_key].update(row)

    def update_changed_context(
        self,
        dataset_name: str,
        layer: int,
        horizon_L: int,
        budget_ratio: float,
        changed: bool,
    ) -> None:
        self.changed_ratio_by_dataset[(dataset_name, horizon_L, budget_ratio)].update(changed)
        self.changed_ratio_by_layer[(layer, horizon_L, budget_ratio)].update(changed)

    def _summary_to_frame(self, table: dict[tuple, SummaryAccumulator], key_names: list[str]) -> pd.DataFrame:
        rows = []
        for key, acc in table.items():
            row = dict(zip(key_names, key))
            row.update(acc.as_dict())
            rows.append(row)
        return pd.DataFrame(rows)

    def _changed_ratio_frame(self, table: dict[tuple, ChangedRatioAccumulator], key_names: list[str]) -> pd.DataFrame:
        rows = []
        for key, acc in table.items():
            row = dict(zip(key_names, key))
            row.update(acc.as_dict())
            rows.append(row)
        return pd.DataFrame(rows)

    def write(self, results_dir: Path) -> None:
        summary_all = self._summary_to_frame(self.all_steps, ["query_type", "method", "horizon_L", "budget_ratio"])
        summary_changed = self._summary_to_frame(self.changed_steps, ["query_type", "method", "horizon_L", "budget_ratio"])
        summary_query_type = self._summary_to_frame(
            self.by_query_type,
            ["step_scope", "query_type", "method", "horizon_L", "budget_ratio"],
        )
        summary_dataset = self._summary_to_frame(
            self.by_dataset,
            ["step_scope", "dataset_name", "query_type", "method", "horizon_L", "budget_ratio"],
        )
        summary_layer = self._summary_to_frame(
            self.by_layer,
            ["step_scope", "layer", "query_type", "method", "horizon_L", "budget_ratio"],
        )
        changed_dataset = self._changed_ratio_frame(
            self.changed_ratio_by_dataset,
            ["dataset_name", "horizon_L", "budget_ratio"],
        )
        changed_layer = self._changed_ratio_frame(
            self.changed_ratio_by_layer,
            ["layer", "horizon_L", "budget_ratio"],
        )

        summary_all.to_csv(results_dir / "summary_all_steps.csv", index=False)
        summary_changed.to_csv(results_dir / "summary_changed_steps.csv", index=False)
        summary_query_type.to_csv(results_dir / "summary_by_query_type.csv", index=False)
        summary_dataset.to_csv(results_dir / "summary_by_dataset.csv", index=False)
        summary_layer.to_csv(results_dir / "summary_by_layer.csv", index=False)
        changed_dataset.to_csv(results_dir / "changed_step_ratio_by_dataset.csv", index=False)
        changed_layer.to_csv(results_dir / "changed_step_ratio_by_layer.csv", index=False)


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query forecast pre-experiment for KV block selection.")
    parser.add_argument("--model-path", type=Path, default=Path(r"D:\preExperiments\model\Qwen3-4B"))
    parser.add_argument("--dataset-family", type=str, choices=("longbench", "reasoning"), default="reasoning")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=7)
    parser.add_argument("--datasets", type=str, default="gsm8k,math500,aime2024")
    parser.add_argument("--history", type=int, default=16)
    parser.add_argument("--horizons", type=str, default="8")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--budget-blocks", type=str, default="32,64,128")
    parser.add_argument("--budget-ratios", type=str, default=None)
    parser.add_argument("--query-types", type=str, default="pre_rope_corrected,post_rope")
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--num-decode-steps", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layers", type=str, default="representative")
    parser.add_argument("--head-stride", type=int, default=8)
    parser.add_argument("--train-tcn", action="store_true")
    parser.add_argument("--tcn-epochs", type=int, default=4)
    parser.add_argument("--tcn-max-examples", type=int, default=12000)
    parser.add_argument("--summary-jsonl", action="store_true")
    parser.add_argument("--dataset-allocation", type=str, default="gsm8k:8,math500:8,aime2024:4")
    return parser.parse_args()


def get_default_data_root(dataset_family: str) -> Path:
    if dataset_family == "longbench":
        return Path(r"D:\preExperiments\LongBench")
    if dataset_family == "reasoning":
        return Path(r"D:\preExperiments\ReasoningData")
    raise ValueError(dataset_family)


def get_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def select_layers(num_layers: int, spec: str) -> list[int]:
    if spec == "all":
        return list(range(num_layers))
    if spec == "representative":
        points = [
            0,
            num_layers // 6,
            (2 * num_layers) // 6,
            (3 * num_layers) // 6,
            (4 * num_layers) // 6,
            (5 * num_layers) // 6,
            num_layers - 1,
        ]
        return sorted(set(min(num_layers - 1, point) for point in points))
    values = sorted(set(parse_int_list(spec)))
    if not values:
        raise ValueError("No valid layers selected.")
    return values


def parse_dataset_allocation(spec: str) -> dict[str, int]:
    allocation: dict[str, int] = {}
    if not spec.strip():
        return allocation
    for item in spec.split(","):
        name, count = item.split(":")
        allocation[name.strip()] = int(count.strip())
    return allocation


def make_selection_spec(model, layer_spec: str, head_stride: int) -> SelectionSpec:
    config = model.config
    layers = select_layers(config.num_hidden_layers, layer_spec)
    heads = list(range(0, config.num_attention_heads, head_stride))
    kv_groups = config.num_attention_heads // config.num_key_value_heads
    kv_heads = sorted({head // kv_groups for head in heads})
    return SelectionSpec(layers=layers, heads=heads, kv_heads=kv_heads, num_key_value_groups=kv_groups)


def build_generation_prompt(tokenizer, raw_prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": raw_prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return raw_prompt


def encode_prompt(tokenizer, prompt: str, max_context_tokens: int) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_context_tokens)
    return encoded["input_ids"]


def sample_next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_to_query(query: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (query * cos) + (rotate_half(query) * sin)


def kv_head_for_attention_head(head: int, kv_groups: int) -> int:
    return head // kv_groups


def gather_key_history(chunks: list[torch.Tensor], kv_head_local_idx: int, prefix_len: int) -> torch.Tensor:
    tensors = []
    consumed = 0
    for chunk in chunks:
        take = min(chunk.shape[1], prefix_len - consumed)
        if take <= 0:
            break
        tensors.append(chunk[kv_head_local_idx, :take].to(dtype=torch.float32))
        consumed += take
        if consumed >= prefix_len:
            break
    if not tensors:
        raise RuntimeError("No key history available for requested prefix.")
    return torch.cat(tensors, dim=0)


def rotary_cos_sin_for_positions(model, positions: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    position_ids = positions.unsqueeze(0).to(device)
    dummy = torch.zeros((1, positions.numel(), model.config.hidden_size), device=device, dtype=model.dtype)
    cos, sin = model.model.rotary_emb(dummy, position_ids)
    return cos.squeeze(0).to(dtype=torch.float32), sin.squeeze(0).to(dtype=torch.float32)


def resolve_budget_ratios(args: argparse.Namespace) -> list[float]:
    if args.budget_ratios:
        ratios = parse_float_list(args.budget_ratios)
    else:
        ratios = [budget_blocks * args.block_size / args.max_context_tokens for budget_blocks in parse_int_list(args.budget_blocks)]
    cleaned = []
    for ratio in ratios:
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError(f"Budget ratio must be in (0, 1], got {ratio}.")
        cleaned.append(ratio)
    return cleaned


def budget_tokens_from_ratio(prefix_len: int, budget_ratio: float) -> int:
    return max(1, min(prefix_len, math.ceil(prefix_len * budget_ratio)))


def topk_token_indices(scores: torch.Tensor, budget_tokens: int) -> torch.Tensor:
    topk = min(budget_tokens, scores.numel())
    return torch.topk(scores, k=topk).indices


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(dtype=torch.float32)
    b = b.to(dtype=torch.float32)
    denom = a.norm() * b.norm()
    if denom.item() == 0:
        return 0.0
    return float(torch.dot(a, b).item() / denom.item())


def ema_delta(history: list[torch.Tensor], decay: float = 0.6) -> torch.Tensor:
    if len(history) < 2:
        return torch.zeros_like(history[-1])
    delta = history[1] - history[0]
    for idx in range(2, len(history)):
        delta = decay * delta + (1.0 - decay) * (history[idx] - history[idx - 1])
    return delta


def history_kind_for_query_type(query_type: str) -> str:
    if query_type == "post_rope":
        return "post_query"
    if query_type == "pre_rope_corrected":
        return "pre_query"
    raise KeyError(query_type)


def to_scoring_query(
    query_type: str,
    base_query: torch.Tensor,
    future_position_id: int,
    model,
    device: torch.device,
) -> torch.Tensor:
    if query_type == "post_rope":
        return base_query.to(dtype=torch.float32)
    if query_type != "pre_rope_corrected":
        raise KeyError(query_type)
    positions = torch.tensor([future_position_id], device=device, dtype=torch.long)
    cos, sin = rotary_cos_sin_for_positions(model, positions, device=device)
    return apply_rope_to_query(base_query.to(dtype=torch.float32), cos[0], sin[0])


def jaccard_similarity(a: list[int], b: list[int]) -> float:
    a_set = set(a)
    b_set = set(b)
    union = a_set | b_set
    if not union:
        return 1.0
    return len(a_set & b_set) / len(union)


def run_collection_pass(
    args: argparse.Namespace,
    model,
    tokenizer,
    samples,
    selection: SelectionSpec,
    query_types: list[str],
    horizons: list[int],
) -> dict[tuple[str, int], ReservoirBuffer]:
    buffers = {
        (query_type, horizon): ReservoirBuffer(max_examples=args.tcn_max_examples, seed=args.sample_seed + horizon)
        for query_type in query_types
        for horizon in horizons
    }
    runtime = model._query_forecast_runtime
    device = torch.device(args.device)
    eos_token_id = tokenizer.eos_token_id
    max_horizon = max(horizons)

    for sample in samples:
        prompt = build_generation_prompt(tokenizer, build_prompt(sample))
        prompt_ids = encode_prompt(tokenizer, prompt, args.max_context_tokens).to(device)
        with torch.inference_mode():
            runtime.begin_step(collect_attn_weights=False)
            outputs = model(input_ids=prompt_ids, use_cache=True)
            runtime.end_step()
            next_token = sample_next_token(outputs.logits[:, -1, :], args.temperature)
            past_key_values = outputs.past_key_values
            histories = {
                (layer_idx, head_local, query_type): deque(maxlen=args.history + max_horizon + 4)
                for layer_idx in selection.layers
                for head_local in range(len(selection.heads))
                for query_type in query_types
            }
            for _ in range(args.num_decode_steps):
                runtime.begin_step(collect_attn_weights=False)
                outputs = model(input_ids=next_token, use_cache=True, past_key_values=past_key_values)
                captured = runtime.end_step()
                past_key_values = outputs.past_key_values
                sampled_token = sample_next_token(outputs.logits[:, -1, :], args.temperature)
                for layer_idx in selection.layers:
                    layer_capture = captured[layer_idx]
                    tensors = {
                        "pre_rope_corrected": layer_capture.pre_query[:, 0, :].to(dtype=torch.float32),
                        "post_rope": layer_capture.post_query[:, 0, :].to(dtype=torch.float32),
                    }
                    for query_type, tensor in tensors.items():
                        for head_local in range(tensor.shape[0]):
                            key = (layer_idx, head_local, query_type)
                            history = histories[key]
                            history.append(tensor[head_local].clone())
                            for horizon in horizons:
                                if len(history) < args.history + horizon:
                                    continue
                                history_list = list(history)
                                source_idx = len(history_list) - horizon - 1
                                hist = torch.stack(history_list[source_idx - args.history + 1 : source_idx + 1], dim=0)
                                target = history_list[-1] - history_list[source_idx]
                                buffers[(query_type, horizon)].add(hist, target)
                next_token = sampled_token
                if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                    break
    return buffers


def evaluate_sample(
    args: argparse.Namespace,
    model,
    tokenizer,
    sample,
    selection: SelectionSpec,
    query_types: list[str],
    horizons: list[int],
    tcn_models: dict[tuple[str, int], TinyTCNPredictor],
    metrics_writer: BufferedMetricsWriter,
    summary_manager: SummaryManager,
    jsonl_handle,
) -> None:
    runtime = model._query_forecast_runtime
    device = torch.device(args.device)
    eos_token_id = tokenizer.eos_token_id
    budget_ratios = resolve_budget_ratios(args)
    head_to_local = {head: idx for idx, head in enumerate(selection.heads)}
    kv_to_local = {head: idx for idx, head in enumerate(selection.kv_heads)}
    max_horizon = max(horizons)

    prompt = build_generation_prompt(tokenizer, build_prompt(sample))
    prompt_ids = encode_prompt(tokenizer, prompt, args.max_context_tokens).to(device)

    key_history = {layer_idx: [] for layer_idx in selection.layers}
    step_records: deque[StepRecord] = deque(maxlen=args.history + max_horizon + 2)

    with torch.inference_mode():
        runtime.begin_step(collect_attn_weights=False)
        outputs = model(input_ids=prompt_ids, use_cache=True)
        prefill_capture = runtime.end_step()
        past_key_values = outputs.past_key_values
        next_token = sample_next_token(outputs.logits[:, -1, :], args.temperature)
        for layer_idx in selection.layers:
            capture = prefill_capture[layer_idx]
            key_history[layer_idx].append(capture.post_key)

        for step_idx in range(args.num_decode_steps):
            runtime.begin_step(collect_attn_weights=True)
            outputs = model(input_ids=next_token, use_cache=True, past_key_values=past_key_values)
            captured = runtime.end_step()
            past_key_values = outputs.past_key_values
            sampled_token = sample_next_token(outputs.logits[:, -1, :], args.temperature)

            current_queries = {}
            current_attn = {}
            current_position_id = None
            prefix_len = prompt_ids.shape[1] + step_idx + 1

            for layer_idx in selection.layers:
                layer_capture = captured[layer_idx]
                if layer_capture.positions.numel() != 1:
                    raise RuntimeError("Generation step expected a single cache position.")
                layer_position = int(layer_capture.positions[-1].item())
                if current_position_id is None:
                    current_position_id = layer_position
                elif current_position_id != layer_position:
                    raise RuntimeError("Inconsistent future position across layers.")

                key_history[layer_idx].append(layer_capture.post_key)
                pre_query = layer_capture.pre_query[:, 0, :].to(dtype=torch.float32)
                post_query = layer_capture.post_query[:, 0, :].to(dtype=torch.float32)
                attn_weights = layer_capture.attn_weights[:, 0, :prefix_len].to(dtype=torch.float32)
                current_queries[layer_idx] = {
                    "pre_query": pre_query,
                    "post_query": post_query,
                    "attn_weights": attn_weights,
                }
                current_attn[layer_idx] = attn_weights

            current_record = StepRecord(
                step_idx=step_idx,
                prefix_len=prefix_len,
                token_id=int(next_token.item()),
                position_id=int(current_position_id),
                per_layer=current_queries,
            )
            step_records.append(current_record)
            record_list = list(step_records)

            for horizon in horizons:
                source_local_idx = len(record_list) - horizon - 1
                if source_local_idx < args.history - 1:
                    continue
                source = record_list[source_local_idx]
                history_records = record_list[source_local_idx - args.history + 1 : source_local_idx + 1]
                current = current_record

                for layer_idx in selection.layers:
                    for head in selection.heads:
                        head_local = head_to_local[head]
                        kv_head = kv_head_for_attention_head(head, selection.num_key_value_groups)
                        kv_local = kv_to_local[kv_head]

                        history_by_type = {}
                        for query_type in query_types:
                            tensor_name = history_kind_for_query_type(query_type)
                            history_by_type[query_type] = [record.per_layer[layer_idx][tensor_name][head_local] for record in history_records]

                        keys = gather_key_history(key_history[layer_idx], kv_local, source.prefix_len)
                        attn_prefix = current_attn[layer_idx][head_local, : source.prefix_len]
                        source_attn_prefix = source.per_layer[layer_idx]["attn_weights"][head_local, : source.prefix_len]

                        for budget_ratio in budget_ratios:
                            budget_tokens = budget_tokens_from_ratio(source.prefix_len, budget_ratio)
                            reuse_blocks_tensor = topk_token_indices(source_attn_prefix, budget_tokens)
                            oracle_blocks_tensor = topk_token_indices(attn_prefix, budget_tokens)
                            reuse_blocks = reuse_blocks_tensor.tolist()
                            oracle_blocks_budget = oracle_blocks_tensor.tolist()
                            oracle_blocks_tensor = torch.tensor(
                                oracle_blocks_budget,
                                device=attn_prefix.device,
                                dtype=torch.long,
                            )
                            reuse_attention_recovery = float(
                                attn_prefix[reuse_blocks_tensor].sum().item() / attn_prefix[oracle_blocks_tensor].sum().item()
                            )
                            reuse_recall = float(torch.isin(reuse_blocks_tensor, oracle_blocks_tensor).float().mean().item())
                            reuse_jaccard = jaccard_similarity(reuse_blocks, oracle_blocks_budget)
                            selection_changed = (reuse_jaccard < 0.8) or (reuse_attention_recovery < 0.9)

                            summary_manager.update_changed_context(
                                sample.dataset_name,
                                layer_idx,
                                horizon,
                                budget_ratio,
                                selection_changed,
                            )

                            reuse_row = {
                                "sample_id": sample.sample_id,
                                "dataset_name": sample.dataset_name,
                                "step_t": source.step_idx,
                                "layer": layer_idx,
                                "head": head,
                                "query_type": "na",
                                "method": "reuse_selection",
                                "horizon_L": horizon,
                                "block_size": args.block_size,
                                "budget_blocks": "",
                                "budget_ratio": budget_ratio,
                                "budget_tokens": budget_tokens,
                                "attention_recovery": reuse_attention_recovery,
                                "token_recall": reuse_recall,
                                "query_cosine": float("nan"),
                                "jaccard_reuse_vs_oracle": reuse_jaccard,
                                "selection_changed": int(selection_changed),
                            }
                            metrics_writer.write(reuse_row)
                            summary_manager.update_metric(reuse_row)
                            if jsonl_handle is not None:
                                jsonl_handle.write(json.dumps(reuse_row, ensure_ascii=False) + "\n")

                            for query_type in query_types:
                                q_history = history_by_type[query_type]
                                base_q_t = q_history[-1]
                                predicted_base_queries = {
                                    "persistence_query": base_q_t,
                                    "linear_drift": base_q_t + horizon * (q_history[-1] - q_history[-2]),
                                    "ema_drift": base_q_t + horizon * ema_delta(q_history),
                                }
                                model_key = (query_type, horizon)
                                if model_key in tcn_models:
                                    hist = torch.stack(q_history, dim=0).unsqueeze(0).to(device)
                                    pred_delta = tcn_models[model_key](hist).squeeze(0)
                                    predicted_base_queries["tiny_tcn"] = base_q_t.to(device) + pred_delta

                                for method, pred_base_query in predicted_base_queries.items():
                                    pred_query = to_scoring_query(
                                        query_type,
                                        pred_base_query.to(device),
                                        current.position_id,
                                        model,
                                        device,
                                    )
                                    scores = torch.matmul(keys, pred_query.to(dtype=torch.float32))
                                    pred_blocks_tensor = topk_token_indices(scores, budget_tokens)
                                    pred_mass = float(attn_prefix[pred_blocks_tensor].sum().item())
                                    oracle_mass = float(attn_prefix[oracle_blocks_tensor].sum().item())
                                    recall = float(torch.isin(pred_blocks_tensor, oracle_blocks_tensor).float().mean().item())

                                    actual_query = current.per_layer[layer_idx]["post_query"][head_local]
                                    query_cos = cosine_similarity(pred_query, actual_query)
                                    row = {
                                        "sample_id": sample.sample_id,
                                        "dataset_name": sample.dataset_name,
                                        "step_t": source.step_idx,
                                        "layer": layer_idx,
                                        "head": head,
                                        "query_type": query_type,
                                        "method": method,
                                        "horizon_L": horizon,
                                        "block_size": args.block_size,
                                        "budget_blocks": "",
                                        "budget_ratio": budget_ratio,
                                        "budget_tokens": budget_tokens,
                                        "attention_recovery": pred_mass / oracle_mass if oracle_mass > 0 else 0.0,
                                        "token_recall": recall,
                                        "query_cosine": query_cos,
                                        "jaccard_reuse_vs_oracle": reuse_jaccard,
                                        "selection_changed": int(selection_changed),
                                    }
                                    metrics_writer.write(row)
                                    summary_manager.update_metric(row)
                                    if jsonl_handle is not None:
                                        jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            next_token = sampled_token
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break


def print_summary_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("No summary rows produced.")
        return
    columns = [
        "query_type",
        "method",
        "horizon_L",
        "budget_ratio",
        "attention_recovery",
        "token_recall",
        "query_cosine",
        "jaccard_reuse_vs_oracle",
        "changed_step_ratio",
    ]
    table = df[columns].copy()
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    args = parse_args()
    query_types = [part.strip() for part in args.query_types.split(",") if part.strip()]
    horizons = parse_int_list(args.horizons)
    budget_ratios = resolve_budget_ratios(args)
    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()] or None
    dataset_allocation = parse_dataset_allocation(args.dataset_allocation)
    data_root = args.data_root or get_default_data_root(args.dataset_family)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation="eager",
        dtype=get_dtype(args.dtype),
        low_cpu_mem_usage=True,
    )
    model = model.to(args.device).eval()
    model.config.output_attentions = True
    selection = make_selection_spec(model, args.layers, args.head_stride)
    runtime = AttentionCaptureRuntime(selection)
    patch_qwen3_attention(model, runtime)
    model._query_forecast_runtime = runtime

    if args.dataset_family == "reasoning" and dataset_allocation:
        if sum(dataset_allocation.values()) != args.num_samples:
            raise ValueError("Sum of --dataset-allocation counts must equal --num-samples.")
        samples = sample_reasoning_data_allocated(data_root, dataset_allocation, args.sample_seed)
    else:
        samples = sample_experiment_data(
            dataset_family=args.dataset_family,
            data_root=data_root,
            num_samples=args.num_samples,
            seed=args.sample_seed,
            datasets=datasets,
        )

    (args.results_dir / "run_config.json").write_text(
        json.dumps(
            {
                "dataset_family": args.dataset_family,
                "data_root": str(data_root),
                "num_samples": args.num_samples,
                "sample_seed": args.sample_seed,
                "datasets": datasets,
                "dataset_allocation": dataset_allocation,
                "history": args.history,
                "horizons": horizons,
                "block_size": args.block_size,
                "budget_blocks": parse_int_list(args.budget_blocks),
                "budget_ratios": budget_ratios,
                "query_types": query_types,
                "max_context_tokens": args.max_context_tokens,
                "num_decode_steps": args.num_decode_steps,
                "dtype": args.dtype,
                "device": args.device,
                "layers": args.layers,
                "selected_layers": selection.layers,
                "head_stride": args.head_stride,
                "selected_heads": selection.heads,
                "train_tcn": args.train_tcn,
                "tcn_epochs": args.tcn_epochs,
                "tcn_max_examples": args.tcn_max_examples,
                "sample_ids": [sample.sample_id for sample in samples],
                "sample_datasets": [sample.dataset_name for sample in samples],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    tcn_models: dict[tuple[str, int], TinyTCNPredictor] = {}
    if args.train_tcn:
        buffers = run_collection_pass(args, model, tokenizer, samples, selection, query_types, horizons)
        training_config = TCNTrainingConfig(
            epochs=args.tcn_epochs,
            max_examples=args.tcn_max_examples,
            device=args.device,
        )
        for key, buffer in buffers.items():
            if len(buffer) < max(32, args.history * 4):
                continue
            histories, targets = buffer.tensors()
            predictor = TinyTCNPredictor(head_dim=model.config.head_dim)
            tcn_models[key] = train_tcn_model(predictor, histories, targets, training_config)

    metrics_path = args.results_dir / "query_forecast_metrics.csv"
    jsonl_handle = (args.results_dir / "query_forecast_metrics.jsonl").open("w", encoding="utf-8") if args.summary_jsonl else None
    summary_manager = SummaryManager()
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "dataset_name",
                "step_t",
                "layer",
                "head",
                "query_type",
                "method",
                "horizon_L",
                "block_size",
                "budget_blocks",
                "budget_ratio",
                "budget_tokens",
                "attention_recovery",
                "token_recall",
                "query_cosine",
                "jaccard_reuse_vs_oracle",
                "selection_changed",
            ],
        )
        writer.writeheader()
        metrics_writer = BufferedMetricsWriter(writer, handle)
        for sample in samples:
            evaluate_sample(
                args,
                model,
                tokenizer,
                sample,
                selection,
                query_types,
                horizons,
                tcn_models,
                metrics_writer,
                summary_manager,
                jsonl_handle,
            )
        metrics_writer.flush()
    if jsonl_handle is not None:
        jsonl_handle.close()

    summary_manager.write(args.results_dir)
    summary_all = pd.read_csv(args.results_dir / "summary_all_steps.csv")
    print_summary_table(summary_all)


if __name__ == "__main__":
    main()
