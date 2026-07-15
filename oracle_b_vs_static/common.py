"""Input adapters for the configured ReasoningData and LongBench folders."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import json
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class InputSample:
    sample_id: str
    dataset_name: str
    task_type: str
    context: str
    prompt_input: str


def _dataset_file(root: Path, dataset_name: str) -> Path:
    candidates = (
        root / dataset_name / "samples.jsonl",
        root / f"{dataset_name}.jsonl",
        root / "data" / f"{dataset_name}.jsonl",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No JSONL file found for {dataset_name}; checked: {candidates}")


def iter_input_samples(root: Path, dataset_name: str) -> Iterable[InputSample]:
    """Read normalized project data or the standard LongBench JSONL layout."""
    path = _dataset_file(root, dataset_name)
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            item = json.loads(line)
            yield InputSample(
                sample_id=str(item.get("sample_id") or item.get("id") or f"{dataset_name}-{index}"),
                dataset_name=str(item.get("dataset_name") or item.get("dataset") or dataset_name),
                task_type=str(item.get("task_type", "long_context_qa")),
                context=str(item.get("context", "")),
                prompt_input=str(item.get("prompt_input") or item.get("input") or item.get("question") or ""),
            )


def build_prompt(sample: InputSample) -> str:
    if sample.context:
        return (
            "Answer the question using only the supplied context. Keep the final answer concise.\n\n"
            f"[Context]\n{sample.context}\n\n[Question]\n{sample.prompt_input}\n\n[Answer]\n"
        )
    if sample.task_type == "code_generation":
        return f"Please reason about the algorithm, then write the final code.\n\n[Problem]\n{sample.prompt_input}\n\n[Reasoning and Code]\n"
    return f"Please solve the problem step by step.\n\n[Problem]\n{sample.prompt_input}\n\n[Detailed Reasoning]\n"


def generation_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    return prompt


def encode_prompt(tokenizer, prompt: str, max_tokens: int) -> torch.Tensor:
    ids = tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"]
    if ids.shape[1] <= max_tokens:
        return ids
    # LongBench-style middle truncation preserves both the chat/instruction
    # prefix and the question/answer suffix instead of silently dropping one.
    head = min(256, max_tokens // 4)
    return torch.cat((ids[:, :head], ids[:, -(max_tokens - head) :]), dim=1)
