import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ExperimentSample:
    sample_id: str
    dataset_name: str
    source_family: str
    task_type: str
    language: str
    context: str
    prompt_input: str
    answers: list[str]
    raw_length: int


def iter_longbench_jsonl(data_root: Path, datasets: list[str] | None = None) -> Iterable[ExperimentSample]:
    jsonl_root = data_root / "data" / "data"
    if not jsonl_root.exists():
        raise FileNotFoundError(f"LongBench jsonl folder not found: {jsonl_root}")

    names = datasets or sorted(path.stem for path in jsonl_root.glob("*.jsonl"))
    for dataset_name in names:
        file_path = jsonl_root / f"{dataset_name}.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        with file_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                item = json.loads(line)
                yield ExperimentSample(
                    sample_id=item.get("_id") or f"{dataset_name}-{idx}",
                    dataset_name=item["dataset"],
                    source_family="longbench",
                    task_type="long_context",
                    language=item["language"],
                    context=item["context"],
                    prompt_input=item["input"],
                    answers=item.get("answers", []),
                    raw_length=int(item.get("length", 0)),
                )


def iter_reasoning_jsonl(data_root: Path, datasets: list[str] | None = None) -> Iterable[ExperimentSample]:
    names = datasets or sorted(path.name for path in data_root.iterdir() if path.is_dir())
    for dataset_name in names:
        file_path = data_root / dataset_name / "samples.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(f"Reasoning dataset file not found: {file_path}")
        with file_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                item = json.loads(line)
                yield ExperimentSample(
                    sample_id=item.get("sample_id") or f"{dataset_name}-{idx}",
                    dataset_name=item.get("dataset_name", dataset_name),
                    source_family="reasoning",
                    task_type=item.get("task_type", "reasoning"),
                    language=item.get("language", "en"),
                    context=item.get("context", ""),
                    prompt_input=item["prompt_input"],
                    answers=item.get("answers", []),
                    raw_length=int(item.get("raw_length", len(item.get("prompt_input", "")))),
                )


def reservoir_sample(samples: Iterable[ExperimentSample], num_samples: int, seed: int) -> list[ExperimentSample]:
    rng = random.Random(seed)
    reservoir: list[ExperimentSample] = []
    for seen, sample in enumerate(samples, start=1):
        if len(reservoir) < num_samples:
            reservoir.append(sample)
            continue
        replace_at = rng.randint(1, seen)
        if replace_at <= num_samples:
            reservoir[replace_at - 1] = sample
    reservoir.sort(key=lambda item: (item.dataset_name, item.sample_id))
    return reservoir


def sample_longbench(
    data_root: Path,
    num_samples: int,
    seed: int,
    datasets: list[str] | None = None,
) -> list[ExperimentSample]:
    return reservoir_sample(iter_longbench_jsonl(data_root, datasets=datasets), num_samples, seed)


def sample_reasoning_data(
    data_root: Path,
    num_samples: int,
    seed: int,
    datasets: list[str] | None = None,
) -> list[ExperimentSample]:
    return reservoir_sample(iter_reasoning_jsonl(data_root, datasets=datasets), num_samples, seed)


def sample_reasoning_data_allocated(
    data_root: Path,
    allocation: dict[str, int],
    seed: int,
) -> list[ExperimentSample]:
    samples: list[ExperimentSample] = []
    for offset, (dataset_name, count) in enumerate(allocation.items()):
        subset = sample_reasoning_data(
            data_root=data_root,
            num_samples=count,
            seed=seed + offset,
            datasets=[dataset_name],
        )
        samples.extend(subset)
    samples.sort(key=lambda item: (item.dataset_name, item.sample_id))
    return samples


def sample_experiment_data(
    dataset_family: str,
    data_root: Path,
    num_samples: int,
    seed: int,
    datasets: list[str] | None = None,
) -> list[ExperimentSample]:
    if dataset_family == "longbench":
        return sample_longbench(data_root, num_samples, seed, datasets=datasets)
    if dataset_family == "reasoning":
        return sample_reasoning_data(data_root, num_samples, seed, datasets=datasets)
    raise ValueError(f"Unsupported dataset family: {dataset_family}")


def build_prompt(sample: ExperimentSample) -> str:
    if sample.source_family == "reasoning":
        if sample.task_type == "code_generation":
            return (
                "Please reason about the algorithm, explain the approach, then write the final code.\n\n"
                f"[Problem]\n{sample.prompt_input}\n\n"
                "[Reasoning and Code]\n"
            )
        return (
            "Please solve the problem step by step. Write a detailed reasoning process before giving the final answer.\n\n"
            f"[Problem]\n{sample.prompt_input}\n\n"
            "[Detailed Reasoning]\n"
        )

    if sample.language.lower().startswith("zh"):
        return (
            "请基于给定上下文完成任务，只输出必要答案。\n\n"
            f"[上下文]\n{sample.context}\n\n"
            f"[任务]\n{sample.prompt_input}\n\n"
            "[答案]\n"
        )
    return (
        "Answer the task using the provided long context. Keep the response concise.\n\n"
        f"[Context]\n{sample.context}\n\n"
        f"[Task]\n{sample.prompt_input}\n\n"
        "[Answer]\n"
    )
