from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset, load_dataset


ROOT = Path(r"D:\preExperiments\ReasoningData")


def ensure_root() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_gsm8k(dataset: Dataset) -> list[dict]:
    rows = []
    for idx, item in enumerate(dataset):
        rows.append(
            {
                "sample_id": f"gsm8k-{idx}",
                "dataset_name": "gsm8k",
                "task_type": "math_reasoning",
                "language": "en",
                "prompt_input": item["question"],
                "answers": [item["answer"]],
                "raw_length": len(item["question"]),
            }
        )
    return rows


def normalize_math500(dataset: Dataset) -> list[dict]:
    rows = []
    for idx, item in enumerate(dataset):
        rows.append(
            {
                "sample_id": item.get("unique_id", f"math500-{idx}"),
                "dataset_name": "math500",
                "task_type": "math_reasoning",
                "language": "en",
                "prompt_input": item["problem"],
                "answers": [item.get("answer", ""), item.get("solution", "")],
                "raw_length": len(item["problem"]),
                "subject": item.get("subject", ""),
                "level": item.get("level", ""),
            }
        )
    return rows


def normalize_aime(dataset: Dataset) -> list[dict]:
    rows = []
    for idx, item in enumerate(dataset):
        rows.append(
            {
                "sample_id": str(item.get("id", f"aime2024-{idx}")),
                "dataset_name": "aime2024",
                "task_type": "math_reasoning",
                "language": "en",
                "prompt_input": item["problem"],
                "answers": [item.get("answer", ""), item.get("solution", "")],
                "raw_length": len(item["problem"]),
                "year": item.get("year", ""),
                "url": item.get("url", ""),
            }
        )
    return rows


def try_download_livecodebench() -> dict:
    dataset_dir = ROOT / "livecodebench"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    status = {"dataset_name": "livecodebench", "downloaded": False, "note": ""}
    try:
        ds = load_dataset("livecodebench/code_generation_lite", split="test")
        rows = []
        for idx, item in enumerate(ds):
            prompt = item.get("question_content") or item.get("prompt") or item.get("question") or json.dumps(item, ensure_ascii=False)
            rows.append(
                {
                    "sample_id": str(item.get("question_id", f"livecodebench-{idx}")),
                    "dataset_name": "livecodebench",
                    "task_type": "code_generation",
                    "language": "en",
                    "prompt_input": prompt,
                    "answers": [item.get("canonical_solution", "")],
                    "raw_length": len(prompt),
                }
            )
        write_jsonl(dataset_dir / "samples.jsonl", rows)
        status["downloaded"] = True
        status["num_rows"] = len(rows)
    except Exception as exc:
        status["note"] = repr(exc)
        (dataset_dir / "download_failed.txt").write_text(repr(exc), encoding="utf-8")
    return status


def main() -> None:
    ensure_root()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    manifest = []

    specs = [
        ("gsm8k", "openai/gsm8k", {"name": "main", "split": "test"}, normalize_gsm8k),
        ("math500", "HuggingFaceH4/MATH-500", {"split": "test"}, normalize_math500),
        ("aime2024", "HuggingFaceH4/aime_2024", {"split": "train"}, normalize_aime),
    ]

    for local_name, hf_name, kwargs, normalizer in specs:
        dataset_dir = ROOT / local_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(hf_name, **kwargs)
        rows = normalizer(ds)
        write_jsonl(dataset_dir / "samples.jsonl", rows)
        manifest.append(
            {
                "dataset_name": local_name,
                "hf_name": hf_name,
                "num_rows": len(rows),
                "path": str((dataset_dir / "samples.jsonl").resolve()),
            }
        )

    manifest.append(try_download_livecodebench())
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
