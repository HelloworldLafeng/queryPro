# Query Forecast KV Selection Experiments

This repository contains a PyTorch / HuggingFace Transformers pre-experiment for testing whether short-horizon query forecasting can predict future KV-cache token selection.

## Scope

- Captures Qwen3 attention internals with minimal attention-module patching.
- Compares `pre_rope_corrected` and `post_rope` query representations.
- Evaluates token-level KV selection with ratio budgets such as `--budget-ratios 0.1`.
- Implements baselines: reuse selection, persistence query, linear drift, EMA drift, and a tiny causal TCN predictor.
- Produces CSV summaries and an optional Markdown experiment report.

Model weights and datasets are intentionally not included. The default local paths used by the scripts are:

```text
D:\preExperiments\model\Qwen3-4B
D:\preExperiments\ReasoningData
D:\preExperiments\LongBench
```

## Main Run

Example 20-sample reasoning run:

```powershell
python run_query_forecast_experiment.py `
  --dataset-family reasoning `
  --num-samples 20 `
  --dataset-allocation gsm8k:8,math500:8,aime2024:4 `
  --history 16 `
  --horizons 8 `
  --budget-ratios 0.1 `
  --max-context-tokens 2048 `
  --num-decode-steps 256 `
  --layers representative `
  --head-stride 8 `
  --query-types pre_rope_corrected,post_rope `
  --results-dir results_formal_l8_reasoning_token_ratio10_20samples `
  --device cuda `
  --dtype bfloat16 `
  --train-tcn `
  --tcn-epochs 4 `
  --tcn-max-examples 12000
```

## Data Preparation

Reasoning datasets can be downloaded and normalized with:

```powershell
python download_reasoning_data.py
```

The script writes normalized JSONL files under `D:\preExperiments\ReasoningData`.

## Outputs

The experiment writes metrics and summaries into the selected `results-dir`, including:

- `query_forecast_metrics.csv`
- `summary_all_steps.csv`
- `summary_changed_steps.csv`
- `summary_by_query_type.csv`
- `summary_by_dataset.csv`
- `summary_by_layer.csv`
- `changed_step_ratio_by_dataset.csv`
- `changed_step_ratio_by_layer.csv`

Generated result directories are ignored by git because they may contain very large CSV files.

