# Round-Ahead Query Forecasting for Self-Speculative Decoding

This repository contains a PyTorch / Hugging Face pre-experiment for a sparse-KV self-speculative decoding policy.

## Follow-up Self-Speculative Experiments

The original query-forecast experiment is kept at the repository root. Later,
independent pre-experiments live in their own folders so their selection units
and evaluation denominators are not mixed:

- `future_query_union_frontier/`: future-query oracle union and missed frontier;
- `restricted_frontier_rerank/`: restricted-candidate reranking;
- `oracle_b_vs_static/`: page-level Static 10% versus per-token Oracle B;
- `best_static_oracle/`: page-level endpoint mean, Best Static Oracle, and Oracle B;
- `token_incremental_selection/`: token-level incremental replacement. Its
  current revision implements the Stage-1 Oracle/candidate upper-bound gate;
  run that gate before adding or training the lightweight entrant MLP.

Each folder has an independent run guide and `results/` directory. In
particular, the token-level experiment recomputes its own Static, Best Static,
and Oracle B baselines; the earlier page-level acceptance values are not reused
in its Oracle-gain recovery denominator.

## Experimental Question

Self-speculative decoding can use one KV cache in two modes:

- **draft:** attend to a sparse subset of the existing KV cache;
- **verify:** attend to the complete KV cache and verify the draft tokens.

The full-attention verify pass of round `n-1` exposes the queries for that round. A conventional stale policy can use the previous round's first-draft and terminal/bonus queries to choose the sparse KV subset for round `n`. This repository tests whether temporal query correlation supports a better policy:

1. collect a causal history ending at the terminal query `q_t` of round `n-1`;
2. predict the terminal query `q_(t+L)` of round `n`, where `L` is the expected draft horizon;
3. score only the KV tokens already available at `t` with the predicted query;
4. compare the selected token set with the set preferred by the real future query from the full verify pass.

This is **token-level sparse KV selection**, not KV block retrieval. The experiment does not maintain two KV caches and does not yet run an end-to-end speculative decoder; it evaluates the selection-policy proxy before integrating it into a decoding system.

## Round and Causality Semantics

For a horizon such as `L=8`, a training/evaluation example is:

```text
available at selection time: q_(t-H+1) ... q_t and K_<=t
prediction target:           q_(t+8)
sparse draft selection:      top-K(predicted q_(t+8) @ K_<=t)
oracle selection:            top-K(actual q_(t+8) @ K_<=t)
```

By default examples are taken only at non-overlapping `L`-token round boundaries. `--rolling-windows` enables the older every-token diagnostic mode.

Prompts are split by whole sample, stratified by dataset, **before** query windows are collected. The defaults are 50% train, 25% validation, and 25% test. Learned predictors train only on train prompts, select their best epoch on validation prompts, and all reported selection metrics use only test prompts. This prevents deterministic greedy trajectories from appearing in both training and evaluation.

## Query Representations

- `pre_rope_corrected`: predict the query after Q/K normalization but before RoPE, then apply the future position's RoPE before KV scoring.
- `post_rope`: predict the already rotated query directly.

## Compared Policies

- `reuse_selection`: reuse the top-K attention selection of the previous terminal query. This is the stalest baseline.
- `previous_round_endpoint_mean`: average the previous round's first and terminal/bonus queries, then re-score the KV cache. This models the conventional self-speculative heuristic.
- `persistence_query`: use the previous terminal query directly.
- `linear_drift` and `ema_drift`: parameter-free temporal extrapolation.
- `temporal_linear`: a learned dimension-shared temporal filter. It has only `H+1` parameters and approximately `H * head_dim` MACs per head forecast.
- `tiny_tcn`: a causal residual TCN. Its default hidden width is 32 instead of `head_dim`; change it with `--tcn-channels`.

`predictor_profiles.json` records parameter counts and approximate MACs per head and across all selected layer/head pairs. Forecast inference is performed once per round and reused across all evaluated KV budgets.

## Main Run

Example reasoning run:

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
  --results-dir results_round_forecast_l8_ratio10 `
  --device cuda `
  --dtype bfloat16 `
  --train-predictors temporal_linear,tiny_tcn `
  --tcn-channels 32 `
  --predictor-epochs 4 `
  --max-training-examples 12000
```

`--train-tcn` remains as a compatibility alias that adds `tiny_tcn`, but new runs should use `--train-predictors`.

For a stronger result, increase the number of prompts and use multiple sample seeds. Twenty prompts leave only about five held-out test prompts under the default split and should be treated as a smoke/pre-experiment rather than a final statistical result.

## Metrics

- `query_cosine`: cosine similarity between the forecast and actual future post-RoPE query.
- `token_recall`: overlap with the future query's oracle top-K token set. Predicted and oracle sets have equal size.
- `retained_attention_mass`: actual future attention mass retained by the selected old KV tokens.
- `oracle_attention_mass`: future attention mass retained by the best top-K old KV tokens.
- `attention_recovery`: `retained_attention_mass / oracle_attention_mass`; this is oracle-normalized and is not a fraction of total attention mass.

The main row-weighted summaries are retained for diagnostics. `summary_by_sample.csv` and `summary_macro_by_sample.csv` should be preferred when comparing methods because they avoid allowing prompts with more valid rounds to dominate the result.

## Data Preparation

Reasoning datasets can be downloaded and normalized with:

```powershell
python download_reasoning_data.py
```

The default local paths are:

```text
D:\preExperiments\model\Qwen3-4B
D:\preExperiments\ReasoningData
D:\preExperiments\LongBench
```

## Outputs

- `run_config.json`: model/data settings and exact train/validation/test sample IDs;
- `predictor_profiles.json`: learned predictor parameter and MAC estimates;
- `query_forecast_metrics.csv`: test-only row-level metrics;
- `summary_all_steps.csv` and `summary_changed_steps.csv`: micro summaries;
- `summary_by_sample.csv` and `summary_macro_by_sample.csv`: sample-level summaries;
- dataset, query-type, and layer diagnostic summaries.

Generated result directories and model weights are ignored by git.

## Important Integration Check

The Qwen3 attention module is patched to capture normalized pre-RoPE queries, post-RoPE queries, keys, and optional attention weights. Hugging Face attention internals change between Transformers releases. Before a formal server run, pin the exact working package versions and compare patched versus unpatched logits on a short prompt. The capture adapter accepts either `cache_position` or `position_ids`, but a successful forward pass alone is not sufficient evidence that the patch is numerically equivalent.
