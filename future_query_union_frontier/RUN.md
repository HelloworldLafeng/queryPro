# Experiment 1: Future-Query Union and Frontier Concentration

## Question

This standalone pre-experiment answers two questions for sparse-KV self-speculative decoding:

1. For the `L` real queries in the next draft round, how much larger is the union of their oracle top-K KV tokens than the terminal query's top-K set?
2. Among future-important tokens missed by a previous-round selection, how many lie immediately below its cutoff in a small `1%`–`10%` frontier?

The code is independent of `query_forecast`. It captures Qwen3 internals using the local `common.py`. All future-query sets are restricted to KV tokens that already exist at the beginning of the round; KV entries created by future draft tokens are excluded.

## Definitions

For round start `t`, horizon `L`, old KV prefix `K_<=t`, and token budget `B`:

```text
S_i = oracle top-B old KV tokens for actual q_(t+i),  i=1...L
U   = union(S_1, ..., S_L)
```

The main union metric is `union_over_terminal_budget = |U| / B`. A value of `1.0` means the terminal set already covers the union; `2.4` means covering every query's oracle set needs 2.4 times the terminal budget.

The experiment evaluates three stale/prior rankings:

- `previous_terminal_attention`: previous terminal query's dense attention;
- `post_rope_endpoint_mean`: mean of the previous round's first and terminal post-RoPE queries;
- `pre_rope_future_corrected_endpoint_mean`: mean in pre-RoPE space, rotated to the next terminal position.

For each ranking, the top-B tokens form the previous selection. The next `ceil(frontier_ratio * prefix_length)` tokens in that same ranking form the frontier. Two complementary concentration metrics are reported:

- `frontier_miss_recall`: fraction of missed future-union tokens found in the frontier;
- `frontier_missed_attention_share_recovered`: fraction of the missed tokens' cumulative future attention importance found in the frontier.

The second metric is usually more meaningful: missing many negligible oracle-boundary tokens may be harmless, while recovering a few high-mass misses may be sufficient.

## Recommended Run

From the repository root:

```powershell
python future_query_union_frontier\run_experiment.py `
  --model-path D:\preExperiments\model\Qwen3-4B `
  --data-root D:\preExperiments\ReasoningData `
  --num-samples 20 `
  --dataset-allocation gsm8k:8,math500:8,aime2024:4 `
  --horizon 8 `
  --budget-ratios 0.1 `
  --frontier-ratios 0.01,0.02,0.05,0.1 `
  --max-context-tokens 2048 `
  --num-decode-steps 256 `
  --layers representative `
  --head-stride 8 `
  --device cuda `
  --dtype bfloat16
```

The run performs dense attention only for the selected representative layers, but it is still a GPU experiment. Start with `--num-samples 2 --dataset-allocation gsm8k:2` as a smoke test. When changing `--num-samples`, the allocation must sum to it.

## Outputs

Results are written under `future_query_union_frontier/results/`:

- `run_config.json`: exact samples and configuration;
- `union_metrics.csv`: row-level union expansion metrics;
- `frontier_metrics.csv`: row-level frontier concentration metrics;
- `summary_union.csv`: mean union results by horizon and budget;
- `summary_frontier.csv`: mean results by ranker, budget, and frontier size;
- `summary_union_by_dataset.csv` and `summary_frontier_by_dataset.csv`: dataset diagnostics.

## Decision Rules

- If `union_over_terminal_budget` is close to `1`, a single representative query is adequate.
- If it is substantially above `1` but a `1%`–`5%` frontier recovers most missed attention mass, stable-core plus residual retrieval is promising.
- A useful preliminary signal is `frontier_missed_attention_share_recovered >= 0.8` at `frontier_ratio <= 0.05` across datasets/layers, not only in the global mean.
- If both miss recall and missed-mass recovery remain low, the misses are not local to the stale ranking boundary and a small-frontier method is unlikely to work.
