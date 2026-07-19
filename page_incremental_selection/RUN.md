# Page-level Incremental KV Selection

## Purpose

This folder adds only the missing page-level Oracle Incremental sweep. It does
not rerun the three established reference methods:

- Static endpoint mean: mean accepted length `4.77`;
- Best Static Oracle: mean accepted length `5.44`;
- per-draft-token Oracle B: mean accepted length `6.54`.

Their complete result files are loaded from `best_static_oracle/results/` and
validated against the current sample IDs and experiment configuration before a
new run starts. The final report combines reused reference rows with the four
new methods and marks the source of every row.

## Matched page-level semantics

All settings intentionally match `oracle_b_vs_static/` and
`best_static_oracle/`:

- page size: 16 historical tokens;
- retained budget: `B=max(1, ceil(0.1*N_round_start_pages))`;
- granularity: one page set per layer and Query head;
- GQA: each Query head scores only its correctly repeated KV head;
- endpoint and Oracle page score: maximum post-RoPE QK score within the page;
- no extra sink, recent-window, or fixed pages;
- greedy decoding, batch size 1, `gamma=8`, full-KV verification;
- same model, datasets, prompts, truncation, precision, seed, and generation cap.

Every incremental round starts from exactly the Static endpoint-mean page set.
Position 1 uses that set unchanged. Before attention at positions 2 through 8,
the current real sparse Query defines the current Oracle-B page set. The method
then adds at most

```text
m = max(1, ceil(r * B))
```

highest-scoring missing Oracle pages and evicts the same number of selected
non-Oracle pages with the lowest current-query page score.

The four tested ratios are fixed to `r = 1%, 5%, 10%, 20%`. For a typical
8192-token prefix, `N=512`, `B=52`, and the maximum refresh is respectively
1, 3, 6, and 11 pages per layer/Query-head/step.

This is an Oracle quality upper bound: full page scoring and the Oracle update
decision are not counted as deployable speed. The actual sparse attention still
uses exactly B pages.

## Five-sample correctness run

Use the existing five-sample reference directory:

```powershell
python page_incremental_selection\run_experiment.py `
  --model-path D:\preExperiments\model\Qwen3-4B `
  --data-root D:\preExperiments\LongBench\data `
  --num-samples 5 `
  --dataset-allocation qasper:1,multifieldqa_en:1,hotpotqa:1,2wikimqa:1,musique:1 `
  --sample-policy longest `
  --sample-seed 7 `
  --min-input-tokens 1024 `
  --max-context-tokens 8192 `
  --max-new-tokens 16 `
  --draft-length 8 `
  --page-size 16 `
  --budget-ratio 0.1 `
  --reference-results-dir best_static_oracle\results\smoke_5 `
  --results-dir page_incremental_selection\results\smoke_5 `
  --device cuda `
  --dtype bfloat16
```

Before the formal run, confirm:

- the patched dense-logit check has zero or tolerated error and identical top-1;
- the reference configuration validation passes;
- all four methods commit the same dense-greedy output sequence;
- position 1 performs zero replacements;
- positions 2 to 8 never exceed `max_update_pages`;
- every selector retains exactly B unique, causal pages;
- `budget_pages` and update rounding match the formulas above.

## Formal 50-sample run

```powershell
python page_incremental_selection\run_experiment.py `
  --model-path D:\preExperiments\model\Qwen3-4B `
  --data-root D:\preExperiments\LongBench\data `
  --num-samples 50 `
  --dataset-allocation qasper:10,multifieldqa_en:10,hotpotqa:10,2wikimqa:10,musique:10 `
  --sample-policy longest `
  --sample-seed 7 `
  --min-input-tokens 1024 `
  --max-context-tokens 8192 `
  --max-new-tokens 64 `
  --draft-length 8 `
  --page-size 16 `
  --budget-ratio 0.1 `
  --reference-results-dir best_static_oracle\results\formal_50 `
  --results-dir page_incremental_selection\results\formal_50 `
  --device cuda `
  --dtype bfloat16
```

Only the four incremental methods are executed. This avoids repeating Static,
Best Static, and Oracle B while still producing paired comparisons against the
same 50 reference prompts.

## Outputs

```text
results/<run>/
  experiment_config.json
  page_incremental_summary.json
  page_incremental_summary.csv
  page_incremental_per_sample.csv
  page_incremental_per_round.csv
  position_acceptance_rate.csv
  acceptance_length_distribution.csv
  experiment_summary.md
  plots/
```

Primary metrics are mean accepted length, full-8 acceptance rate, late-position
conditional acceptance, and the fraction of the reference Static-to-Oracle-B
acceptance gap recovered. Each sample row contains its paired Static, Best
Static, and Oracle B values.

Also report the actual number of pages replaced, because overlap may already be
high enough that an update uses fewer than its maximum m. Selection recall and
attention recovery remain explanatory diagnostics rather than final goals.

## Interpretation

- If 1% already approaches Oracle B and larger ratios add little, page-level
  selection aging is driven by a very small set of important entrant pages.
- If performance rises steadily with the update ratio, the token-level result
  did not transfer cleanly to pages and the refresh budget must remain larger.
- If overlap/attention recovery rises but acceptance does not, max-QK Oracle
  membership is not sufficiently acceptance-aware; later work should predict
  logit-sensitive page utility rather than optimize overlap alone.
