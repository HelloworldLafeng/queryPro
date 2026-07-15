# Future-Query Union and Frontier Concentration Summary

## Setup

- Experiment: `future_query_union_frontier`
- Model: `D:\preExperiments\model\Qwen3-4B`
- Data: `D:\preExperiments\ReasoningData`
- Samples: `20` (`gsm8k:8`, `math500:8`, `aime2024:4`)
- Horizon: `8`
- Budget ratio: `0.1`
- Frontier ratios: `0.01`, `0.02`, `0.05`, `0.1`
- Max context tokens: `2048`
- Decode steps: `256`
- Layers: `0, 6, 12, 18, 24, 30, 35`
- Heads: `0, 8, 16, 24`
- Union rows: `16,800`
- Frontier rows: `201,600`

Note: `summary_frontier.csv` reports `num_samples=21` because pandas reads mixed numeric/string `sample_id` values during summarization. Reading `sample_id` as string confirms exactly `20` unique dataset/sample pairs, matching `run_config.json`.

## Union Expansion

| horizon | budget_ratio | union_over_terminal_budget | union_fraction_of_prefix | mean_terminal_set_recall_across_queries | terminal_selection_attention_recovery_across_round |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.1 | 2.6836 | 0.2749 | 0.5613 | 0.8407 |

The union of oracle top-K tokens across the next 8 real queries is about `2.68x` the terminal-query budget. A single terminal query top-K is therefore not enough to cover all future per-token oracle sets. However, terminal selection still retains about `84.1%` of round-level oracle attention mass, so many missed union tokens are lower-mass boundary tokens.

## Frontier Concentration

| ranker | frontier_ratio | prior_union_recall | frontier_miss_recall | frontier_missed_attention_share_recovered | prior_attention_recovery_across_round |
| --- | ---: | ---: | ---: | ---: | ---: |
| post_rope_endpoint_mean | 0.01 | 0.3538 | 0.0631 | 0.0922 | 0.8426 |
| post_rope_endpoint_mean | 0.02 | 0.3538 | 0.1090 | 0.1551 | 0.8426 |
| post_rope_endpoint_mean | 0.05 | 0.3538 | 0.2287 | 0.3059 | 0.8426 |
| post_rope_endpoint_mean | 0.10 | 0.3538 | 0.3815 | 0.4762 | 0.8426 |
| pre_rope_future_corrected_endpoint_mean | 0.01 | 0.3565 | 0.0638 | 0.0904 | 0.8368 |
| pre_rope_future_corrected_endpoint_mean | 0.02 | 0.3565 | 0.1104 | 0.1523 | 0.8368 |
| pre_rope_future_corrected_endpoint_mean | 0.05 | 0.3565 | 0.2313 | 0.3031 | 0.8368 |
| pre_rope_future_corrected_endpoint_mean | 0.10 | 0.3565 | 0.3860 | 0.4775 | 0.8368 |
| previous_terminal_attention | 0.01 | 0.3533 | 0.0619 | 0.0880 | 0.8410 |
| previous_terminal_attention | 0.02 | 0.3533 | 0.1065 | 0.1477 | 0.8410 |
| previous_terminal_attention | 0.05 | 0.3533 | 0.2226 | 0.2913 | 0.8410 |
| previous_terminal_attention | 0.10 | 0.3533 | 0.3708 | 0.4577 | 0.8410 |

## Dataset Breakdown

### Union Metrics

| dataset | union_over_terminal_budget | mean_terminal_set_recall_across_queries | terminal_selection_attention_recovery_across_round |
| --- | ---: | ---: | ---: |
| aime2024 | 2.6625 | 0.5615 | 0.8339 |
| gsm8k | 2.7155 | 0.5543 | 0.8370 |
| math500 | 2.6622 | 0.5682 | 0.8478 |

### Best Frontier Ratio per Ranker

Using `frontier_ratio=0.10`, which is the strongest tested frontier:

| dataset | ranker | frontier_miss_recall | frontier_missed_attention_share_recovered |
| --- | --- | ---: | ---: |
| aime2024 | post_rope_endpoint_mean | 0.3812 | 0.4723 |
| aime2024 | pre_rope_future_corrected_endpoint_mean | 0.3857 | 0.4729 |
| aime2024 | previous_terminal_attention | 0.3746 | 0.4562 |
| gsm8k | post_rope_endpoint_mean | 0.3790 | 0.4752 |
| gsm8k | pre_rope_future_corrected_endpoint_mean | 0.3833 | 0.4766 |
| gsm8k | previous_terminal_attention | 0.3630 | 0.4521 |
| math500 | post_rope_endpoint_mean | 0.3841 | 0.4791 |
| math500 | pre_rope_future_corrected_endpoint_mean | 0.3887 | 0.4807 |
| math500 | previous_terminal_attention | 0.3767 | 0.4640 |

## Frontier Size

Actual old-prefix lengths range from `50` to `444` tokens, with mean `218.15`. Mean frontier token counts are:

| frontier_ratio | mean frontier_tokens |
| ---: | ---: |
| 0.01 | 2.6867 |
| 0.02 | 4.8533 |
| 0.05 | 11.3800 |
| 0.10 | 22.2700 |

Unlike the page-level restricted rerank experiment, these token-level frontier ratios produce meaningfully different frontier sizes.

## Interpretation

- The future-query oracle union is large: about `2.68x` a single terminal-query top-K budget. This argues against relying on one representative future query if exact future per-token oracle coverage is the goal.
- Misses are not strongly concentrated immediately below the stale ranking cutoff. At `frontier_ratio=0.05`, the best missed attention recovery is only about `30.6%`; at `0.10`, it is still only about `47.7%`.
- The RUN.md preliminary useful signal was `frontier_missed_attention_share_recovered >= 0.8` at frontier `<=0.05`. This run is far below that threshold across all datasets and rankers.
- The three stale rankers are very close. `pre_rope_future_corrected_endpoint_mean` has the best union recall and slightly best 10% frontier missed-mass recovery, while `post_rope_endpoint_mean` has the best prior attention recovery across the round.
- The practical implication is negative for a small local frontier strategy: a stable-core plus `1%` to `5%` residual frontier is unlikely to recover enough of the future-important misses under this setup.

## Files

- `union_metrics.csv`: row-level future-query union metrics
- `frontier_metrics.csv`: row-level stale-ranking frontier concentration metrics
- `summary_union.csv`: main union summary
- `summary_frontier.csv`: main frontier summary
- `summary_union_by_dataset.csv`: dataset-level union diagnostics
- `summary_frontier_by_dataset.csv`: dataset-level frontier diagnostics
- `run_config.json`: exact run configuration
- `run_stderr.log`: checkpoint loading output and one pandas dtype warning

