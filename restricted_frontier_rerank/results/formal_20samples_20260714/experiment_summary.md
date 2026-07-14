# Restricted Actual-Query Frontier Reranking Summary

## Setup

- Experiment: `restricted_frontier_rerank`
- Model: `D:\preExperiments\model\Qwen3-4B`
- Data: `D:\preExperiments\ReasoningData`
- Samples: `20` (`gsm8k:8`, `math500:8`, `aime2024:4`)
- Horizon: `8`
- Page size: `16`
- Budget ratio: `0.1`
- Frontier page ratios: `0.01`, `0.02`, `0.05`
- Near-full mass threshold: `0.985`
- Near-full joint threshold: mass `>=0.985` and page recall `>=0.9`
- Layers: `0, 6, 12, 18, 24, 30, 35`
- Heads: `0, 8, 16, 24`
- Row-level metrics: `1,209,600`

Note: `summary_overall.csv` reports `num_samples=21` because pandas reads mixed numeric/string `sample_id` values during summarization. Reading `sample_id` as string confirms there are exactly `20` unique dataset/sample pairs, matching `run_config.json`.

## Main Results

| ranker | frontier_page_ratio | candidate_recall_of_full_query_pages | restricted_page_recall_vs_full_query | attention_recovery_vs_full_query | attention_recovery_vs_oracle | full_query_recovery_vs_oracle | candidate_oracle_upper_bound_recovery | near_full_query_mass_rate | near_full_query_joint_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| post_rope_endpoint_mean | 0.01 | 0.8121 | 0.8121 | 0.9413 | 0.9031 | 0.9630 | 0.9230 | 0.7777 | 0.6718 |
| post_rope_endpoint_mean | 0.02 | 0.8121 | 0.8121 | 0.9413 | 0.9031 | 0.9630 | 0.9230 | 0.7777 | 0.6718 |
| post_rope_endpoint_mean | 0.05 | 0.8171 | 0.8171 | 0.9431 | 0.9049 | 0.9630 | 0.9252 | 0.7857 | 0.6823 |
| pre_rope_future_corrected_endpoint_mean | 0.01 | 0.8054 | 0.8054 | 0.9311 | 0.8936 | 0.9630 | 0.9123 | 0.7621 | 0.6609 |
| pre_rope_future_corrected_endpoint_mean | 0.02 | 0.8054 | 0.8054 | 0.9311 | 0.8936 | 0.9630 | 0.9123 | 0.7621 | 0.6609 |
| pre_rope_future_corrected_endpoint_mean | 0.05 | 0.8104 | 0.8104 | 0.9332 | 0.8956 | 0.9630 | 0.9147 | 0.7704 | 0.6712 |
| previous_terminal_attention | 0.01 | 0.7748 | 0.7748 | 0.9438 | 0.9009 | 0.9630 | 0.9174 | 0.7620 | 0.6265 |
| previous_terminal_attention | 0.02 | 0.7748 | 0.7748 | 0.9438 | 0.9009 | 0.9630 | 0.9174 | 0.7620 | 0.6265 |
| previous_terminal_attention | 0.05 | 0.7806 | 0.7806 | 0.9458 | 0.9029 | 0.9630 | 0.9198 | 0.7699 | 0.6381 |

## Best Settings

| metric | best setting | value |
| --- | --- | ---: |
| `attention_recovery_vs_full_query` | `previous_terminal_attention`, frontier `0.05` | 0.9458 |
| `attention_recovery_vs_oracle` | `post_rope_endpoint_mean`, frontier `0.05` | 0.9049 |
| `candidate_recall_of_full_query_pages` | `post_rope_endpoint_mean`, frontier `0.05` | 0.8171 |
| `near_full_query_mass_rate` | `post_rope_endpoint_mean`, frontier `0.05` | 0.7857 |
| `near_full_query_joint_rate` | `post_rope_endpoint_mean`, frontier `0.05` | 0.6823 |
| `candidate_oracle_upper_bound_recovery` | `post_rope_endpoint_mean`, frontier `0.05` | 0.9252 |

## Draft Offset Trend

The `0.05` frontier is the only setting that differs materially from `0.01/0.02` in this run.

| ranker | offset 1 recovery vs full | offset 8 recovery vs full | offset 1 joint rate | offset 8 joint rate |
| --- | ---: | ---: | ---: | ---: |
| post_rope_endpoint_mean | 0.9518 | 0.9368 | 0.7576 | 0.6320 |
| previous_terminal_attention | 0.9707 | 0.9295 | 0.7419 | 0.5760 |
| pre_rope_future_corrected_endpoint_mean | 0.9201 | 0.9386 | 0.6876 | 0.6679 |

`previous_terminal_attention` has the highest recovery at draft offset 1, but degrades the most by offset 8. `post_rope_endpoint_mean` is less sharp at offset 1 but keeps stronger late-offset joint rates. `pre_rope_future_corrected_endpoint_mean` is lower early and improves slightly toward later offsets.

## Dataset Breakdown

For frontier `0.05`:

| ranker | dataset | candidate recall | recovery vs full | recovery vs oracle |
| --- | --- | ---: | ---: | ---: |
| post_rope_endpoint_mean | aime2024 | 0.8023 | 0.9363 | 0.9004 |
| post_rope_endpoint_mean | gsm8k | 0.8139 | 0.9427 | 0.9051 |
| post_rope_endpoint_mean | math500 | 0.8277 | 0.9471 | 0.9069 |
| previous_terminal_attention | aime2024 | 0.7794 | 0.9431 | 0.9039 |
| previous_terminal_attention | gsm8k | 0.7719 | 0.9420 | 0.9000 |
| previous_terminal_attention | math500 | 0.7900 | 0.9509 | 0.9052 |
| pre_rope_future_corrected_endpoint_mean | aime2024 | 0.7942 | 0.9244 | 0.8893 |
| pre_rope_future_corrected_endpoint_mean | gsm8k | 0.8068 | 0.9349 | 0.8979 |
| pre_rope_future_corrected_endpoint_mean | math500 | 0.8221 | 0.9360 | 0.8965 |

## Frontier Size Note

Actual `prefix_pages` range from `4` to `28`, with mean `14.15`. Because frontier pages are computed as `ceil(prefix_pages * frontier_page_ratio)`, both `0.01` and `0.02` always append exactly one page in this run. The `0.05` setting appends one page in most cases and two pages in a minority of rows.

Mean candidate pool sizes:

| frontier_page_ratio | mean frontier_pages | mean candidate_pool_pages | mean candidate_pool_ratio |
| ---: | ---: | ---: | ---: |
| 0.01 | 1.0000 | 2.8300 | 0.2144 |
| 0.02 | 1.0000 | 2.8300 | 0.2144 |
| 0.05 | 1.0967 | 2.9267 | 0.2187 |

This means the experiment is not really testing a fine-grained 1% vs 2% frontier at the current prompt lengths. It is testing a very small absolute frontier, usually one extra page beyond the 10% base page budget.

## Interpretation

- Restricted actual-query reranking does not approach the configured `near_full_query_joint` target consistently. The best joint rate is `0.6823`, well below a robust near-full selector.
- Candidate frontier coverage is the main limitation for page recall. The best candidate recall of full-query pages is only `0.8171`.
- Attention mass recovery is much higher than page recall. The best restricted recovery vs full-query is `0.9458`, suggesting many missed pages are lower-mass boundary pages, but this is still below the `0.985` near-full mass threshold in roughly 21% of cases.
- Full query-aware max-QK page selection itself recovers only `0.9630` of dense-attention oracle mass. This separates candidate-pool failure from a second limitation: page-max QK scoring is not perfectly aligned with dense attention oracle pages.
- The candidate oracle upper bound peaks at `0.9252` recovery vs oracle. Even perfect reranking inside the candidate pool would leave a gap to the dense-attention oracle under this prior/frontier construction.
- `post_rope_endpoint_mean` is the best prior for candidate recall, oracle recovery, and near-full rates. `previous_terminal_attention` wins only on recovery vs full-query, partly because full-query selection itself is not the dense-attention oracle.

## Files

- `rerank_metrics.csv`: row-level metrics
- `summary_overall.csv`: main aggregate summary
- `summary_by_draft_offset.csv`: draft-offset stability diagnostics
- `summary_by_dataset.csv`: dataset-level diagnostics
- `run_config.json`: exact run configuration
- `run_stderr.log`: checkpoint loading output and one pandas dtype warning

