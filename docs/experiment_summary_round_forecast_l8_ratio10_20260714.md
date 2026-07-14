# Round-Ahead Query Forecast Experiment Summary

## Setup

- Commit: `d56d855` (`[experiment] Align query forecasting with self-speculative decoding`)
- Model: `D:\preExperiments\model\Qwen3-4B`
- Data: `reasoning`
- Sample allocation: `gsm8k:8, math500:8, aime2024:4`
- Split: `train=10`, `validation=5`, `test=5`
- Horizon: `L=8`
- History: `H=16`
- Sampling: non-overlapping horizon boundaries
- Selection unit: token
- Budget: `budget_ratio=0.1`
- Layers: `0, 6, 12, 18, 24, 30, 35`
- Heads: `0, 8, 16, 24`
- Learned predictors: `temporal_linear`, `tiny_tcn`
- TCN channels: `32`
- Predictor epochs: `4`

This run follows the updated self-speculative proxy: predictors are trained on train prompts, selected on validation prompts, and all reported selection metrics are computed on held-out test prompts.

## Main Result: Macro by Sample, All Steps

`summary_macro_by_sample.csv` is the preferred comparison table because it avoids prompts with more valid rounds dominating the average.

| query_type | method | attention_recovery | retained_attention_mass | oracle_attention_mass | token_recall | query_cosine | changed_step_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| na | reuse_selection | 0.8068 | 0.4510 | 0.5486 | 0.4626 |  | 0.9852 |
| pre_rope_corrected | persistence_query | 0.8315 | 0.4545 | 0.5486 | 0.5182 | 0.7551 | 0.9852 |
| pre_rope_corrected | previous_round_endpoint_mean | 0.8479 | 0.4618 | 0.5486 | 0.5495 | 0.7971 | 0.9852 |
| pre_rope_corrected | temporal_linear | 0.8371 | 0.4571 | 0.5486 | 0.5290 | 0.7673 | 0.9852 |
| pre_rope_corrected | tiny_tcn | 0.8304 | 0.4542 | 0.5486 | 0.5164 | 0.7544 | 0.9852 |
| pre_rope_corrected | ema_drift | 0.5693 | 0.3224 | 0.5486 | 0.2883 | 0.3973 | 0.9852 |
| pre_rope_corrected | linear_drift | 0.3173 | 0.1756 | 0.5486 | 0.1836 | 0.1910 | 0.9852 |
| post_rope | persistence_query | 0.8072 | 0.4511 | 0.5486 | 0.4631 | 0.6757 | 0.9852 |
| post_rope | previous_round_endpoint_mean | 0.8190 | 0.4558 | 0.5486 | 0.4826 | 0.7201 | 0.9852 |
| post_rope | temporal_linear | 0.8126 | 0.4530 | 0.5486 | 0.4725 | 0.6883 | 0.9852 |
| post_rope | tiny_tcn | 0.8071 | 0.4511 | 0.5486 | 0.4625 | 0.6781 | 0.9852 |
| post_rope | ema_drift | 0.5169 | 0.3101 | 0.5486 | 0.2601 | 0.3526 | 0.9852 |
| post_rope | linear_drift | 0.2900 | 0.1674 | 0.5486 | 0.1804 | 0.1846 | 0.9852 |

## Changed Steps Only: Macro by Sample

| query_type | method | attention_recovery | token_recall | query_cosine |
| --- | --- | ---: | ---: | ---: |
| na | reuse_selection | 0.8040 | 0.4557 |  |
| pre_rope_corrected | persistence_query | 0.8292 | 0.5125 | 0.7526 |
| pre_rope_corrected | previous_round_endpoint_mean | 0.8459 | 0.5444 | 0.7951 |
| pre_rope_corrected | temporal_linear | 0.8349 | 0.5233 | 0.7649 |
| pre_rope_corrected | tiny_tcn | 0.8281 | 0.5108 | 0.7519 |
| post_rope | persistence_query | 0.8044 | 0.4562 | 0.6724 |
| post_rope | previous_round_endpoint_mean | 0.8167 | 0.4768 | 0.7174 |
| post_rope | temporal_linear | 0.8100 | 0.4659 | 0.6851 |
| post_rope | tiny_tcn | 0.8044 | 0.4558 | 0.6748 |
| pre_rope_corrected | ema_drift | 0.5663 | 0.2835 | 0.3932 |
| pre_rope_corrected | linear_drift | 0.3145 | 0.1809 | 0.1879 |
| post_rope | ema_drift | 0.5132 | 0.2549 | 0.3481 |
| post_rope | linear_drift | 0.2867 | 0.1775 | 0.1816 |

## Changed-Step Ratio by Dataset

| dataset_name | num_steps | num_changed_steps | changed_ratio |
| --- | ---: | ---: | ---: |
| aime2024 | 812 | 795 | 0.9791 |
| gsm8k | 1624 | 1602 | 0.9865 |
| math500 | 1624 | 1603 | 0.9871 |

## Predictor Cost Profile

| method | query_type | parameters | MACs per head forecast | MACs across selected heads/layers |
| --- | --- | ---: | ---: | ---: |
| temporal_linear | pre_rope_corrected | 17 | 2048 | 57344 |
| tiny_tcn | pre_rope_corrected | 16992 | 200704 | 5619712 |
| temporal_linear | post_rope | 17 | 2048 | 57344 |
| tiny_tcn | post_rope | 16992 | 200704 | 5619712 |

## Key Comparisons

All values below use macro-by-sample attention recovery on changed steps.

| comparison | delta |
| --- | ---: |
| pre_rope_corrected previous_round_endpoint_mean - reuse_selection | +0.0418 |
| pre_rope_corrected temporal_linear - reuse_selection | +0.0308 |
| pre_rope_corrected tiny_tcn - reuse_selection | +0.0240 |
| pre_rope_corrected temporal_linear - persistence_query | +0.0057 |
| pre_rope_corrected tiny_tcn - persistence_query | -0.0011 |
| post_rope previous_round_endpoint_mean - reuse_selection | +0.0127 |
| post_rope temporal_linear - reuse_selection | +0.0059 |
| post_rope tiny_tcn - reuse_selection | +0.0003 |

## Interpretation

- The updated self-speculative framing changes the conclusion relative to the earlier rolling-window diagnostic. With prompt-level train/validation/test split, `tiny_tcn` no longer shows a clear held-out advantage.
- The strongest policy is `pre_rope_corrected + previous_round_endpoint_mean`, with changed-step attention recovery `0.8459`. This conventional endpoint-mean heuristic beats both learned predictors in this 20-sample run.
- `pre_rope_corrected + temporal_linear` is the best learned predictor, reaching `0.8349` on changed steps. It is above `pre_rope_corrected persistence_query` by `+0.0057`, but still below endpoint mean by about `-0.0110`.
- `pre_rope_corrected` remains consistently stronger than `post_rope` for selection quality in this setting.
- `linear_drift` and `ema_drift` remain poor choices for this task.
- Because only five prompts are in the held-out test split, this should still be treated as a pre-experiment. The next statistically useful run should increase prompt count and repeat over multiple `sample_seed` values.

## Files

- `query_forecast_metrics.csv`: held-out test row-level metrics
- `summary_macro_by_sample.csv`: preferred method comparison
- `summary_by_sample.csv`: per-sample diagnostics
- `summary_all_steps.csv`: row-weighted micro summary
- `summary_changed_steps.csv`: row-weighted changed-step summary
- `predictor_profiles.json`: learned predictor parameter and MAC estimates
- `run_config.json`: exact split and run configuration

