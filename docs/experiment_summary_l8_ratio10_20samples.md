# Query Forecast Experiment Summary

## Setup

- Model: `Qwen3-4B`
- Data family: `reasoning`
- Sample allocation: `gsm8k:8, math500:8, aime2024:4`
- Horizon: `L=8`
- History: `H=16`
- Budget mode: token-level retrieval with `budget_ratio=0.1`
- Query types: `pre_rope_corrected`, `post_rope`
- Representative layers: `0, 6, 12, 18, 24, 30, 35`
- Head stride / selected heads: `8` / `0, 8, 16, 24`
- TCN: enabled, `epochs=4`, `max_examples=12000`

## Main Table: All Steps

| query_type | method | attention_recovery | token_recall | query_cosine | changed_step_ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| na | reuse_selection | 0.8089 | 0.4610 |  | 0.9870 |
| pre_rope_corrected | persistence_query | 0.8348 | 0.5176 | 0.7560 | 0.9870 |
| pre_rope_corrected | linear_drift | 0.3025 | 0.1785 | 0.1861 | 0.9870 |
| pre_rope_corrected | ema_drift | 0.5578 | 0.2792 | 0.3930 | 0.9870 |
| pre_rope_corrected | tiny_tcn | 0.8417 | 0.5324 | 0.7779 | 0.9870 |
| post_rope | persistence_query | 0.8091 | 0.4614 | 0.6764 | 0.9870 |
| post_rope | linear_drift | 0.2783 | 0.1782 | 0.1803 | 0.9870 |
| post_rope | ema_drift | 0.5054 | 0.2546 | 0.3496 | 0.9870 |
| post_rope | tiny_tcn | 0.8320 | 0.5098 | 0.7586 | 0.9870 |

## Main Table: Changed Steps Only

| query_type | method | attention_recovery | token_recall | query_cosine |
| --- | --- | ---: | ---: | ---: |
| na | reuse_selection | 0.8065 | 0.4550 |  |
| pre_rope_corrected | persistence_query | 0.8328 | 0.5126 | 0.7538 |
| pre_rope_corrected | linear_drift | 0.2999 | 0.1760 | 0.1835 |
| pre_rope_corrected | ema_drift | 0.5548 | 0.2748 | 0.3894 |
| pre_rope_corrected | tiny_tcn | 0.8399 | 0.5278 | 0.7759 |
| post_rope | persistence_query | 0.8068 | 0.4554 | 0.6734 |
| post_rope | linear_drift | 0.2751 | 0.1756 | 0.1778 |
| post_rope | ema_drift | 0.5016 | 0.2497 | 0.3456 |
| post_rope | tiny_tcn | 0.8301 | 0.5050 | 0.7565 |

## Key Findings

- With `10%` token budget, the experiment is no longer degenerate. Changed-step ratio is around `98.6%` to `98.9%` across datasets.
- On changed steps, `pre_rope_corrected + tiny_tcn` reaches `0.8399` attention recovery, which is `+0.0334` over `reuse_selection` and `+0.0071` over `pre_rope_corrected persistence_query`.
- On changed steps, `post_rope + tiny_tcn` reaches `0.8301` attention recovery, which is `+0.0236` over `reuse_selection` and `+0.0233` over `post_rope persistence_query`.
- `linear_drift` performs poorly, and `ema_drift` remains substantially below persistence / TCN.
- In this setting, `pre_rope_corrected` is stronger than `post_rope` for both persistence and TCN.

