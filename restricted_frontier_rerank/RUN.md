# Experiment 2: Restricted Actual-Query Frontier Reranking

## Question

Can the real query of each draft token rerank only a small `1%`–`5%` candidate frontier and approach a full query-aware scan over every KV page?

This is a page-level selection experiment. It is independent of `query_forecast` and writes results only under this directory's `results/` folder. Low-level Qwen3 capture/data utilities are shared with the sibling pre-experiment, not with the original package.

## Compared Selections

At round start `t`, the old KV cache is divided into pages of `page_size` tokens. With a page budget `B`:

1. **Prior/base selection:** choose the top-B pages from a previous-round ranking.
2. **Candidate frontier:** append the next `1%`, `2%`, or `5%` of all pages from the same ranking.
3. **Restricted actual-query selection:** for each real `q_(t+i)`, compute exact per-page maximum QK score only inside `base + frontier`, then keep B pages.
4. **Full query-aware selection:** compute the same per-page maximum QK score over every old page, then keep B pages.
5. **Oracle attention selection:** keep the B pages with the largest dense future attention mass. This is an offline upper bound, not an implementable selector.

Using the same actual-query page score for restricted and full selection isolates the effect of limiting the candidate pool. Comparing both with the dense-attention oracle separately shows whether page-max QK scoring itself is adequate.

The three prior rankings are the same as Experiment 1:

- previous terminal attention;
- post-RoPE previous-round endpoint mean;
- pre-RoPE endpoint mean rotated to the future terminal position.

## What “Near Full” Means

Query-aware sparse-attention papers commonly report recall of dense/oracle critical pages or tokens. Quest, for example, measures recall against full-attention critical tokens. Recent work also reports attention capture within roughly `1.5%` of oracle. Therefore this experiment uses two explicit criteria:

- `near_full_query_mass`: restricted attention mass is at least `98.5%` of the full query-aware selection's mass;
- `near_full_query_joint`: the mass criterion holds and page recall against full query-aware selection is at least `90%`.

References:

- [Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2406.10774)
- [Measure Once, Mask Once: Delta Refined Block Sparse Attention](https://openreview.net/forum?id=5HzrYMUlRd)

These are pre-experiment thresholds, not universal quality guarantees. Ultimately an integrated self-speculative decoder must report accepted length and end-to-end latency.

## Recommended Run

From the repository root:

```powershell
python restricted_frontier_rerank\run_experiment.py `
  --model-path D:\preExperiments\model\Qwen3-4B `
  --data-root D:\preExperiments\ReasoningData `
  --num-samples 20 `
  --dataset-allocation gsm8k:8,math500:8,aime2024:4 `
  --horizon 8 `
  --page-size 16 `
  --budget-ratios 0.1 `
  --frontier-page-ratios 0.01,0.02,0.05 `
  --near-mass-threshold 0.985 `
  --near-page-recall-threshold 0.9 `
  --max-context-tokens 2048 `
  --num-decode-steps 256 `
  --layers representative `
  --head-stride 8 `
  --device cuda `
  --dtype bfloat16
```

Use `--num-samples 2 --dataset-allocation gsm8k:2` first. This experiment emits a row for every future draft offset, layer, head, prior ranking, and frontier ratio, so its CSV is substantially larger than Experiment 1.

## Primary Metrics

- `candidate_recall_of_full_query_pages`: whether the base+frontier pool contains pages the full query-aware method wants;
- `restricted_page_recall_vs_full_query`: selection overlap after actual-query reranking;
- `attention_recovery_vs_full_query`: future attention mass retained relative to full query-aware selection;
- `attention_recovery_vs_oracle`: retained mass relative to dense-attention oracle pages;
- `full_query_recovery_vs_oracle`: separates candidate-pool failure from page-scoring failure;
- `candidate_oracle_upper_bound_recovery`: best possible oracle recovery if reranking inside the candidate pool were perfect;
- `near_full_query_mass_rate` and `near_full_query_joint_rate`: fraction of cases meeting the stated thresholds.

## Outputs

Results are written under `restricted_frontier_rerank/results/`:

- `run_config.json`;
- `rerank_metrics.csv`;
- `summary_overall.csv`;
- `summary_by_draft_offset.csv`;
- `summary_by_dataset.csv`.

## Interpretation

- High candidate recall but low restricted recovery means max-QK page scoring inside the pool is inadequate.
- Low candidate recall and a low candidate oracle upper bound mean the prior ranking does not produce a useful local frontier.
- If a `<=5%` frontier has high mass recovery but page recall is slightly lower, the missed pages may be selection-boundary pages with negligible mass; inspect both metrics rather than recall alone.
- Results should also remain stable at later `draft_offset` values. Strong performance only at offsets 1–2 does not support an eight-token draft round.
