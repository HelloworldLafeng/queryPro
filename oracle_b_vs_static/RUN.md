# Oracle B vs Static 10% KV Selection

This folder is an independent, correctness-first end-to-end self-speculative
decoding pre-experiment. It compares only:

1. one static `post-RoPE endpoint mean` page selection reused for a whole round;
2. Oracle B, which uses the actual current post-RoPE query to rescore all
   available keys before every sparse drafting decision.

The code does **not** train a query predictor and does not report Oracle B as a
speedup. Oracle full-key page scoring is outside the primary comparison.

## Exact semantics

- Model: configured `Qwen3-4B` path.
- Default data: configured `D:\preExperiments\LongBench` path. The loader accepts
  either standard LongBench `<dataset>.jsonl` files or normalized
  `<dataset>/samples.jsonl` files. It never downloads data.
- Default datasets: five long-context, short-answer LongBench QA tasks.
- Greedy decoding, batch size 1, `gamma=8`, at most 64 new tokens.
- Page size: 16 tokens, matching the existing page-level experiment.
- Page score: maximum post-RoPE QK score within each page.
- Budget: `B=max(1, ceil(0.1 * round_start_pages))` pages per layer and query
  head. B stays fixed during the round, and both methods use exactly B pages.
- GQA: each query head is scored only against the KV head assigned to its GQA
  group. No unrelated Q/K head pairing is allowed.
- Forced pages: none, matching the existing page-level proxy. There is no
  additional sink/recent window hidden outside the 10% page budget.
- Static initialization: final prompt token's post-RoPE query.
- Later static rounds: mean of the previous dense verification pass's first
  draft-token query and the final committed verifier/bonus-token query (or the
  terminal accepted draft query when decoding stops without a bonus).
- Static pages are chosen on the first drafting decision and never changed
  during that round. If a new physical page appears during the eight tokens, it
  is not silently added to the static set.
- Oracle B recomputes B page IDs for every layer/head/draft decision using that
  decision's real query and all causally available keys.

Autoregressive indexing deserves care: a causal LM query at position `i`
produces the logits for token `i+1`. Therefore `draft_position=1` logs the real
query that produces the first proposed token. The code never uses a future
token to choose the KV pages that generate that same token.

## Self-speculative round

For each method and round, the script:

1. clones the clean committed dense cache and crops the clone to immediately
   before the round's final committed token;
2. processes the final committed token and subsequent proposed tokens with
   sparse attention to draft at most eight tokens;
3. runs one full-attention verification forward on the same state plus drafts;
4. accepts the matching greedy prefix;
5. commits the verifier correction at the first mismatch, or the verifier bonus
   token when every draft matches;
6. crops the verification cache to the accepted prefix, appends the verifier
   token densely, and discards all temporary caches before the next round.

Both methods must commit exactly the same dense-greedy token sequence. The run
aborts if they differ, which catches verifier alignment and cache-contamination
errors.

## Before a formal run

The script performs an automatic short dense-equivalence check between the
unpatched model and the patched dense path. It requires the final top-1 token to
match and maximum logit error to stay below `--dense-check-atol`.

Start with the requested five-sample, 16-token correctness run:

```powershell
python oracle_b_vs_static\run_experiment.py `
  --model-path D:\preExperiments\model\Qwen3-4B `
  --data-root D:\preExperiments\LongBench `
  --num-samples 5 `
  --dataset-allocation qasper:1,multifieldqa_en:1,hotpotqa:1,2wikimqa:1,musique:1 `
  --min-input-tokens 1024 `
  --max-context-tokens 8192 `
  --max-new-tokens 16 `
  --draft-length 8 `
  --page-size 16 `
  --budget-ratio 0.1 `
  --results-dir oracle_b_vs_static\results\smoke_5 `
  --device cuda `
  --dtype bfloat16
```

Inspect before proceeding:

- `experiment_config.json`: dense patch check passed, page/budget semantics are
  correct, and selected prompts are the expected ones;
- per-round `budget_pages` equals `ceil(0.1 * round_start_pages)`;
- both methods generated the same dense token sequence (the script enforces it);
- all acceptance lengths are in `[0, draft_length]`;
- position IDs/RoPE are indirectly checked by dense-logit equivalence;
- input lengths and page counts are large enough to distinguish dynamic pages.

## Formal pre-experiment

The defaults select the longest eligible prompts deterministically from each
dataset. This intentionally avoids the roughly 14-page inputs of the earlier
reasoning experiment.

```powershell
python oracle_b_vs_static\run_experiment.py `
  --model-path D:\preExperiments\model\Qwen3-4B `
  --data-root D:\preExperiments\LongBench `
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
  --results-dir oracle_b_vs_static\results\formal_50 `
  --device cuda `
  --dtype bfloat16
```

If the server uses a different absolute mount point, change only
`--model-path`/`--data-root`; do not download or substitute a dataset. If fewer
than 50 valid long prompts exist, use at least 30 and make the allocation sum to
`--num-samples`.

The implementation deliberately clones/crops Hugging Face cache objects to
prioritize isolation and correctness. Expect it to be slower than an optimized
decoder and do not interpret runtime as a speed result.

## Outputs

Each result directory contains:

```text
oracle_b_vs_static_summary.json
oracle_b_vs_static_per_sample.csv
oracle_b_vs_static_per_round.csv
acceptance_length_distribution.csv
position_acceptance_rate.csv
experiment_config.json
experiment_summary.md
```

The per-round CSV includes position-wise static-to-Oracle recall, adjacent and
first-to-last Oracle selection overlap, and attention recovery. The JSON also
groups overlap and attention recovery by accepted length so they can be related
to rejection behavior.

## Reading the result

- A clear mean-acceptance improvement, especially at later conditional
  positions, supports temporal per-token KV routing.
- A small improvement means the stale selection is not the dominant error at a
  10% budget.
- Better attention recovery without better acceptance means attention mass is
  not a sufficient routing objective.
- Poor Oracle B acceptance means true-query Top-10% page selection itself is
  insufficient. Only then add a separate 10/15/20/30% budget scan; it is not
  part of this script's default experiment.
