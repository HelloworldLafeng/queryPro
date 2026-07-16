# Best Static Oracle Pre-experiment

## Question

Under the same page-level 10% KV budget and `gamma=8`, compare:

1. **Static endpoint mean**: previous verification's first-draft and final
   verifier/bonus post-RoPE query mean chooses one fixed page set;
2. **Best Static Oracle**: future dense queries are known offline, and one fixed
   page set maximizes their summed attention coverage;
3. **Oracle B**: every sparse drafting decision uses its current real query to
   choose a new page set.

This separates two explanations:

- the current endpoint heuristic chooses a poor shared set; or
- no single 10% set can serve the whole eight-query horizon.

## Exact Best Static Oracle definition

At a round start with `N` available pages and
`B=max(1, ceil(0.1*N))`, run the dense model causally for up to eight decision
queries. For layer `l`, query head `h`, page `p`, and future position `j`, let

```text
A[l,h,j,p] = dense attention mass assigned to page p
```

The Best Static score is

```text
score[l,h,p] = sum_j A[l,h,j,p]
```

and the fixed set is the B pages with the largest score. This is the exact
maximizer of total dense-attention coverage for a fixed-size shared set; it is
not a union heuristic and does not train a predictor.

Only page IDs already present at round start are eligible. If the final page is
partial, later tokens that fill that already-selected physical page are
naturally visible. A newly created page cannot be added to a static set.

Autoregressive indexing follows the actual decoder: the query at position `i`
produces logits for token `i+1`. Thus the eight oracle decision queries are the
round's final committed token query followed by the first seven dense future
token queries. No future token is used to select pages for its own creation.

Best Static selection uses dense attention mass because the experiment asks for
the strongest possible fixed attention-coverage set. Endpoint static and Oracle
B retain the previous experiment's page score: maximum post-RoPE QK within a
page. All methods execute real sparse drafting after selection.

## Shared settings and correctness

- Qwen3-4B from the configured local path; no download.
- Existing LongBench path and five long-context short-answer QA datasets.
- Greedy decoding, batch size 1, page size 16, 10% pages, `gamma=8`.
- No uncounted sink/recent pages, matching the previous Oracle-B experiment.
- Correct GQA mapping: each query head uses its assigned KV head.
- Same dense verifier, samples, generation length, precision, and seed.
- Independent cloned/cropped cache state for probe, draft, and verification.
- All three methods must commit exactly the same dense-greedy output; otherwise
  the run aborts.
- A patched-vs-original dense-logit check runs before the experiment.

The dense future probe and Oracle scoring costs are excluded. This experiment
measures the acceptance-quality upper bounds, not speed.

## Five-sample correctness run

Run this first:

```powershell
python best_static_oracle\run_experiment.py `
  --model-path D:\preExperiments\model\Qwen3-4B `
  --data-root D:\preExperiments\LongBench\data `
  --num-samples 5 `
  --dataset-allocation qasper:1,multifieldqa_en:1,hotpotqa:1,2wikimqa:1,musique:1 `
  --min-input-tokens 1024 `
  --max-context-tokens 8192 `
  --max-new-tokens 16 `
  --draft-length 8 `
  --page-size 16 `
  --budget-ratio 0.1 `
  --results-dir best_static_oracle\results\smoke_5 `
  --device cuda `
  --dtype bfloat16
```

Check that:

- the dense patch check passes;
- every method commits the same token sequence;
- `budget_pages=ceil(0.1*round_start_pages)`;
- each Best Static round has `probe_query_count` between 1 and 8;
- Best Static probe coverage is never below endpoint probe coverage, apart from
  negligible floating-point tolerance;
- acceptance lengths remain between 0 and the actual draft length.

## Formal 50-sample run

```powershell
python best_static_oracle\run_experiment.py `
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
  --results-dir best_static_oracle\results\formal_50 `
  --device cuda `
  --dtype bfloat16
```

This run adds an eight-step dense oracle probe to every Best Static round, so it
will be slower than the preceding two-method experiment.

## Outputs

```text
results/<run>/
  best_static_oracle_summary.json
  best_static_oracle_per_sample.csv
  best_static_oracle_per_round.csv
  acceptance_length_distribution.csv
  position_acceptance_rate.csv
  experiment_config.json
  experiment_summary.md
```

Primary acceptance metrics are mean accepted length, full-8 rate, zero-accept
rate, conditional acceptance by position, and per-sample paired differences.

The per-position `fixed_reference_recall_of_query_pages` diagnostic uses the
active fixed set for endpoint/Best Static rows. Oracle-B rows retain endpoint
static as the reference because Oracle B itself has no fixed set.

The key derived metric is:

```text
(BestStatic - EndpointStatic) / (OracleB - EndpointStatic)
```

It reports the fraction of Oracle B's acceptance gain closed by the best
possible shared set.

The dense probe also reports endpoint and Best Static coverage relative to an
even stronger per-query **attention-mass** Top-B oracle. This diagnostic is
separate from Oracle B, whose operational selector remains real-query max-QK.

## Decision rule

- If Best Static is close to Oracle B in mean accepted length, full-8 rate, and
  later-position conditional acceptance, per-token refresh is not necessary;
  future work should predict the shared KV importance of the next eight queries.
- If Best Static improves substantially over endpoint mean but retains a clear,
  consistent gap to Oracle B, part of the problem is the shared-set constraint;
  refresh or incremental updates are necessary.
- If Best Static barely improves over endpoint mean, the existing endpoint
  heuristic is already near the best shared set and the Oracle-B gain is almost
  entirely query-specific.
