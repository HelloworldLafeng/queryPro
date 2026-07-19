# Token-level Incremental KV Selection: Stage-1 Oracle Gate

This independent folder implements the mandatory first stage of the Token
Entrant Predictor program: determine whether replacing only a few historical KV
tokens per draft position can theoretically recover a meaningful fraction of
Oracle B's acceptance gain.

It deliberately does **not** train the MLP yet. Predictor data collection and
training should begin only after this upper-bound gate is positive.

## What is compared

The three token-level reference methods below describe the previously recorded
baselines. They are not rerun by this revision; each suite executes only the
incremental methods needed for its gate.

- `static_token_10pct`: endpoint-mean selects one token set for the round.
- `best_static_token_oracle`: future dense attention chooses the best fixed set.
- `oracle_b_token`: the current real sparse query scans all historical keys and
  selects a fresh token set at every draft decision.
- `oracle_incremental_r*`: at most `m=ceil(r*K)` tokens enter/leave per step;
  both entrants and evictions use the next position's real sparse-query Oracle-B
  target, obtained after that Query is formed but before its attention.
- `candidate_oracle_f*_r*`: the same incremental oracle, but entrants must lie
  in a causally constructed candidate pool of size `f*K`.
- `random`, `score_verify`, `score_current`, and `score_hybrid`: non-learned
  causal entrant baselines for the later predictor comparison.

The primary metric is actual greedy self-speculative accepted length. Selection
overlap and attention recovery are diagnostics only.

For causal candidate methods, the cost proxy counts the full-history
current-Query/Key scoring currently used to construct the pool. The first
version favors correctness over an optimized candidate builder; Oracle target
scans used only to define an upper bound are reported separately from deployable
cost. A later predictor is useful only if this causal scan can also be reduced.

Because this stage intentionally changes the selection unit from 16-token pages
to individual tokens and aggregates heads into one set per layer, its new
Static/Best-Static/Oracle-B baselines must be used for all gap calculations.
Do not mix the earlier page-level values `4.77/5.44/6.54` directly into the
token-level recovery denominator.

## Token and budget semantics

- Selection unit is an individual historical token, never a page or block.
- Each layer has one token-position set shared across query heads.
- QK scores first use correct GQA head mapping, then average over query heads.
- `K=ceil(0.1*T_history)` is fixed for the eight-token round.
- The token currently being processed is not yet part of historical KV Cache;
  it is always visible to itself and is not charged to the historical-cache K.
- Once that token has been processed, it becomes an eligible historical entrant
  for the next draft position.
- No sink, recent-window, page, block, or physical KV-buffer mechanism is added.

This experiment simulates sparse access using an attention mask over the intact
dense cache. It is an accuracy experiment, not a kernel or runtime benchmark.

## Oracle Incremental update

For the set carried from the previous position, the current position's real
sparse-query Oracle-B set, and update limit `m`:

1. compute true entrants `S*_(j+1) - S_j`;
2. take at most m entrants with the highest next-query QK score;
3. evict the same number of tokens from `S_j - S*_(j+1)`, lowest next-query
   score first;
4. require exactly K unique selected tokens after every update.

This uses unavailable full-history scoring at the current position and is
reported only as an upper bound. It does not substitute a dense Query for the
real sparse Query.

## Causal candidate pool

The candidate pool never uses the next query. It combines:

- high/boundary endpoint-verification scores;
- high current-query scores;
- recent historical positions;
- neighbors of currently selected positions;
- deterministic random negatives when necessary.

At the next position, Candidate Oracle may use the real-query Oracle label only
to choose among this already causal pool. Its entrant recall measures whether
the candidate construction is adequate before training a predictor.

## Correctness smoke test

Start with five samples and 16 generated tokens:

```powershell
python token_incremental_selection\run_experiment.py `
  --suite upper_bound `
  --model-path D:\preExperiments\model\Qwen3-4B `
  --data-root D:\preExperiments\LongBench\data `
  --num-samples 5 `
  --dataset-allocation qasper:1,multifieldqa_en:1,hotpotqa:1,2wikimqa:1,musique:1 `
  --max-context-tokens 8192 `
  --max-new-tokens 16 `
  --draft-length 8 `
  --budget-ratio 0.1 `
  --update-ratios 0.01,0.02,0.05,0.1,0.2 `
  --results-dir token_incremental_selection\results\smoke_upper_5 `
  --device cuda `
  --dtype bfloat16
```

Before a formal run, check:

- patched dense logits exactly preserve top-1 and satisfy the configured error
  tolerance;
- every method commits the same dense-greedy output tokens;
- every incremental set contains exactly K unique causal historical positions;
- actual replacements never exceed m;
- every reported method is one of the five requested Oracle Incremental ratios;
- Oracle Incremental alone consumes future/full-scan information.

## Formal Stage-1 upper bound

```powershell
python token_incremental_selection\run_experiment.py `
  --suite upper_bound `
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
  --budget-ratio 0.1 `
  --update-ratios 0.01,0.02,0.05,0.1,0.2 `
  --results-dir token_incremental_selection\results\formal_upper_50 `
  --device cuda `
  --dtype bfloat16
```

The Oracle Incremental sweep skips the dense future probe because its update
target comes directly from each real sparse Query and the fixed-set probe is
not used by this suite. Absolute update budgets can be tested separately with,
for example:

```text
--update-ratios "" --absolute-update-counts 1,2,4,8,16,32
```

## Candidate-pool upper bound

After the upper-bound sweep, run the candidate gate at `m=0.05K`:

```powershell
python token_incremental_selection\run_experiment.py `
  --suite candidate `
  --model-path D:\preExperiments\model\Qwen3-4B `
  --data-root D:\preExperiments\LongBench\data `
  --num-samples 50 `
  --dataset-allocation qasper:10,multifieldqa_en:10,hotpotqa:10,2wikimqa:10,musique:10 `
  --candidate-factors 0.5,1,2,4 `
  --heuristic-update-ratio 0.05 `
  --max-context-tokens 8192 `
  --max-new-tokens 64 `
  --results-dir token_incremental_selection\results\formal_candidate_50 `
  --device cuda `
  --dtype bfloat16
```

## Causal non-learned baselines

```powershell
python token_incremental_selection\run_experiment.py `
  --suite heuristic `
  --model-path D:\preExperiments\model\Qwen3-4B `
  --data-root D:\preExperiments\LongBench\data `
  --num-samples 50 `
  --dataset-allocation qasper:10,multifieldqa_en:10,hotpotqa:10,2wikimqa:10,musique:10 `
  --heuristic-candidate-factor 2 `
  --heuristic-update-ratio 0.05 `
  --max-context-tokens 8192 `
  --max-new-tokens 64 `
  --results-dir token_incremental_selection\results\formal_heuristic_50 `
  --device cuda `
  --dtype bfloat16
```

## Outputs

```text
results/<run>/
  experiment_config.json
  token_incremental_per_round.csv
  token_incremental_per_sample.csv
  token_incremental_summary.csv
  token_incremental_summary.json
  position_acceptance_rate.csv
  acceptance_length_distribution.csv
  experiment_summary.md
  plots/
```

The summary reports mean accepted length, full-8 and zero-accept rates, actual
replacement count, and candidate recall. Static-to-Oracle-B gap recovery is
left blank because those reference methods are not rerun. The per-round file additionally
contains position-wise selection recall, attention recovery, entrant precision,
and update diagnostics. Heuristic candidate/entrant labels are resolved one
step later against the next real sparse Query, not against the dense probe
trajectory.

When `matplotlib` is installed, the script also writes the requested
conditional-acceptance curve, update-amount curve, entrant-recall scatter, and
selection-overlap scatter. Otherwise it records `plots_skipped.txt` without
failing the experiment.

## Gate for Stage 2/3

Proceed to the MLP Token Entrant Predictor only if:

1. a small Oracle Incremental budget, preferably `m<=0.05K`, recovers a
   substantial fraction of Oracle B's acceptance gain;
2. a causal pool no larger than `2K` covers most true entrants and its
   candidate-limited Oracle preserves much of the incremental upper bound;
3. late draft positions, especially 7 and 8, improve materially;
4. random perturbation does not explain the gain.

If this gate passes, the next folder revision should add prompt-level 60/20/20
splits, offline feature datasets, the `input_dim -> 16 -> 1` weighted-BCE MLP,
validation selection by complete decoding accepted length, and held-out test
evaluation. No next-query feature may enter that predictor.
