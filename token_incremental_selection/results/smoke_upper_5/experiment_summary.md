# Token-level Incremental KV Selection

## Acceptance and update upper bounds

| Method | Mean accepted | Full-8 rate | Mean replacements | Candidate recall |
| --- | ---: | ---: | ---: | ---: |
| oracle_incremental_r0.01 | 6.6500 | 0.3333 | 8.46 | 1.0000 |
| oracle_incremental_r0.02 | 6.6500 | 0.3333 | 15.98 | 1.0000 |
| oracle_incremental_r0.05 | 6.6500 | 0.3333 | 38.58 | 1.0000 |
| oracle_incremental_r0.1 | 6.0833 | 0.3077 | 74.86 | 1.0000 |
| oracle_incremental_r0.2 | 5.9133 | 0.2857 | 124.47 | 1.0000 |

## Conditional acceptance

- `oracle_incremental_r0.01`: P1=1.0000, P2=1.0000, P3=0.8333, P4=0.9000, P5=1.0000, P6=1.0000, P7=1.0000, P8=1.0000
- `oracle_incremental_r0.02`: P1=1.0000, P2=1.0000, P3=0.8333, P4=0.9000, P5=1.0000, P6=1.0000, P7=1.0000, P8=1.0000
- `oracle_incremental_r0.05`: P1=1.0000, P2=1.0000, P3=0.8333, P4=0.9000, P5=1.0000, P6=1.0000, P7=1.0000, P8=1.0000
- `oracle_incremental_r0.1`: P1=1.0000, P2=1.0000, P3=0.8333, P4=0.9000, P5=0.8889, P6=1.0000, P7=1.0000, P8=1.0000
- `oracle_incremental_r0.2`: P1=0.9286, P2=0.9231, P3=0.9091, P4=0.9000, P5=0.8889, P6=1.0000, P7=1.0000, P8=1.0000

## Gate for the predictor stage

Reference methods are not rerun. Compare these accepted lengths with the previously recorded token-level Static and Oracle-B results before deciding whether to proceed to Token Entrant Predictor training.
