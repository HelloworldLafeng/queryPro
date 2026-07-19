# Token-level Incremental KV Selection

## Acceptance and update upper bounds

| Method | Mean accepted | Full-8 rate | Mean replacements | Candidate recall |
| --- | ---: | ---: | ---: | ---: |
| oracle_incremental_r0.01 | 6.7071 | 0.7005 | 8.86 | 1.0000 |
| oracle_incremental_r0.02 | 6.6529 | 0.6792 | 16.74 | 1.0000 |
| oracle_incremental_r0.05 | 6.7090 | 0.6887 | 41.07 | 1.0000 |
| oracle_incremental_r0.1 | 6.7089 | 0.6863 | 78.94 | 1.0000 |
| oracle_incremental_r0.2 | 6.7629 | 0.7055 | 123.96 | 1.0000 |

## Conditional acceptance

- `oracle_incremental_r0.01`: P1=0.9788, P2=0.9826, P3=0.9619, P4=0.9729, P5=0.9608, P6=0.9644, P7=0.9472, P8=0.9867
- `oracle_incremental_r0.02`: P1=0.9813, P2=0.9707, P3=0.9646, P4=0.9731, P5=0.9528, P6=0.9583, P7=0.9465, P8=0.9764
- `oracle_incremental_r0.05`: P1=0.9811, P2=0.9730, P3=0.9619, P4=0.9758, P5=0.9638, P6=0.9675, P7=0.9377, P8=0.9832
- `oracle_incremental_r0.1`: P1=0.9788, P2=0.9827, P3=0.9671, P4=0.9757, P5=0.9639, P6=0.9618, P7=0.9346, P8=0.9798
- `oracle_incremental_r0.2`: P1=0.9762, P2=0.9798, P3=0.9716, P4=0.9837, P5=0.9721, P6=0.9708, P7=0.9301, P8=0.9802

## Gate for the predictor stage

Reference methods are not rerun. Compare these accepted lengths with the previously recorded token-level Static and Oracle-B results before deciding whether to proceed to Token Entrant Predictor training.
