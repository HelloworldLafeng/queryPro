# Best Static Oracle Experiment

## Acceptance

| Method | Mean accepted | Full-8 rate | Zero rate | Rounds | Attention recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static endpoint mean | 4.7655 | 0.2771 | 0.0302 | 563 | 0.9183 |
| Best Static Oracle | 5.4385 | 0.3710 | 0.0238 | 504 | 0.9466 |
| Oracle B | 6.5419 | 0.6674 | 0.0233 | 430 | 0.9916 |

Best Static gain over endpoint static: `0.6730`.
Best Static gap to Oracle B: `1.1034`.
Fraction of Oracle-B acceptance gain closed by Best Static (round-micro): `37.88%`.
Fraction closed using paired sample-macro means: `35.80%`.

## Conditional acceptance

| Method | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 | Pos 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Static endpoint mean | 0.9698 | 0.8392 | 0.9157 | 0.8889 | 0.8620 | 0.8900 | 0.7354 | 0.8525 |
| Best Static Oracle | 0.9762 | 0.8574 | 0.9570 | 0.9494 | 0.9322 | 0.9226 | 0.7585 | 0.8500 |
| Oracle B | 0.9767 | 0.9634 | 0.9744 | 0.9731 | 0.9635 | 0.9676 | 0.9080 | 0.9795 |

## Dense future-query probe

- Endpoint coverage / per-query attention oracle: `0.9648`
- Best Static coverage / per-query attention oracle: `0.9896`

## Decision

If Best Static closes most of Oracle B's acceptance gain, a predicted shared future-query KV set is sufficient. If a substantial acceptance gap remains, query-specific refresh or incremental routing is necessary.
