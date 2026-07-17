# Best Static Oracle Experiment

## Acceptance

| Method | Mean accepted | Full-8 rate | Zero rate | Rounds | Attention recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static endpoint mean | 3.5556 | 0.2222 | 0.1111 | 18 | 0.8895 |
| Best Static Oracle | 4.1875 | 0.2500 | 0.0625 | 16 | 0.9161 |
| Oracle B | 6.0000 | 0.3333 | 0.0000 | 12 | 0.9884 |

Best Static gain over endpoint static: `0.6319`.
Best Static gap to Oracle B: `1.8125`.
Fraction of Oracle-B acceptance gain closed by Best Static (round-micro): `25.85%`.
Fraction closed using paired sample-macro means: `18.44%`.

## Conditional acceptance

| Method | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 | Pos 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Static endpoint mean | 0.8889 | 0.6875 | 0.9000 | 1.0000 | 0.6667 | 0.8333 | 1.0000 | 1.0000 |
| Best Static Oracle | 0.9375 | 0.6667 | 1.0000 | 1.0000 | 0.8000 | 1.0000 | 0.8000 | 1.0000 |
| Oracle B | 1.0000 | 1.0000 | 0.8333 | 0.9000 | 1.0000 | 1.0000 | 0.8750 | 1.0000 |

## Dense future-query probe

- Endpoint coverage / per-query attention oracle: `0.9531`
- Best Static coverage / per-query attention oracle: `0.9854`

## Decision

If Best Static closes most of Oracle B's acceptance gain, a predicted shared future-query KV set is sufficient. If a substantial acceptance gap remains, query-specific refresh or incremental routing is necessary.
