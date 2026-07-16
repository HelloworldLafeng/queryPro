# Oracle B vs Static 10% KV Selection

## Overall acceptance

| Method | KV budget | Draft length | Mean accepted length | Full-8 rate | Zero-accept rate | Rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static endpoint mean | 10% | 8 | 3.5556 | 0.2222 | 0.1111 | 18 |
| Oracle B | 10% | 8 | 6.0000 | 0.3333 | 0.0000 | 12 |

Oracle B absolute improvement: `2.4444`.
Oracle B relative improvement: `68.75%`.
Mean attention recovery (static / Oracle B): `0.8895 / 0.9884`.

## Conditional acceptance by draft position

| Method | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 | Pos 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Static endpoint mean | 0.8889 | 0.6875 | 0.9000 | 1.0000 | 0.6667 | 0.8333 | 1.0000 | 1.0000 |
| Oracle B | 1.0000 | 1.0000 | 0.8333 | 0.9000 | 1.0000 | 1.0000 | 0.8750 | 1.0000 |

## Acceptance-length distribution

| Accepted length | Static rounds | Static ratio | Oracle B rounds | Oracle B ratio |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 0.1111 | 0 | 0.0000 |
| 1 | 5 | 0.2778 | 0 | 0.0000 |
| 2 | 2 | 0.1111 | 2 | 0.1667 |
| 3 | 0 | 0.0000 | 1 | 0.0833 |
| 4 | 3 | 0.1667 | 0 | 0.0000 |
| 5 | 1 | 0.0556 | 0 | 0.0000 |
| 6 | 1 | 0.0556 | 2 | 0.1667 |
| 7 | 0 | 0.0000 | 3 | 0.2500 |
| 8 | 4 | 0.2222 | 4 | 0.3333 |

## Input coverage

- Samples: `5`
- Input tokens (mean / median / min / max): `8192.0 / 8192.0 / 8192 / 8192`
- KV pages (mean / median / min / max): `512.0 / 512.0 / 512 / 512`
- Round-start KV pages (mean / median / min / max): `512.7 / 513.0 / 512 / 513`

## Interpretation checklist

Use the decision cases in `RUN.md`. Oracle page-scoring time is deliberately excluded; this report compares acceptance quality only.
