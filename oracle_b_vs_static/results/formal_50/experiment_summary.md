# Oracle B vs Static 10% KV Selection

## Overall acceptance

| Method | KV budget | Draft length | Mean accepted length | Full-8 rate | Zero-accept rate | Rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static endpoint mean | 10% | 8 | 4.7655 | 0.2771 | 0.0302 | 563 |
| Oracle B | 10% | 8 | 6.5419 | 0.6674 | 0.0233 | 430 |

Oracle B absolute improvement: `1.7763`.
Oracle B relative improvement: `37.27%`.
Mean attention recovery (static / Oracle B): `0.9183 / 0.9916`.

## Conditional acceptance by draft position

| Method | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 | Pos 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Static endpoint mean | 0.9698 | 0.8392 | 0.9157 | 0.8889 | 0.8620 | 0.8900 | 0.7354 | 0.8525 |
| Oracle B | 0.9767 | 0.9634 | 0.9744 | 0.9731 | 0.9635 | 0.9676 | 0.9080 | 0.9795 |

## Acceptance-length distribution

| Accepted length | Static rounds | Static ratio | Oracle B rounds | Oracle B ratio |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 17 | 0.0302 | 10 | 0.0233 |
| 1 | 92 | 0.1634 | 25 | 0.0581 |
| 2 | 41 | 0.0728 | 14 | 0.0326 |
| 3 | 53 | 0.0941 | 19 | 0.0442 |
| 4 | 54 | 0.0959 | 19 | 0.0442 |
| 5 | 47 | 0.0835 | 14 | 0.0326 |
| 6 | 70 | 0.1243 | 33 | 0.0767 |
| 7 | 33 | 0.0586 | 9 | 0.0209 |
| 8 | 156 | 0.2771 | 287 | 0.6674 |

## Input coverage

- Samples: `50`
- Input tokens (mean / median / min / max): `8192.0 / 8192.0 / 8192 / 8192`
- KV pages (mean / median / min / max): `512.0 / 512.0 / 512 / 512`
- Round-start KV pages (mean / median / min / max): `514.3 / 514.0 / 512 / 516`

## Interpretation checklist

Use the decision cases in `RUN.md`. Oracle page-scoring time is deliberately excluded; this report compares acceptance quality only.
