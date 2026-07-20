# Page-level Incremental KV Selection

The three reference rows are loaded from the validated Best Static Oracle result directory; they were not rerun.

## Acceptance

| Method | Source | Mean accepted | Full-8 rate | Max update pages | Actual replacements | Macro Oracle-B gain recovery |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Static endpoint mean (reused) | reused_reference | 3.5556 | 0.2222 | — | — | 0.00% |
| Best Static Oracle (reused) | reused_reference | 4.1875 | 0.2500 | — | — | 18.44% |
| Oracle B (reused) | reused_reference | 6.0000 | 0.3333 | — | — | 100.00% |
| Page Incremental 1% | current_run | 4.8571 | 0.2857 | 1.00 | 0.98 | 66.78% |
| Page Incremental 5% | current_run | 5.3846 | 0.3077 | 3.00 | 2.75 | 90.43% |
| Page Incremental 10% | current_run | 5.3846 | 0.3077 | 6.00 | 4.93 | 90.43% |
| Page Incremental 20% | current_run | 5.3846 | 0.3077 | 11.00 | 7.39 | 90.43% |

## Conditional acceptance

| Method | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Static endpoint mean (reused) | 0.8889 | 0.6875 | 0.9000 | 1.0000 | 0.6667 | 0.8333 | 1.0000 | 1.0000 |
| Best Static Oracle (reused) | 0.9375 | 0.6667 | 1.0000 | 1.0000 | 0.8000 | 1.0000 | 0.8000 | 1.0000 |
| Oracle B (reused) | 1.0000 | 1.0000 | 0.8333 | 0.9000 | 1.0000 | 1.0000 | 0.8750 | 1.0000 |
| Page Incremental 1% | 0.9286 | 0.9231 | 0.9167 | 0.8000 | 1.0000 | 1.0000 | 0.5714 | 1.0000 |
| Page Incremental 5% | 0.9231 | 0.9167 | 0.9091 | 0.9000 | 1.0000 | 1.0000 | 0.7500 | 1.0000 |
| Page Incremental 10% | 0.9231 | 0.9167 | 0.9091 | 0.9000 | 1.0000 | 1.0000 | 0.7500 | 1.0000 |
| Page Incremental 20% | 0.9231 | 0.9167 | 0.9091 | 0.9000 | 1.0000 | 1.0000 | 0.7500 | 1.0000 |

## Paired sample comparisons

| Method | vs Static W/E/L | vs Best Static W/E/L | vs Oracle B W/E/L |
| --- | ---: | ---: | ---: |
| Page Incremental 1% | 4/1/0 | 3/1/1 | 0/1/4 |
| Page Incremental 5% | 4/1/0 | 4/0/1 | 0/3/2 |
| Page Incremental 10% | 4/1/0 | 4/0/1 | 0/3/2 |
| Page Incremental 20% | 4/1/0 | 4/0/1 | 0/3/2 |

## Interpretation

Use mean accepted length and late-position conditional acceptance as the primary evidence. Selection recall and attention recovery are diagnostics and must not replace acceptance-quality comparisons.
