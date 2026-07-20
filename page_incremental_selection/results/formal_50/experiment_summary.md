# Page-level Incremental KV Selection

The three reference rows are loaded from the validated Best Static Oracle result directory; they were not rerun.

## Acceptance

| Method | Source | Mean accepted | Full-8 rate | Max update pages | Actual replacements | Macro Oracle-B gain recovery |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Static endpoint mean (reused) | reused_reference | 4.7655 | 0.2771 | — | — | 0.00% |
| Best Static Oracle (reused) | reused_reference | 5.4385 | 0.3710 | — | — | 35.80% |
| Oracle B (reused) | reused_reference | 6.5419 | 0.6674 | — | — | 100.00% |
| Page Incremental 1% | current_run | 6.2854 | 0.5843 | 1.00 | 0.99 | 89.15% |
| Page Incremental 5% | current_run | 6.3348 | 0.6222 | 3.00 | 2.89 | 92.66% |
| Page Incremental 10% | current_run | 6.3515 | 0.6281 | 6.00 | 5.29 | 93.40% |
| Page Incremental 20% | current_run | 6.4404 | 0.6491 | 11.00 | 7.85 | 96.34% |

## Conditional acceptance

| Method | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Static endpoint mean (reused) | 0.9698 | 0.8392 | 0.9157 | 0.8889 | 0.8620 | 0.8900 | 0.7354 | 0.8525 |
| Best Static Oracle (reused) | 0.9762 | 0.8574 | 0.9570 | 0.9494 | 0.9322 | 0.9226 | 0.7585 | 0.8500 |
| Oracle B (reused) | 0.9767 | 0.9634 | 0.9744 | 0.9731 | 0.9635 | 0.9676 | 0.9080 | 0.9795 |
| Page Incremental 1% | 0.9596 | 0.9667 | 0.9678 | 0.9634 | 0.9614 | 0.9355 | 0.8734 | 0.9524 |
| Page Incremental 5% | 0.9615 | 0.9591 | 0.9672 | 0.9652 | 0.9581 | 0.9528 | 0.9094 | 0.9582 |
| Page Incremental 10% | 0.9615 | 0.9688 | 0.9701 | 0.9525 | 0.9440 | 0.9669 | 0.9057 | 0.9719 |
| Page Incremental 20% | 0.9656 | 0.9635 | 0.9769 | 0.9706 | 0.9581 | 0.9705 | 0.8985 | 0.9759 |

## Paired sample comparisons

| Method | vs Static W/E/L | vs Best Static W/E/L | vs Oracle B W/E/L |
| --- | ---: | ---: | ---: |
| Page Incremental 1% | 50/0/0 | 42/6/2 | 7/28/15 |
| Page Incremental 5% | 49/0/1 | 44/3/3 | 7/31/12 |
| Page Incremental 10% | 49/0/1 | 45/2/3 | 5/36/9 |
| Page Incremental 20% | 49/1/0 | 44/4/2 | 7/33/10 |

## Interpretation

Use mean accepted length and late-position conditional acceptance as the primary evidence. Selection recall and attention recovery are diagnostics and must not replace acceptance-quality comparisons.
