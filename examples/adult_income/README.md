# Adult Income real-data check

This VS-anchor example uses OpenML Adult dataset 1590 (48,842 records; seed 42; 50/25/25 train/calibration/test split).

| Method | NLL ↓ | Accuracy ↑ |
| --- | ---: | ---: |
| Base | 0.5680 | 82.87% |
| VS | 0.3696 | 82.61% |
| VS+LAPE | 0.3439 | 84.64% |

The VS-to-VS+LAPE relative NLL reduction is 6.97%, matching the paper appendix. For data preparation and commands, see [`experiments/adult_income/README.md`](../../experiments/adult_income/README.md).
