# UCI Multiple Features results

The CFAF-LAPE script uses the OpenML Fourier and Zernike views (2,000 aligned samples; seed 42). The table summarizes the 500-sample test split.

| Method | NLL ↓ | ECE ↓ | Accuracy ↑ |
| --- | ---: | ---: | ---: |
| Late fusion | 0.1117 | 0.0294 | 96.0% |
| CFAF-LAPE | 0.0919 | 0.0252 | 96.4% |

To reproduce these values, run [`experiments/multiview_digits/run.py`](../../experiments/multiview_digits/README.md) with its default seed. The script retains OpenML's float64 input through preprocessing.
