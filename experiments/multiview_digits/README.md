# Real UCI Multiple Features / CFAF-LAPE

This two-view experiment uses OpenML `mfeat-fourier` and `mfeat-zernike`, treating digits 0–4 versus 5–9 as a binary task. It trains separate MLPs, cross-fits VS anchors on the calibration set, learns a fusion weight there, and applies LAPE to joint hidden features. The loader checks row-label alignment across the two views.

```bash
python experiments/multiview_digits/run.py
```

The command downloads the two OpenML views on first use and writes `experiments/multiview_digits/outputs/metrics.json`. With seed 42 and float64 OpenML preprocessing, expected NLL is 0.1117256041 for late fusion and 0.0919494182 for CFAF-LAPE, with accuracy 96.0% and 96.4%.
