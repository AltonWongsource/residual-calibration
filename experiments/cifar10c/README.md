# CIFAR-10-C target-domain calibration

This is the real-image corruption experiment behind the paper's Gaussian-noise Table 9. A frozen ImageNet-pretrained ResNet18 extracts features; a multinomial linear probe is fitted on **clean CIFAR-10 train** only. Each corrupted target condition has a stratified 20% labelled calibration/reference split and an untouched 80% test split, repeated for seeds 42–46. The same target reference samples are given to every anchor and LAPE refinement. Target labels are needed: this is not unsupervised test-time adaptation.

`calibration_core.py` contains the calibration methods used by this runner. The entry point evaluates Base, TS, VS, Dir, ETS, LTS, Pure LAPE, and anchor-plus-LAPE variants. The paper table additionally reports IROvA, MIR, FC, and SAG baselines.

Install PyTorch, torchvision, numpy, pandas, scikit-learn, scipy, requests, and Pillow. From this directory:

```bash
python run.py --corruption gaussian_noise
```

The first run downloads clean CIFAR-10 and the approximately 2.9 GB CIFAR-10-C archive, then builds reusable image-feature and probe caches. By default it evaluates Gaussian-noise severities 1, 3, and 5 at seeds 42–46. `--corruption motion_blur` or `snow` runs optional extensions. Data and pretrained weights are obtained separately.

To reuse feature/probe caches from real CIFAR images, run:

```bash
python run.py --cache-only --cache-dir PATH_TO_REAL_FEATURE_CACHE --corruption gaussian_noise --severity 1 --seeds 42 --output-dir outputs/check_s1_seed42
```

`--cache-only` requires both the probe and target feature caches. Each condition/seed CSV is written after its methods finish; `--summarize-only --output-dir ...` recomputes aggregates from those CSVs.
