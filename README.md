# LAPE: local refinement of calibration anchors

This repository provides selected experiment code and a usage demo for *Residual Calibration via Local Feature-Space Refinement*. Each experiment folder describes its data, command, and outputs. The [`examples/`](examples/) folder contains result previews.

## Selected reproduction tracks

| Track | Setup | Entry point | Required external assets |
| --- | --- | --- | --- |
| Obesity Risk / MLP | Real 7-class tabular classification; train-time CE/MDCA/MbLS and post-hoc comparisons | `experiments/obesity_and_dtd/train_time_cpu/run.py` | UCI Obesity dataset (script can fetch it) |
| BreastMNIST / ResNet18 | Real medical-image transfer setting from the paper's main comparison | `experiments/breastmnist/train_export.py`, then `demo/run_demo.py --npz ...` | Official MedMNIST `breastmnist_224.npz`, ImageNet ResNet18 weights |
| DTD / cached CoOp + DAC | Paper's real ViT-B/32 prompt-tuning extension, with five novel-reference splits | `experiments/dtd_coop_dac/tools/` then `experiments/dtd_coop_dac/run_fast_rebuttal.py` | DTD images, official Zhou split JSON, OpenCLIP ViT-B/32 weights |
| Letter Recognition / MLP | Real 26-class tabular benchmark that complements Obesity | `experiments/letter_recognition/train_export.py` | Official UCI Letter Recognition CSV |
| UCI Multiple Features / CFAF-LAPE | Real two-view handwritten-digit classification using calibration-only fusion selection | `experiments/multiview_digits/run.py` | Two OpenML feature views (script fetches them) |
| Adult Income / MLP | Real socioeconomic tabular extension using the original split/model protocol | `experiments/adult_income/train_export.py` | OpenML dataset 1590 (script fetches it) |
| CIFAR-10-C / ResNet18 linear probe | Corrupted-image target shift with labelled target calibration | `experiments/cifar10c/run.py` | CIFAR-10 and CIFAR-10-C images, ImageNet ResNet18 weights; or real-image feature caches |
| CIFAR-100 / ResNet50 linear probe | Frozen ResNet50 features and a 100-class linear probe | `experiments/cifar100/run.py --dataset cifar100 --backbone resnet50` | CIFAR-100 images and ImageNet ResNet50 weights |
| HAM10000 / image + metadata CFAF | Lesion-level split and joint image/clinical-metadata calibration | `experiments/ham10000/run.py --task ham10000_melanoma_nevus` | HAM10000 JPEGs, matching metadata, ImageNet ResNet18 weights |

Datasets and pretrained weights are obtained from their providers; generated caches and outputs stay in each experiment directory. Image experiments can require substantial compute. See each experiment README and script's `--help` for details.

## Try the method now

Install the small demo environment and run the included BreastMNIST / ResNet18 model outputs:

```bash
pip install -r requirements-demo.txt
python demo/run_demo.py
python demo/make_walkthrough.py demo/output
```

Open `demo/output/report.html` to see the result. The included NPZ contains calibration/test features and logits from the BreastMNIST ResNet18 run, not images or pretrained weights. The command refits a VS anchor, creates five-fold out-of-fold anchor probabilities, and applies `LAPECalibrator` from `src/lape_core.py`. It writes `report.html`, `comparison.png`, `metrics.json`, and (with the second command) `walkthrough.gif` under `demo/output/`. The same run is previewed in [`examples/breastmnist/`](examples/breastmnist/).

To try the small scikit-learn MLP example instead, run `python demo/run_demo.py --sklearn-example`. Its results are illustrative and can differ from the paper experiment.

For your own frozen classifier or an exported paper-model artifact:

```bash
python demo/run_demo.py --npz path/to/model_outputs.npz --anchor VS --output demo/output_my_model
```

For the two-file exports produced by the Obesity or DTD scripts:

```bash
python demo/run_demo.py --calibration path/to/calibration.npz --test path/to/test.npz --anchor VS --output demo/output_my_model
```

The NPZ must contain calibration and held-out test arrays: `features_cal`, `features_test`, `logits_cal`, `logits_test`, `labels_cal`, `labels_test`. The original experiment aliases (`h_cal`, `h_test`, `l_cal`, `l_test`, `y_cal`, `y_test`) also work. Features and logits must come from a model trained without using calibration/test labels. Test labels are used **only** to calculate displayed metrics.

`src/lape_core.py` contains `TemperatureScaling`, `VectorScaling`, and `LAPECalibrator` from the experiment runner. The demo uses out-of-fold anchor predictions for LAPE's reference samples. Individual experiment scripts specify their own model and calibration settings.

## Implementation notes

- The demo covers TS, VS, and their LAPE refinements; the experiment scripts contain the corresponding paper settings.
- DTD uses cached augmented image views for CoOp training. DAC is evaluated without novel-class labels; the LAPE extension uses labelled novel-class reference samples.
- CIFAR-10-C uses a labelled target-domain calibration split. The included entry point covers Base, TS, VS, Dir, ETS, LTS, Pure LAPE, and anchor-plus-LAPE variants; its cache mode requires real-image features.
- HAM10000 uses the `concat/std/cosine, alpha<=0.5` CFAF-LAPE configuration reported in the paper.
- Dataset and pretrained-weight use remains subject to their providers' terms. No repository license has been selected yet.

## Verification

Run `python -m pytest -q` in the experiment directories that include tests. The Obesity, DTD, and CIFAR-10-C test suites pass locally. Full image experiments require the datasets and pretrained weights listed above.
