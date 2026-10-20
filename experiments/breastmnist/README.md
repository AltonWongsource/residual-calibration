# BreastMNIST / ResNet18

Place the official `breastmnist_224.npz` in `data/` and make ImageNet ResNet18 weights available to torchvision. From the repository root, run:

```bash
python experiments/breastmnist/train_export.py
python demo/run_demo.py --npz experiments/breastmnist/outputs/breastmnist_resnet18_partialft_artifacts.npz --anchor VS --output demo/output_breastmnist
```

The first command trains a lightly fine-tuned ResNet18 with train/holdout/calibration/test splits and exports features and logits. The second evaluates VS and LAPE using five-fold out-of-fold reference probabilities. The resulting report shows the VS-anchor comparison.
