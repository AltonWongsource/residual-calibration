# CIFAR-100 / frozen ResNet50

This script reproduces the paper's CIFAR-100 experiment with a frozen ImageNet-pretrained ResNet50 and a multinomial logistic-regression probe. It uses seed 42, 40,000 CIFAR-100 images for probe training, 10,000 for calibration, and the official 10,000-image test set. Dataset images, extracted features, and model checkpoints are not included.

```bash
pip install torch torchvision numpy pandas scipy scikit-learn pillow
python run.py --dataset cifar100 --backbone resnet50 --seed 42 --batch-size 64
```

The script saves extracted features, a trained probe, calibrated probabilities, `cifar100_resnet50_detailed.csv`, and `summary.json` under `outputs/vision_cifar100_resnet50/`. The summary lists each method by name; the detailed CSV includes NLL, accuracy, ECE, and ACE.

For the paper's VS comparison, expected NLL is 1.0136393590 for VS and 1.0136006189 for VS+LAPE; accuracy is 0.7147 for both on the 10,000-image test set.
