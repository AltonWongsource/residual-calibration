from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
REPORT_DIR = ROOT / "outputs"
ARTIFACT_PATH = REPORT_DIR / "breastmnist_resnet18_partialft_artifacts.npz"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GrayToRGB:
    def __call__(self, img: Any) -> Any:
        return img.convert("RGB")


class BreastNpzDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: Any) -> None:
        self.images = images
        self.labels = labels.reshape(-1).astype(int)
        self.transform = transform

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        from PIL import Image

        image = Image.fromarray(self.images[idx], mode="L")
        image = self.transform(image)
        return image, int(self.labels[idx])


class ResNet18PartialFT(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

        # freeze all first
        for p in self.stem.parameters():
            p.requires_grad = False
        # unfreeze final stage only
        for p in backbone.layer4.parameters():
            p.requires_grad = True
        for p in self.fc.parameters():
            p.requires_grad = True

        # reuse modules from backbone for proper params
        self.layer4 = backbone.layer4

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.stem(x).flatten(1)
        return feats

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Any:
        feats = self.extract_features(x)
        logits = self.fc(feats)
        if return_features:
            return logits, feats
        return logits


@dataclass
class SplitArtifacts:
    logits: np.ndarray
    probs: np.ndarray
    features: np.ndarray
    labels: np.ndarray


def make_transform() -> transforms.Compose:
    mean = ResNet18_Weights.IMAGENET1K_V1.transforms().mean
    std = ResNet18_Weights.IMAGENET1K_V1.transforms().std
    return transforms.Compose(
        [
            GrayToRGB(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_arrays() -> dict[str, np.ndarray]:
    path = DATA_ROOT / "breastmnist_224.npz"
    arr = np.load(path)
    return {k: arr[k] for k in arr.files}


def build_dataloaders(batch_size: int = 24) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    arr = load_arrays()
    train_images = arr["train_images"]
    train_labels = arr["train_labels"].reshape(-1)
    train_idx, hold_idx = train_test_split(
        np.arange(len(train_labels)),
        test_size=0.1,
        random_state=42,
        stratify=train_labels,
    )
    tfm = make_transform()
    train_loader = DataLoader(
        BreastNpzDataset(train_images[train_idx], train_labels[train_idx], tfm),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    hold_loader = DataLoader(
        BreastNpzDataset(train_images[hold_idx], train_labels[hold_idx], tfm),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    cal_loader = DataLoader(
        BreastNpzDataset(arr["val_images"], arr["val_labels"], tfm),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        BreastNpzDataset(arr["test_images"], arr["test_labels"], tfm),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, hold_loader, cal_loader, test_loader


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * len(labels)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(len(labels))
    return total_loss / max(total, 1), total_correct / max(total, 1)


def train_model(model: ResNet18PartialFT, train_loader: DataLoader, hold_loader: DataLoader, device: torch.device, epochs: int = 10) -> tuple[ResNet18PartialFT, list[dict[str, float]], float]:
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-4)
    best_state = None
    best_hold_nll = float("inf")
    history: list[dict[str, float]] = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(len(labels))
        train_nll = total_loss / max(total, 1)
        train_acc = total_correct / max(total, 1)
        hold_nll, hold_acc = evaluate_model(model, hold_loader, criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_nll": float(train_nll),
                "train_acc": float(train_acc),
                "hold_nll": float(hold_nll),
                "hold_acc": float(hold_acc),
            }
        )
        print(
            f"[breast-partialft] epoch={epoch} train_nll={train_nll:.4f} train_acc={train_acc:.4f} "
            f"hold_nll={hold_nll:.4f} hold_acc={hold_acc:.4f}"
        )
        if hold_nll < best_hold_nll:
            best_hold_nll = hold_nll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, time.perf_counter() - start


def collect_split_artifacts(model: ResNet18PartialFT, loader: DataLoader, device: torch.device) -> SplitArtifacts:
    model.eval()
    logits_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    features_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits, features = model(images, return_features=True)
            probs = torch.softmax(logits, dim=1)
            logits_all.append(logits.cpu().numpy().astype(np.float64))
            probs_all.append(probs.cpu().numpy().astype(np.float64))
            features_all.append(features.cpu().numpy().astype(np.float32))
            labels_all.append(labels.numpy().astype(int))
    return SplitArtifacts(
        logits=np.concatenate(logits_all, axis=0),
        probs=np.concatenate(probs_all, axis=0),
        features=np.concatenate(features_all, axis=0),
        labels=np.concatenate(labels_all, axis=0),
    )


def main() -> None:
    set_seed(42)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    print(f"[breast-partialft] device={device}")
    train_loader, hold_loader, cal_loader, test_loader = build_dataloaders()
    model = ResNet18PartialFT().to(device)
    model, history, train_seconds = train_model(model, train_loader, hold_loader, device=device, epochs=10)
    cal_art = collect_split_artifacts(model, cal_loader, device)
    test_art = collect_split_artifacts(model, test_loader, device)
    artifacts = {
        "h_cal": cal_art.features,
        "y_cal": cal_art.labels,
        "p_cal": cal_art.probs,
        "l_cal": cal_art.logits,
        "h_test": test_art.features,
        "y_test": test_art.labels,
        "p_test": test_art.probs,
        "l_test": test_art.logits,
        "history": history,
    }
    np.savez_compressed(ARTIFACT_PATH, **artifacts)
    print(f"Saved calibration/test features and logits to {ARTIFACT_PATH}")
    print(f"Training time: {train_seconds:.1f} s")


if __name__ == "__main__":
    main()
