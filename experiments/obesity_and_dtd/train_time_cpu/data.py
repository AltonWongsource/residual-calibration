from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray | None = None) -> None:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x/y sample counts differ")
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.indices = (
            np.arange(y.size, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if self.indices.size != y.size:
            raise ValueError("indices length differs from labels")

    def __len__(self) -> int:
        return self.y.numel()

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], int(self.indices[idx])


@dataclass
class ArraySplits:
    train_x: np.ndarray
    train_y: np.ndarray
    selection_x: np.ndarray
    selection_y: np.ndarray
    calibration_x: np.ndarray
    calibration_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    class_names: list[str]
    metadata: dict[str, Any]


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    indices: np.ndarray | None = None,
) -> DataLoader:
    return DataLoader(
        ArrayDataset(x, y, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=False,
        drop_last=False,
    )


def _dense_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _load_obesity_frames(csv_path: str | None) -> tuple[pd.DataFrame, pd.Series, str]:
    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        target_name = "NObeyesdad" if "NObeyesdad" in frame.columns else frame.columns[-1]
        return frame.drop(columns=[target_name]), frame[target_name], str(path)

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "Install ucimlrepo or provide --obesity-csv PATH to the downloaded CSV."
        ) from exc

    dataset = fetch_ucirepo(id=544)
    x = dataset.data.features.copy()
    y_frame = dataset.data.targets.copy()
    if y_frame.shape[1] != 1:
        raise ValueError(f"expected one target column, got {y_frame.shape[1]}")
    y = y_frame.iloc[:, 0]
    return x, y, "UCI id=544 via ucimlrepo"


def build_obesity_splits(
    *,
    seed: int,
    csv_path: str | None = None,
    test_fraction: float = 0.20,
    calibration_fraction: float = 0.20,
    selection_fraction: float = 0.10,
) -> ArraySplits:
    if min(test_fraction, calibration_fraction, selection_fraction) <= 0:
        raise ValueError("all split fractions must be positive")
    if test_fraction + calibration_fraction + selection_fraction >= 0.8:
        raise ValueError("too little training data remains")

    x_frame, y_series, source = _load_obesity_frames(csv_path)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_series.astype(str)).astype(np.int64)
    all_idx = np.arange(y.size, dtype=np.int64)

    remaining_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_fraction,
        random_state=seed,
        stratify=y,
    )
    calib_relative = calibration_fraction / (1.0 - test_fraction)
    remaining_idx, calibration_idx = train_test_split(
        remaining_idx,
        test_size=calib_relative,
        random_state=seed + 101,
        stratify=y[remaining_idx],
    )
    selection_relative = selection_fraction / (
        1.0 - test_fraction - calibration_fraction
    )
    train_idx, selection_idx = train_test_split(
        remaining_idx,
        test_size=selection_relative,
        random_state=seed + 202,
        stratify=y[remaining_idx],
    )

    numeric_columns = [
        c
        for c in ("Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE")
        if c in x_frame.columns
    ]
    categorical_columns = [c for c in x_frame.columns if c not in numeric_columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", _dense_one_hot_encoder(), categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    train_x = np.asarray(preprocessor.fit_transform(x_frame.iloc[train_idx]), dtype=np.float32)

    def transform(indices: np.ndarray) -> np.ndarray:
        return np.asarray(preprocessor.transform(x_frame.iloc[indices]), dtype=np.float32)

    return ArraySplits(
        train_x=train_x,
        train_y=y[train_idx],
        selection_x=transform(selection_idx),
        selection_y=y[selection_idx],
        calibration_x=transform(calibration_idx),
        calibration_y=y[calibration_idx],
        test_x=transform(test_idx),
        test_y=y[test_idx],
        class_names=[str(v) for v in label_encoder.classes_],
        metadata={
            "source": source,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "train_indices": train_idx.tolist(),
            "selection_indices": selection_idx.tolist(),
            "calibration_indices": calibration_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "preprocessing": "fit on training split only",
        },
    )


def _extract_resnet18_features(
    *,
    root: Path,
    batch_size: int,
    workers: int,
) -> dict[str, np.ndarray]:
    try:
        import medmnist
        from medmnist import INFO
        from torchvision import transforms
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError("Install medmnist and torchvision for BreastMNIST.") from exc

    info = INFO["breastmnist"]
    data_class = getattr(medmnist, info["python_class"])
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    datasets = {
        split: data_class(split=split, transform=transform, download=True, root=str(root))
        for split in ("train", "val", "test")
    }
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    output: dict[str, np.ndarray] = {}
    for split, dataset in datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            persistent_workers=workers > 0,
            pin_memory=False,
        )
        features: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        with torch.inference_mode():
            for images, target in loader:
                embedding = model(images.to(device)).float().cpu().numpy()
                features.append(embedding)
                labels.append(np.asarray(target).reshape(-1).astype(np.int64))
        output[f"{split}_features"] = np.concatenate(features, axis=0).astype(np.float32)
        output[f"{split}_labels"] = np.concatenate(labels, axis=0).astype(np.int64)
    output["class_names"] = np.asarray([str(v) for v in info["label"].values()], dtype=object)
    return output


def load_or_build_breastmnist_cache(
    *,
    data_root: str | Path,
    cache_path: str | Path,
    batch_size: int,
    workers: int,
    force_rebuild: bool = False,
) -> dict[str, np.ndarray]:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        with np.load(cache_path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _extract_resnet18_features(
        root=Path(data_root), batch_size=batch_size, workers=workers
    )
    np.savez_compressed(cache_path, **arrays)
    return arrays


def build_breastmnist_splits(
    *,
    seed: int,
    data_root: str | Path,
    cache_path: str | Path,
    feature_batch_size: int,
    workers: int,
    selection_fraction: float = 0.15,
    force_rebuild_cache: bool = False,
) -> ArraySplits:
    arrays = load_or_build_breastmnist_cache(
        data_root=data_root,
        cache_path=cache_path,
        batch_size=feature_batch_size,
        workers=workers,
        force_rebuild=force_rebuild_cache,
    )
    train_features = np.asarray(arrays["train_features"], dtype=np.float32)
    train_labels = np.asarray(arrays["train_labels"], dtype=np.int64)
    all_train_idx = np.arange(train_labels.size, dtype=np.int64)
    train_idx, selection_idx = train_test_split(
        all_train_idx,
        test_size=selection_fraction,
        random_state=seed,
        stratify=train_labels,
    )
    class_names = [str(v) for v in arrays["class_names"].tolist()]
    return ArraySplits(
        train_x=train_features[train_idx],
        train_y=train_labels[train_idx],
        selection_x=train_features[selection_idx],
        selection_y=train_labels[selection_idx],
        calibration_x=np.asarray(arrays["val_features"], dtype=np.float32),
        calibration_y=np.asarray(arrays["val_labels"], dtype=np.int64),
        test_x=np.asarray(arrays["test_features"], dtype=np.float32),
        test_y=np.asarray(arrays["test_labels"], dtype=np.int64),
        class_names=class_names,
        metadata={
            "source": "MedMNIST BreastMNIST",
            "backbone": "ImageNet-pretrained ResNet-18, frozen",
            "feature_cache": str(cache_path),
            "train_indices": train_idx.tolist(),
            "selection_indices": selection_idx.tolist(),
            "calibration_split": "official val",
            "test_split": "official test",
        },
    )
