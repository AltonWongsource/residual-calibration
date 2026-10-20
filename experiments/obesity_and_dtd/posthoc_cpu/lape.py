from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


_EPS = 1e-12


@dataclass(frozen=True)
class LAPEConfig:
    alpha_grid: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 21).tolist())
    m_grid: tuple[int, ...] | None = None
    feature_mode: str = "none"  # none | zscore | l2
    pca_dim: int = 0
    probability_floor: float = 1e-12
    tie_tolerance: float = 1e-12
    neighbor_algorithm: str = "auto"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class LAPEResult:
    probs: np.ndarray
    selected_m: np.ndarray
    selected_alpha: np.ndarray
    objective: np.ndarray
    bandwidth: np.ndarray
    nearest_distance: np.ndarray
    m_grid: np.ndarray
    alpha_grid: np.ndarray


def default_m_grid(n_reference: int) -> np.ndarray:
    if n_reference < 2:
        raise ValueError("LAPE requires at least two reference examples")
    root = np.sqrt(n_reference)
    m_min = max(2, int(np.floor(root / 10.0)))
    m_max = max(m_min, int(np.floor(root / 2.0)))
    m_max = min(m_max, n_reference)
    return np.arange(m_min, m_max + 1, dtype=np.int64)


def _validate_probs(probs: np.ndarray, n: int, k: int | None = None) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != n:
        raise ValueError("probability array has invalid shape")
    if k is not None and p.shape[1] != k:
        raise ValueError("probability arrays have inconsistent class counts")
    if np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    sums = p.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("probability rows must have positive mass")
    p = p / sums
    return p


def _prepare_features(
    reference_features: np.ndarray,
    query_features: np.ndarray,
    *,
    mode: str,
    pca_dim: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    ref = np.asarray(reference_features, dtype=np.float64)
    query = np.asarray(query_features, dtype=np.float64)
    if ref.ndim != 2 or query.ndim != 2 or ref.shape[1] != query.shape[1]:
        raise ValueError("reference/query feature arrays must be compatible matrices")
    metadata: dict[str, object] = {"feature_mode": mode, "input_dim": ref.shape[1]}
    if mode == "zscore":
        scaler = StandardScaler().fit(ref)
        ref = scaler.transform(ref)
        query = scaler.transform(query)
    elif mode == "l2":
        ref_norm = np.linalg.norm(ref, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        ref = ref / np.maximum(ref_norm, _EPS)
        query = query / np.maximum(query_norm, _EPS)
    elif mode != "none":
        raise ValueError("feature_mode must be one of: none, zscore, l2")

    if pca_dim > 0 and pca_dim < ref.shape[1]:
        max_dim = min(pca_dim, ref.shape[0] - 1, ref.shape[1])
        if max_dim >= 1:
            pca = PCA(n_components=max_dim, random_state=0)
            ref = pca.fit_transform(ref)
            query = pca.transform(query)
            metadata["pca_dim"] = int(max_dim)
            metadata["pca_explained_variance"] = float(
                pca.explained_variance_ratio_.sum()
            )
    metadata["output_dim"] = ref.shape[1]
    return ref.astype(np.float32), query.astype(np.float32), metadata


def _reference_neighbors_including_self(
    features: np.ndarray,
    max_m: int,
    *,
    algorithm: str,
) -> tuple[np.ndarray, np.ndarray]:
    n = features.shape[0]
    request = min(n, max_m + 1)
    nn = NearestNeighbors(n_neighbors=request, metric="euclidean", algorithm=algorithm)
    nn.fit(features)
    distances, indices = nn.kneighbors(features, return_distance=True)
    output_idx = np.empty((n, max_m), dtype=np.int64)
    output_dist = np.empty((n, max_m), dtype=np.float64)
    for i in range(n):
        row_idx = indices[i].tolist()
        row_dist = distances[i].tolist()
        filtered = [(j, d) for j, d in zip(row_idx, row_dist) if j != i]
        ordered = [(i, 0.0)] + filtered
        if len(ordered) < max_m:
            # This can only occur when max_m == n and all available points were requested.
            seen = {j for j, _ in ordered}
            for j in range(n):
                if j not in seen:
                    ordered.append((j, float(np.linalg.norm(features[i] - features[j]))))
        ordered = ordered[:max_m]
        output_idx[i] = [j for j, _ in ordered]
        output_dist[i] = [d for _, d in ordered]
    return output_dist, output_idx


def _precompute_reference_loo_true_label_probs(
    labels: np.ndarray,
    neighbors_including_self: np.ndarray,
    m_grid: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    n = labels.size
    one_hot = np.eye(num_classes, dtype=np.float64)[labels]
    out = np.empty((m_grid.size, n), dtype=np.float64)
    for pos, m in enumerate(m_grid):
        idx = neighbors_including_self[:, :m]
        counts = one_hot[idx].sum(axis=1) - one_hot
        local = (counts + 1.0 / num_classes) / float(m)
        out[pos] = local[np.arange(n), labels]
    return out


def fit_predict_lape(
    *,
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    reference_anchor_probs: np.ndarray,
    query_features: np.ndarray,
    query_anchor_probs: np.ndarray,
    config: LAPEConfig | None = None,
) -> tuple[LAPEResult, dict[str, object]]:
    """Apply Algorithm 1 from the paper under a supplied anchor.

    Reference anchor scores should be out-of-fold whenever the same labeled
    reference set was used to fit the post-hoc anchor.  Query anchor scores are
    obtained from an anchor refitted on the full reference set.
    """
    cfg = config or LAPEConfig()
    ref_y = np.asarray(reference_labels, dtype=np.int64).reshape(-1)
    n = ref_y.size
    if n < 2:
        raise ValueError("LAPE requires at least two reference points")
    ref_anchor = _validate_probs(reference_anchor_probs, n)
    k = ref_anchor.shape[1]
    if ref_y.min() < 0 or ref_y.max() >= k:
        raise ValueError("reference labels contain invalid class indices")
    query_anchor = _validate_probs(query_anchor_probs, len(query_features), k)

    m_grid = (
        default_m_grid(n)
        if cfg.m_grid is None
        else np.unique(np.asarray(cfg.m_grid, dtype=np.int64))
    )
    m_grid = m_grid[(m_grid >= 2) & (m_grid <= n)]
    if m_grid.size == 0:
        raise ValueError("m_grid contains no valid neighborhood sizes")
    alpha_grid = np.unique(np.clip(np.asarray(cfg.alpha_grid, dtype=np.float64), 0.0, 1.0))
    if alpha_grid.size == 0:
        raise ValueError("alpha_grid must be non-empty")

    ref_z, query_z, feature_meta = _prepare_features(
        reference_features,
        query_features,
        mode=cfg.feature_mode,
        pca_dim=cfg.pca_dim,
    )
    max_m = int(m_grid.max())
    bandwidth_m = min(n, max(1, int(np.floor(np.sqrt(n) / 2.0))))
    retrieve_m = max(max_m, bandwidth_m)

    _, ref_neighbors = _reference_neighbors_including_self(
        ref_z, retrieve_m, algorithm=cfg.neighbor_algorithm
    )
    ref_loo_true = _precompute_reference_loo_true_label_probs(
        ref_y, ref_neighbors, m_grid, k
    )

    query_nn = NearestNeighbors(
        n_neighbors=retrieve_m, metric="euclidean", algorithm=cfg.neighbor_algorithm
    ).fit(ref_z)
    query_dist, query_idx = query_nn.kneighbors(query_z, return_distance=True)

    qn = query_z.shape[0]
    output = np.empty((qn, k), dtype=np.float64)
    selected_m = np.empty(qn, dtype=np.int64)
    selected_alpha = np.empty(qn, dtype=np.float64)
    selected_objective = np.empty(qn, dtype=np.float64)
    bandwidth = np.empty(qn, dtype=np.float64)
    nearest_distance = query_dist[:, 0].astype(np.float64)
    one_hot = np.eye(k, dtype=np.float64)[ref_y]
    floor = max(float(cfg.probability_floor), _EPS)

    for q in range(qn):
        sigma = max(float(query_dist[q, bandwidth_m - 1]), 1e-12)
        bandwidth[q] = sigma
        best_key: tuple[float, float, int] | None = None
        best_m = int(m_grid[0])
        best_alpha = float(alpha_grid[0])
        best_obj = float("inf")

        for m_pos, m_value in enumerate(m_grid):
            m = int(m_value)
            idx = query_idx[q, :m]
            distances = query_dist[q, :m]
            raw_w = np.exp(-(distances * distances) / (sigma * sigma))
            weights = raw_w / max(float(raw_w.sum()), floor)
            anchor_y = ref_anchor[idx, ref_y[idx]]
            local_y = ref_loo_true[m_pos, idx]

            candidate_y = (
                (1.0 - alpha_grid[:, None]) * anchor_y[None, :]
                + alpha_grid[:, None] * local_y[None, :]
            )
            objectives = -np.sum(weights[None, :] * np.log(np.clip(candidate_y, floor, 1.0)), axis=1)
            alpha_pos = int(np.argmin(objectives))
            obj = float(objectives[alpha_pos])
            alpha = float(alpha_grid[alpha_pos])
            # Stable tie-breaking: preserve more anchor, then use the smaller neighborhood.
            key = (round(obj / max(cfg.tie_tolerance, 1e-15)) * max(cfg.tie_tolerance, 1e-15), alpha, m)
            if best_key is None or key < best_key:
                best_key = key
                best_m = m
                best_alpha = alpha
                best_obj = obj

        final_idx = query_idx[q, :best_m]
        counts = one_hot[final_idx].sum(axis=0)
        local_probability = (counts + 1.0 / k) / float(best_m + 1)
        final = (1.0 - best_alpha) * query_anchor[q] + best_alpha * local_probability
        final = np.clip(final, floor, None)
        output[q] = final / final.sum()
        selected_m[q] = best_m
        selected_alpha[q] = best_alpha
        selected_objective[q] = best_obj

    result = LAPEResult(
        probs=output.astype(np.float32),
        selected_m=selected_m,
        selected_alpha=selected_alpha.astype(np.float32),
        objective=selected_objective.astype(np.float32),
        bandwidth=bandwidth.astype(np.float32),
        nearest_distance=nearest_distance.astype(np.float32),
        m_grid=m_grid,
        alpha_grid=alpha_grid.astype(np.float32),
    )
    metadata = {
        "config": cfg.to_dict(),
        "n_reference": n,
        "num_classes": k,
        "bandwidth_neighbor_rank": bandwidth_m,
        "feature_processing": feature_meta,
        "mean_selected_m": float(selected_m.mean()),
        "mean_selected_alpha": float(selected_alpha.mean()),
        "fraction_alpha_zero": float(np.mean(selected_alpha == 0.0)),
        "fraction_alpha_one": float(np.mean(selected_alpha == 1.0)),
    }
    return result, metadata


def pure_local_config(config: LAPEConfig | None = None) -> LAPEConfig:
    cfg = config or LAPEConfig()
    return LAPEConfig(
        alpha_grid=(1.0,),
        m_grid=cfg.m_grid,
        feature_mode=cfg.feature_mode,
        pca_dim=cfg.pca_dim,
        probability_floor=cfg.probability_floor,
        tie_tolerance=cfg.tie_tolerance,
        neighbor_algorithm=cfg.neighbor_algorithm,
    )
