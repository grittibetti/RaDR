from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np


def as_random_state(random_state: Optional[int | np.random.Generator]) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def ensure_bag_list(X: Any, dtype=np.float64, *, allow_nan=False) -> list[np.ndarray]:
    """Normalize supported bag containers to a list of 2D numpy arrays."""
    if hasattr(X, "to_numpy") and hasattr(X, "columns"):
        raise TypeError(
            "DataFrame inputs need BagPreprocessor or explicit dataframe_to_bags "
            "arguments before estimator.fit."
        )
    if isinstance(X, np.ndarray):
        if X.ndim == 3:
            X = [np.asarray(X[i], dtype=dtype) for i in range(X.shape[0])]
        elif X.dtype == object:
            X = [np.asarray(bag, dtype=dtype) for bag in X.tolist()]
        else:
            raise ValueError(f"Expected a 3D ndarray of bags, got shape {X.shape}.")
    if not isinstance(X, Sequence):
        raise TypeError("X must be a sequence of 2D bag arrays or a 3D ndarray.")

    bags: list[np.ndarray] = []
    input_dim = None
    for i, bag in enumerate(X):
        arr = np.asarray(bag, dtype=dtype)
        if arr.ndim != 2:
            raise ValueError(f"Bag {i} must be 2D, got shape {arr.shape}.")
        if arr.shape[1] == 0 or np.isinf(arr).any() or (not allow_nan and np.isnan(arr).any()):
            raise ValueError(f"Bag {i} must have features and finite numeric values.")
        if arr.shape[0] == 0:
            raise ValueError(f"Bag {i} is empty.")
        if input_dim is None:
            input_dim = arr.shape[1]
        elif arr.shape[1] != input_dim:
            raise ValueError(
                f"All bags must share feature dimension {input_dim}; bag {i} has dimension {arr.shape[1]}."
            )
        bags.append(arr)
    if not bags:
        raise ValueError("At least one bag is required.")
    return bags


def stack_points_from_bags(
    X: Any,
    *,
    max_points: Optional[int] = 2048,
    random_state: Optional[int | np.random.Generator] = None,
    dtype=np.float64,
) -> np.ndarray:
    bags = ensure_bag_list(X, dtype=dtype)
    total = sum(len(bag) for bag in bags)
    if max_points is None or total <= int(max_points):
        return np.concatenate(bags, axis=0)
    if int(max_points) < 1:
        raise ValueError("max_points must be positive.")
    rng = as_random_state(random_state)
    indices = np.sort(rng.choice(total, size=int(max_points), replace=False))
    pieces = []
    offset = 0
    for bag in bags:
        local = indices[(indices >= offset) & (indices < offset + len(bag))] - offset
        if len(local):
            pieces.append(bag[local])
        offset += len(bag)
    return np.concatenate(pieces, axis=0)


def pairwise_sqdist(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Y = X if Y is None else np.asarray(Y, dtype=np.float64)
    x_norm = np.sum(X * X, axis=1, keepdims=True)
    y_norm = np.sum(Y * Y, axis=1, keepdims=True)
    dist = x_norm + y_norm.T - 2.0 * X @ Y.T
    return np.maximum(dist, 0.0)


def median_heuristic_gamma(
    X: np.ndarray,
    *,
    fallback: float = 1.0,
    max_points: Optional[int] = 2048,
    random_state: Optional[int | np.random.Generator] = None,
) -> float:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or len(X) < 2:
        return float(fallback)
    if max_points is not None and len(X) > int(max_points):
        rng = as_random_state(random_state)
        idx = rng.choice(len(X), size=int(max_points), replace=False)
        X = X[np.sort(idx)]
    sqdist = pairwise_sqdist(X)
    tri = sqdist[np.triu_indices_from(sqdist, k=1)]
    tri = tri[np.isfinite(tri) & (tri > 1e-12)]
    if tri.size == 0:
        return float(fallback)
    med = float(np.median(tri))
    if med <= 0.0 or not np.isfinite(med):
        return float(fallback)
    return float(1.0 / med)


def candidate_scale_values(base: float, multipliers) -> list[float]:
    values = []
    for multiplier in multipliers:
        value = float(base) * float(multiplier)
        if value > 0.0 and np.isfinite(value):
            values.append(value)
    return sorted(set(values)) or [float(base) if float(base) > 0 else 1.0]
