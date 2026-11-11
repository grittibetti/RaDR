from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def center_columns(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=np.float64)
    return F - F.mean(axis=0, keepdims=True)


def distribution_variance_scores(F: np.ndarray, y=None, task: Optional[str] = None) -> np.ndarray:
    """Distribution variance score, paper Eq. (44), on pool-normalized features."""
    Fc = center_columns(F)
    return F.shape[1] * np.sum(Fc * Fc, axis=0)


def label_alignment_scores(F: np.ndarray, y, task: str = "classification") -> np.ndarray:
    """Squared kernel alignment score, paper Eq. (39), without target centering."""
    F = np.asarray(F, dtype=np.float64)
    y = np.asarray(y)
    if len(y) != F.shape[0]:
        raise ValueError("Expected one target for each row of F.")

    if task == "classification":
        classes = np.unique(y)
        Y = (
            np.where(y == classes[-1], 1.0, -1.0)[:, None]
            if len(classes) <= 2
            else (y[:, None] == classes[None, :]).astype(float)
        )
    else:
        Y = y.astype(float).reshape(-1, 1)
    # Pool-size factor recovers unnormalized candidate vectors (paper Eq. 33).
    return F.shape[1] * np.sum((Y.T @ F) ** 2, axis=0)


def empirical_leverage_scores(
    F: np.ndarray, y=None, task: Optional[str] = None, ridge: Optional[float] = None
) -> np.ndarray:
    """Empirical leverage scores used by the ARaDR leverage selector."""
    F = np.asarray(F, dtype=np.float64)
    if F.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {F.shape}.")
    n_samples, n_features = F.shape
    if n_samples == 0 or n_features == 0:
        return np.zeros(n_features, dtype=np.float64)
    ridge_value = 1.0 / n_samples if ridge is None else float(ridge)
    if ridge_value <= 0 or not np.isfinite(ridge_value):
        raise ValueError("ridge must be finite and strictly positive.")
    # Thin SVD avoids a pool-by-pool inverse and handles rank deficiency.
    _, singular, vt = np.linalg.svd(F, full_matrices=False)
    factors = singular**2 / (singular**2 + ridge_value)
    return n_features * np.sum(factors[:, None] * vt**2, axis=0)


def random_scores(F: np.ndarray, y=None, task: Optional[str] = None) -> np.ndarray:
    return np.ones(F.shape[1], dtype=np.float64)


def available_selectors() -> list[str]:
    """List score modules shipped in :mod:`radr.scores`."""
    import pkgutil
    from . import scores

    return sorted(m.name for m in pkgutil.iter_modules(scores.__path__) if not m.name.startswith("_"))


def resolve_selector(selector: str | Callable) -> Callable:
    """Resolve a callable, bundled score filename, or ``module:function``."""
    import importlib

    if callable(selector):
        return selector
    if not isinstance(selector, str):
        raise TypeError("selector must be a string or callable.")
    aliases = {
        "distribution_variance": "dv",
        "radr": "dv",
        "label": "ka",
        "label_alignment": "ka",
        "supervised": "ka",
        "eerf": "ka",
        "ls": "leverage",
        "elss": "leverage",
    }
    key = aliases.get(selector.lower(), selector.lower())
    if ":" in selector:
        module, name = selector.split(":", 1)
    else:
        if key not in available_selectors():
            raise ValueError(f"Unknown selector {selector!r}; available: {available_selectors()}")
        module, name = f"radr.scores.{key}", "score"
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise TypeError(f"{module}:{name} must be callable.")
    return fn


def select_feature_indices(
    scores: np.ndarray,
    n_features: int,
    *,
    strategy: str = "topk",
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("scores must be a nonempty, finite one-dimensional array.")
    n_total = len(scores)
    if not isinstance(n_features, (int, np.integer)) or not 1 <= n_features <= n_total:
        raise ValueError("n_features must be an integer between 1 and the pool size.")
    n_keep = n_features
    strategy_key = str(strategy).lower()
    if strategy_key == "topk":
        keep = np.argsort(-scores, kind="stable")[:n_keep]
        return keep, None
    if strategy_key == "sample":
        if np.any(scores < 0):
            raise ValueError("Sampling requires nonnegative scores.")
        clipped = scores / max(float(scores.max()), 1.0)
        total = float(clipped.sum())
        probs = clipped / total if total > 0 else np.full(n_total, 1.0 / n_total)
        if np.count_nonzero(probs) < n_keep:
            raise ValueError("Too few positive scores for sampling without replacement; use topk.")
        rng = np.random.default_rng(random_state)
        keep = rng.choice(n_total, size=n_keep, replace=False, p=probs)
        keep = keep[np.argsort(probs[keep])[::-1]]
        weights = None  # Selected coordinates retain equal weights (paper Eq. 28).
        return keep, weights
    raise ValueError("selection_strategy must be 'topk' or 'sample'.")
