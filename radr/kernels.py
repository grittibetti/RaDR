from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .utils import pairwise_sqdist


def rbf_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None, gamma: float = 1.0) -> np.ndarray:
    return np.exp(-0.5 * float(gamma) * pairwise_sqdist(X, Y))


def linear_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    Y = X if Y is None else Y
    return np.asarray(X, dtype=np.float64) @ np.asarray(Y, dtype=np.float64).T


def resolve_kernel(kernel: str | Callable) -> Callable:
    if callable(kernel):
        return kernel
    key = str(kernel).lower()
    if key == "rbf":
        return rbf_kernel
    if key == "linear":
        return linear_kernel
    raise ValueError(f"Unsupported kernel {kernel!r}. Use 'rbf', 'linear', or a callable.")
