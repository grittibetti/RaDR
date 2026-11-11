from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils.validation import check_is_fitted
from .utils import as_random_state


FrequencySampler = Callable[[int, int, float, np.random.Generator], np.ndarray]


class RandomFourierFeatures(TransformerMixin, BaseEstimator):
    """Random Fourier features for instance-level shift-invariant kernels.

    The default Gaussian sampler draws frequencies from N(0, gamma I),
    approximating
    exp(-0.5 * gamma * ||x-y||^2).
    """

    def __init__(
        self,
        n_features: int = 256,
        gamma: float = 1.0,
        distribution: str | FrequencySampler = "gaussian",
        feature_map: str = "cosine",
        random_state: Optional[int] = None,
        dtype: str = "float64",
    ):
        self.n_features = n_features
        self.gamma = gamma
        self.distribution = distribution
        self.feature_map = feature_map
        self.random_state = random_state
        self.dtype = dtype

    def fit(self, X, y=None):
        if not isinstance(self.n_features, (int, np.integer)) or self.n_features < 1:
            raise ValueError("n_features must be a positive integer.")
        if not np.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("gamma must be finite and positive.")
        if np.dtype(self.dtype).kind != "f":
            raise ValueError("dtype must be floating point.")
        X = np.asarray(X, dtype=self.dtype)
        if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError(f"Expected 2D array, got shape {X.shape}.")
        self.n_input_features_ = int(X.shape[1])
        rng = as_random_state(self.random_state)
        self.random_weights_ = self._sample_frequencies(
            self.n_input_features_,
            int(self.n_features),
            float(self.gamma),
            rng,
        ).astype(self.dtype, copy=False)
        if not np.isfinite(self.random_weights_).all():
            raise ValueError("Sampled frequencies must be finite.")
        if self.feature_map == "cosine":
            self.random_offsets_ = rng.uniform(0.0, 2.0 * np.pi, size=int(self.n_features)).astype(
                self.dtype, copy=False
            )
        elif self.feature_map == "cos_sin":
            self.random_offsets_ = None
        else:
            raise ValueError("feature_map must be 'cosine' or 'cos_sin'.")
        return self

    def transform(self, X):
        check_is_fitted(self, "random_weights_")
        X = np.asarray(X, dtype=self.dtype)
        if X.ndim != 2 or X.shape[1] != self.n_input_features_ or not np.isfinite(X).all():
            raise ValueError("X must be finite and match the fitted feature dimension.")
        projection = X @ self.random_weights_
        if self.feature_map == "cosine":
            projection = projection + self.random_offsets_[None, :]
            return np.sqrt(2.0 / float(self.n_features)) * np.cos(projection)
        if self.feature_map == "cos_sin":
            scale = np.sqrt(1.0 / float(self.n_features))
            return scale * np.concatenate([np.cos(projection), np.sin(projection)], axis=1)
        raise ValueError("feature_map must be 'cosine' or 'cos_sin'.")

    def _sample_frequencies(
        self,
        input_dim: int,
        n_features: int,
        gamma: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if callable(self.distribution):
            weights = self.distribution(input_dim, n_features, gamma, rng)
            weights = np.asarray(weights, dtype=self.dtype)
            if weights.shape != (input_dim, n_features):
                raise ValueError(
                    "Custom random_feature_distribution must return shape "
                    f"{(input_dim, n_features)}, got {weights.shape}."
                )
            return weights

        key = str(self.distribution).lower()
        scale = np.sqrt(max(gamma, 1e-12))
        if key == "gaussian":
            return rng.normal(0.0, scale, size=(input_dim, n_features))
        if key == "laplace":
            return rng.laplace(0.0, scale / np.sqrt(2.0), size=(input_dim, n_features))
        if key == "cauchy":
            return rng.standard_cauchy(size=(input_dim, n_features)) * scale
        if key == "student_t":
            return rng.standard_t(df=3.0, size=(input_dim, n_features)) * scale / np.sqrt(3.0)
        raise ValueError(
            "distribution must be 'gaussian', 'laplace', 'cauchy', 'student_t', or a callable sampler."
        )


def compute_bag_embeddings(bags, random_features: RandomFourierFeatures) -> np.ndarray:
    embeddings = []
    for bag in bags:
        Z = random_features.transform(bag)
        embeddings.append(Z.mean(axis=0))
    return np.vstack(embeddings)
