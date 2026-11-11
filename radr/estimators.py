from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from .features import RandomFourierFeatures, compute_bag_embeddings
from .kernels import resolve_kernel
from .metrics import classification_report_dict, regression_report_dict
from .selectors import resolve_selector, select_feature_indices
from .utils import (
    ensure_bag_list,
    median_heuristic_gamma,
    stack_points_from_bags,
)


def _cv_splitter(y, task: str, n_splits: int, random_state: Optional[int]):
    y = np.asarray(y)
    n_splits = max(2, min(int(n_splits), len(y)))
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        if len(counts) and int(counts.min()) >= 2:
            n_splits = max(2, min(n_splits, int(counts.min())))
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


class _BaseRaDR(BaseEstimator):
    _task = "classification"
    _adaptive = False

    def __init__(
        self,
        n_features: int = 256,
        rf_gamma: Optional[float | str] = "median",
        random_feature_distribution: str | Callable = "gaussian",
        feature_map: str = "cosine",
        pool_multiplier: float = 8.0,
        pool_size: Optional[int] = None,
        selector: str | Callable = "dv",
        selection_strategy: str = "auto",
        leverage_ridge: Optional[float] = None,
        downstream: str | BaseEstimator = "linear",
        downstream_kernel: str | Callable = "rbf",
        kernel_gamma: Optional[float | str] = "median",
        C: float = 1.0,
        alpha: float = 1e-3,
        C_grid: Optional[tuple[float, ...]] = (0.1, 1.0, 10.0, 100.0),
        alpha_grid: Optional[tuple[float, ...]] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
        kernel_gamma_grid: Optional[tuple[float, ...]] = (0.25, 0.5, 1.0, 2.0, 4.0),
        kernel_gamma_scale: float = 1.0,
        tune: bool = False,
        cv: int = 3,
        max_iter: int = 5000,
        probability: bool = False,
        random_state: Optional[int] = None,
        dtype: str = "float64",
    ):
        self.n_features = n_features
        self.rf_gamma = rf_gamma
        self.random_feature_distribution = random_feature_distribution
        self.feature_map = feature_map
        self.pool_multiplier = pool_multiplier
        self.pool_size = pool_size
        self.selector = selector
        self.selection_strategy = selection_strategy
        self.leverage_ridge = leverage_ridge
        self.downstream = downstream
        self.downstream_kernel = downstream_kernel
        self.kernel_gamma = kernel_gamma
        self.C = C
        self.alpha = alpha
        self.C_grid = C_grid
        self.alpha_grid = alpha_grid
        self.kernel_gamma_grid = kernel_gamma_grid
        self.kernel_gamma_scale = kernel_gamma_scale
        self.tune = tune
        self.cv = cv
        self.max_iter = max_iter
        self.probability = probability
        self.random_state = random_state
        self.dtype = dtype

    def fit(self, X, y):
        start = time.perf_counter()
        bags = ensure_bag_list(X, dtype=self.dtype)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional with one target per bag.")
        if self._task == "classification":
            check_classification_targets(y)
            if len(np.unique(y)) < 2:
                raise ValueError("Classification requires at least two classes.")
        elif not np.isfinite(y.astype(float)).all():
            raise ValueError("Regression targets must be finite.")
        if not isinstance(self.n_features, (int, np.integer)) or self.n_features < 1:
            raise ValueError("n_features must be a positive integer.")
        if self.feature_map == "cos_sin" and self._adaptive:
            raise ValueError("A-RaDR uses cosine features; cos_sin is supported only for RaDR.")
        if not np.isfinite(self.kernel_gamma_scale) or self.kernel_gamma_scale <= 0:
            raise ValueError("kernel_gamma_scale must be finite and positive.")
        if len(y) != len(bags):
            raise ValueError(f"Expected {len(bags)} targets, got {len(y)}.")

        if self.tune:
            return self._fit_with_search(bags, y)
        self.input_dim_ = int(bags[0].shape[1])
        self.n_features_in_ = self.input_dim_
        self.classes_ = np.unique(y) if self._task == "classification" else None
        self.rf_gamma_ = self._resolve_rf_gamma(bags)
        n_pool = self._resolve_pool_size()

        rf = RandomFourierFeatures(
            n_features=n_pool,
            gamma=self.rf_gamma_,
            distribution=self.random_feature_distribution,
            feature_map=self.feature_map,
            random_state=self.random_state,
            dtype=self.dtype,
        ).fit(np.zeros((1, self.input_dim_), dtype=self.dtype))
        F_pool = compute_bag_embeddings(bags, rf)

        selector_time = 0.0
        if self._adaptive:
            selector_start = time.perf_counter()
            selector_callable = resolve_selector(self.selector)
            if str(self.selector).lower() in {"leverage", "ls", "elss"}:
                scores = selector_callable(F_pool, y, self._task, ridge=self.leverage_ridge)
            else:
                scores = selector_callable(F_pool, y, self._task)
            scores = np.asarray(scores, dtype=float)
            if scores.shape != (F_pool.shape[1],) or not np.isfinite(scores).all():
                raise ValueError("Selector must return one finite score per candidate column.")
            strategy = self._resolve_selection_strategy()
            keep, weights = select_feature_indices(
                scores,
                int(self.n_features),
                strategy=strategy,
                random_state=self.random_state,
            )
            selector_time = time.perf_counter() - selector_start
            F = F_pool[:, keep]
            if weights is not None:
                F = F * weights[None, :]
            self.selector_ = {
                "name": self.selector,
                "strategy": strategy,
                "scores": scores,
                "keep": keep,
                "feature_weights": weights,
                "score_mean": float(np.mean(scores)),
                "score_max": float(np.max(scores)),
                "score_min": float(np.min(scores)),
            }
        else:
            keep = np.arange(F_pool.shape[1])
            F = F_pool[:, keep]
            self.selector_ = None

        self.feature_scale_ = np.sqrt(n_pool / self.n_features) if self._adaptive else 1.0
        F = F * self.feature_scale_
        self.random_features_ = rf
        self.feature_indices_ = keep
        self.train_embeddings_ = F
        self.y_train_ = y
        self.kernel_gamma_ = self._resolve_kernel_gamma(F)
        self.downstream_estimator_ = self._fit_downstream(F, y)
        self.fit_time_seconds_ = float(time.perf_counter() - start)
        self.params_ = self._build_params()
        self.fit_summary_ = {
            "n_train_bags": int(len(bags)),
            "input_dim": int(self.input_dim_),
            "n_features": int(F.shape[1]),
            "pool_size": int(n_pool),
            "adaptive": bool(self._adaptive),
            "selector_time_seconds": float(selector_time),
            "fit_time_seconds": float(self.fit_time_seconds_),
        }
        return self

    def transform(self, X):
        check_is_fitted(self, "random_features_")
        bags = ensure_bag_list(X, dtype=self.dtype)
        F = compute_bag_embeddings(bags, self.random_features_)
        F = F[:, self.feature_indices_]
        if self.selector_ is not None and self.selector_.get("feature_weights") is not None:
            F = F * self.selector_["feature_weights"][None, :]
        return F * self.feature_scale_

    def predict(self, X):
        check_is_fitted(self, "downstream_estimator_")
        F = self.transform(X)
        if self._uses_kernel_downstream():
            K = self._kernel(F, self.train_embeddings_)
            return self.downstream_estimator_.predict(K)
        return self.downstream_estimator_.predict(F)

    def decision_function(self, X):
        check_is_fitted(self, "downstream_estimator_")
        F = self.transform(X)
        X_model = self._kernel(F, self.train_embeddings_) if self._uses_kernel_downstream() else F
        if not hasattr(self.downstream_estimator_, "decision_function"):
            raise AttributeError("The downstream estimator has no decision_function.")
        return self.downstream_estimator_.decision_function(X_model)

    def predict_proba(self, X):
        check_is_fitted(self, "downstream_estimator_")
        F = self.transform(X)
        X_model = self._kernel(F, self.train_embeddings_) if self._uses_kernel_downstream() else F
        if not hasattr(self.downstream_estimator_, "predict_proba"):
            raise AttributeError(
                "The downstream estimator has no predict_proba. Set probability=True "
                "for the default kernel classifier or provide a probabilistic estimator."
            )
        return self.downstream_estimator_.predict_proba(X_model)

    def test(self, X, y) -> dict:
        pred = self.predict(X)
        if self._task == "classification":
            return classification_report_dict(y, pred)
        return regression_report_dict(y, pred)

    def score(self, X, y):
        pred = self.predict(X)
        if self._task == "classification":
            return float(accuracy_score(y, pred))
        return float(r2_score(y, pred))

    def _resolve_rf_gamma(self, bags) -> float:
        if self.rf_gamma is None or str(self.rf_gamma).lower() == "median":
            points = stack_points_from_bags(bags, max_points=2048, random_state=self.random_state)
            return median_heuristic_gamma(points, fallback=1.0, random_state=self.random_state)
        value = float(self.rf_gamma)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("rf_gamma must be finite and positive.")
        return value

    def _resolve_pool_size(self) -> int:
        if self._adaptive:
            if self.pool_size is not None:
                if not isinstance(self.pool_size, (int, np.integer)) or self.pool_size < self.n_features:
                    raise ValueError("pool_size must be an integer >= n_features.")
                return int(self.pool_size)
            if not np.isfinite(self.pool_multiplier) or self.pool_multiplier < 1:
                raise ValueError("pool_multiplier must be finite and >= 1.")
            return max(
                int(self.n_features), int(np.ceil(float(self.n_features) * float(self.pool_multiplier)))
            )
        return int(self.n_features)

    def _resolve_selection_strategy(self) -> str:
        key = str(self.selection_strategy).lower()
        if key == "auto":
            selector_key = str(self.selector).lower() if not callable(self.selector) else ""
            return "sample" if selector_key == "random" else "topk"
        if key in {"topk", "sample"}:
            return key
        raise ValueError("selection_strategy must be 'auto', 'topk', or 'sample'.")

    def _resolve_kernel_gamma(self, F: np.ndarray) -> Optional[float]:
        if not self._uses_kernel_downstream():
            return None
        if self.kernel_gamma is None or str(self.kernel_gamma).lower() == "median":
            return self.kernel_gamma_scale * median_heuristic_gamma(
                F, fallback=1.0, random_state=self.random_state
            )
        value = float(self.kernel_gamma) * self.kernel_gamma_scale
        if not np.isfinite(value) or value <= 0:
            raise ValueError("kernel_gamma must be finite and positive.")
        return value

    def _uses_kernel_downstream(self) -> bool:
        if isinstance(self.downstream, str):
            return self.downstream.lower() == "kernel"
        return False

    def _kernel(self, X, Y=None):
        kernel = resolve_kernel(self.downstream_kernel)
        if str(self.downstream_kernel).lower() == "rbf":
            return kernel(X, Y, gamma=float(self.kernel_gamma_))
        return kernel(X, Y)

    def _fit_downstream(self, F: np.ndarray, y: np.ndarray):
        if not isinstance(self.downstream, str):
            model = clone(self.downstream)
            model.fit(F, y)
            self.downstream_params_ = {"custom_estimator": model.__class__.__name__}
            return model

        key = self.downstream.lower()
        if key == "linear":
            return self._fit_linear(F, y)
        if key == "kernel":
            return self._fit_kernel(F, y)
        raise ValueError("downstream must be 'linear', 'kernel', or a sklearn estimator.")

    def _fit_linear(self, F, y):
        if self._task == "classification":
            self.downstream_params_ = {"C": float(self.C)}
            return LinearSVC(
                C=self.C, dual=False, max_iter=self.max_iter, random_state=self.random_state
            ).fit(F, y)
        self.downstream_params_ = {"alpha": float(self.alpha)}
        return Ridge(alpha=self.alpha, fit_intercept=False).fit(F, y.astype(float))

    def _fit_kernel(self, F, y):
        K = self._kernel(F, F)
        if self._task == "classification":
            self.downstream_params_ = {"C": float(self.C), "kernel_gamma": self.kernel_gamma_}
            return SVC(
                kernel="precomputed",
                C=self.C,
                **({"probability": True} if self.probability else {}),
                random_state=self.random_state,
            ).fit(K, y)
        self.downstream_params_ = {"alpha": float(self.alpha), "kernel_gamma": self.kernel_gamma_}
        return KernelRidge(alpha=self.alpha, kernel="precomputed").fit(K, y.astype(float))

    def _fit_with_search(self, bags, y):
        if not isinstance(self.cv, (int, np.integer)) or self.cv < 2 or len(y) < 2:
            raise ValueError("tune=True requires cv >= 2 and at least two bags.")
        if self._task == "classification" and np.unique(y, return_counts=True)[1].min() < 2:
            raise ValueError("Tuning requires at least two bags in every class.")
        grid = {}
        if isinstance(self.downstream, str):
            key = "C" if self._task == "classification" else "alpha"
            grid[key] = list((self.C_grid if key == "C" else self.alpha_grid) or [getattr(self, key)])
            if self._uses_kernel_downstream() and self.downstream_kernel == "rbf":
                grid["kernel_gamma_scale"] = list(self.kernel_gamma_grid or [self.kernel_gamma_scale])
        search = GridSearchCV(
            clone(self).set_params(tune=False),
            grid,
            scoring="accuracy" if self._task == "classification" else "neg_mean_squared_error",
            cv=_cv_splitter(y, self._task, self.cv, self.random_state),
            error_score="raise",
        ).fit(bags, y)
        for name, value in vars(search.best_estimator_).items():
            if name.endswith("_"):
                setattr(self, name, value)
        self.best_params_ = search.best_params_
        self.cv_results_ = search.cv_results_
        return self

    def _build_params(self) -> dict:
        params = {
            "task": self._task,
            "adaptive": bool(self._adaptive),
            "n_features": int(len(self.feature_indices_)),
            "pool_size": int(self.random_features_.n_features),
            "rf_gamma": float(self.rf_gamma_),
            "random_feature_distribution": self.random_feature_distribution,
            "feature_map": self.feature_map,
            "downstream": self.downstream
            if isinstance(self.downstream, str)
            else self.downstream.__class__.__name__,
            **getattr(self, "downstream_params_", {}),
        }
        if self.selector_ is not None:
            params.update(
                {
                    "selector": self.selector,
                    "selection_strategy": self.selector_["strategy"],
                    "score_mean": self.selector_["score_mean"],
                    "score_max": self.selector_["score_max"],
                    "score_min": self.selector_["score_min"],
                }
            )
        return params


class RaDRClassifier(ClassifierMixin, _BaseRaDR):
    """Random Feature Distribution Regression classifier."""

    _task = "classification"
    _adaptive = False


class RaDRRegressor(RegressorMixin, _BaseRaDR):
    """Random Feature Distribution Regression regressor."""

    _task = "regression"
    _adaptive = False


class ARaDRClassifier(ClassifierMixin, _BaseRaDR):
    """Adaptive RaDR classifier with random-feature pool selection."""

    _task = "classification"
    _adaptive = True


class ARaDRRegressor(RegressorMixin, _BaseRaDR):
    """Adaptive RaDR regressor with random-feature pool selection."""

    _task = "regression"
    _adaptive = True
