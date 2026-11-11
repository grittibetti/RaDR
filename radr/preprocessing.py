from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder

from sklearn.utils.validation import check_is_fitted
from .utils import ensure_bag_list


@dataclass
class BagDataset:
    bags: list[np.ndarray]
    targets: Optional[np.ndarray] = None
    bag_ids: Optional[list[Any]] = None
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.bags)

    @property
    def input_dim(self) -> int:
        return int(self.bags[0].shape[1]) if self.bags else 0

    @property
    def bag_sizes(self) -> np.ndarray:
        return np.asarray([bag.shape[0] for bag in self.bags], dtype=np.int64)

    def to_torch(self):
        try:
            import torch
            from torch.utils.data import Dataset
        except ImportError as exc:
            raise ImportError("Install radr[torch] to use BagDataset.to_torch().") from exc

        parent = self

        class TorchBagDataset(Dataset):
            def __len__(self):
                return len(parent.bags)

            def __getitem__(self, idx):
                y = None if parent.targets is None else parent.targets[idx]
                return {
                    "x": torch.as_tensor(parent.bags[idx], dtype=torch.float32),
                    "y": None if y is None else torch.as_tensor(y),
                    "bag_id": None if parent.bag_ids is None else parent.bag_ids[idx],
                }

        return TorchBagDataset()


def torch_collate_variable_bags(batch):
    """Pad variable-size bags into a PyTorch batch with a boolean mask."""
    try:
        import torch
        from torch.nn.utils.rnn import pad_sequence
    except ImportError as exc:
        raise ImportError("Install radr[torch] to use torch collate helpers.") from exc

    x_list = [item["x"] for item in batch]
    lengths = torch.as_tensor([x.shape[0] for x in x_list], dtype=torch.long)
    x_padded = pad_sequence(x_list, batch_first=True, padding_value=0.0)
    mask = torch.arange(x_padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    output = {"x": x_padded, "mask": mask, "length": lengths}
    if batch[0].get("y") is not None:
        output["y"] = torch.stack([item["y"] for item in batch])
    if batch[0].get("bag_id") is not None:
        output["bag_id"] = [item["bag_id"] for item in batch]
    return output


def torch_dataloader(dataset: BagDataset, batch_size: int = 32, shuffle: bool = True, **kwargs):
    """Build a PyTorch DataLoader for a BagDataset."""
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("Install radr[torch] to use torch_dataloader().") from exc
    return DataLoader(
        dataset.to_torch(),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=torch_collate_variable_bags,
        **kwargs,
    )


def _aggregate_target(values: np.ndarray, task: str, label_mode: str, aggregation: str):
    values = np.asarray(values)
    if label_mode == "bag":
        unique = pd.unique(values)
        if len(unique) != 1:
            raise ValueError(f"Expected one unique bag target, got {unique.tolist()}.")
        return unique[0]
    if aggregation == "first":
        return values[0]
    if task == "classification":
        if aggregation == "mode":
            uniq, counts = np.unique(values, return_counts=True)
            return uniq[np.argmax(counts)]
        numeric = values.astype(float)
        if aggregation == "majority":
            return int(np.mean(numeric) > 0.5)
        if aggregation == "any_positive":
            return int(np.any(numeric > 0))
        if aggregation == "all_positive":
            return int(np.all(numeric > 0))
        if aggregation == "proportion":
            return float(np.mean(numeric))
        raise ValueError(f"Unsupported classification aggregation {aggregation!r}.")
    numeric = values.astype(float)
    if aggregation == "mean":
        return float(np.mean(numeric))
    if aggregation == "median":
        return float(np.median(numeric))
    if aggregation == "max":
        return float(np.max(numeric))
    if aggregation == "min":
        return float(np.min(numeric))
    raise ValueError(f"Unsupported regression aggregation {aggregation!r}.")


def dataframe_to_bags(
    df: pd.DataFrame,
    *,
    bag_id_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    target_col: Optional[str] = None,
    task: str = "classification",
    label_mode: str = "bag",
    aggregation: str = "first",
    sort_cols: Optional[Sequence[str]] = None,
    bag_label_map: Optional[dict] = None,
    dropna: bool = True,
) -> BagDataset:
    if task not in {"classification", "regression"} or label_mode not in {"bag", "aggregate"}:
        raise ValueError("Invalid task or label_mode.")
    if not df.columns.is_unique:
        raise ValueError("DataFrame column names must be unique.")
    if df[bag_id_col].isna().any():
        raise ValueError("Bag IDs cannot be missing.")
    if target_col is not None and df[target_col].isna().any():
        raise ValueError("Targets cannot be missing.")
    work = df.copy()
    if feature_cols is None:
        excluded = {bag_id_col}
        if target_col is not None:
            excluded.add(target_col)
        if sort_cols is not None:
            excluded.update(sort_cols)
        feature_cols = [c for c in work.columns if c not in excluded]
    feature_cols = list(feature_cols)
    if (
        not feature_cols
        or bag_id_col in feature_cols
        or (target_col is not None and target_col in feature_cols)
    ):
        raise ValueError("Select nonempty feature columns excluding bag ID and target.")
    if sort_cols:
        work = work.sort_values(list(sort_cols)).reset_index(drop=True)
    if dropna:
        required = [bag_id_col] + feature_cols
        if target_col is not None:
            required.append(target_col)
        work = work.dropna(subset=required)

    bags: list[np.ndarray] = []
    targets: list[Any] = []
    bag_ids: list[Any] = []
    for bag_id, group in work.groupby(bag_id_col, sort=False, observed=True):
        bag = group.loc[:, feature_cols].to_numpy(dtype=np.float64)
        if bag.shape[0] == 0:
            continue
        if bag_label_map is not None:
            if bag_id not in bag_label_map:
                raise ValueError(f"Bag {bag_id!r} is missing from bag_label_map.")
            y_value = bag_label_map[bag_id]
        elif target_col is not None:
            y_value = _aggregate_target(
                group[target_col].to_numpy(),
                task=task,
                label_mode=label_mode,
                aggregation=aggregation,
            )
        else:
            y_value = None
        bags.append(bag)
        if y_value is not None:
            targets.append(y_value)
        bag_ids.append(bag_id)

    y = None
    if targets:
        y = np.asarray(targets, dtype=None if task == "classification" else np.float64)
    return BagDataset(
        bags=bags,
        targets=y,
        bag_ids=bag_ids,
        metadata={
            "task": task,
            "bag_id_col": bag_id_col,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "label_mode": label_mode,
            "aggregation": aggregation,
        },
    )


class BagPreprocessor(TransformerMixin, BaseEstimator):
    """Prepare unorganized set-valued data for RaDR estimators."""

    def __init__(
        self,
        bag_id_col: Optional[str] = None,
        feature_cols: Optional[Sequence[str]] = None,
        target_col: Optional[str] = None,
        task: str = "classification",
        label_mode: str = "bag",
        aggregation: str = "first",
        sort_cols: Optional[Sequence[str]] = None,
        bag_label_map: Optional[dict] = None,
        scaler: Optional[str] = "standard",
        missing: str = "drop",
        fill_value: float = 0.0,
        min_bag_size: int = 1,
        max_bag_size: Optional[int] = None,
        subsample: str = "first",
        random_state: Optional[int] = None,
        dtype: str = "float64",
        categorical_encoding: str = "onehot",
        categorical_cols: Optional[Sequence[str]] = None,
        handle_unknown: str = "ignore",
    ):
        self.bag_id_col = bag_id_col
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.task = task
        self.label_mode = label_mode
        self.aggregation = aggregation
        self.sort_cols = sort_cols
        self.bag_label_map = bag_label_map
        self.scaler = scaler
        self.missing = missing
        self.fill_value = fill_value
        self.min_bag_size = min_bag_size
        self.max_bag_size = max_bag_size
        self.subsample = subsample
        self.random_state = random_state
        self.dtype = dtype
        self.categorical_encoding = categorical_encoding
        self.categorical_cols = categorical_cols
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        if self.missing not in {"drop", "fill", "error"}:
            raise ValueError("missing must be 'drop', 'fill', or 'error'.")
        if self.min_bag_size < 1 or (self.max_bag_size is not None and self.max_bag_size < self.min_bag_size):
            raise ValueError("Require 1 <= min_bag_size <= max_bag_size.")
        if self.subsample not in {"first", "last", "random"}:
            raise ValueError("Invalid subsample strategy.")
        if self.categorical_encoding not in {"onehot", "ordinal", "numeric"}:
            raise ValueError("categorical_encoding must be 'onehot', 'ordinal', or 'numeric'.")
        if self.handle_unknown not in {"ignore", "error"}:
            raise ValueError("handle_unknown must be 'ignore' or 'error'.")
        if np.dtype(self.dtype).kind != "f" or not np.isfinite(self.fill_value):
            raise ValueError("Use a floating dtype and finite fill_value.")
        self.dataframe_input_ = isinstance(X, pd.DataFrame)
        dataset = self._coerce_to_dataset(X, y, fitting=True)
        points = np.concatenate(dataset.bags, axis=0)
        if self.scaler is None or str(self.scaler).lower() in {"none", "identity"}:
            self.scaler_ = None
        elif str(self.scaler).lower() == "standard":
            self.scaler_ = StandardScaler().fit(points)
        elif str(self.scaler).lower() == "minmax":
            self.scaler_ = MinMaxScaler().fit(points)
        else:
            raise ValueError("scaler must be None, 'standard', or 'minmax'.")
        self.metadata_ = dict(dataset.metadata)
        self.n_features_in_ = dataset.input_dim
        return self

    def transform(self, X, y=None, return_metadata: bool = False):
        check_is_fitted(self, "scaler_")
        if isinstance(X, pd.DataFrame) != self.dataframe_input_:
            raise TypeError("Use the same input container type as at fit.")
        dataset = self._coerce_to_dataset(X, y, fitting=False)
        if dataset.input_dim != self.n_features_in_:
            raise ValueError("Feature dimension differs from training data.")
        bags = [self._transform_one_bag(bag, i) for i, bag in enumerate(dataset.bags)]
        targets = dataset.targets
        metadata = {
            **dataset.metadata,
            "bag_ids": dataset.bag_ids,
            "preprocessor": self.get_params(deep=False),
        }
        if return_metadata:
            return bags, targets, metadata
        return bags if targets is None else (bags, targets)

    def fit_transform(self, X, y=None, return_metadata: bool = False, **fit_params):
        return self.fit(X, y).transform(X, y, return_metadata=return_metadata)

    def _coerce_to_dataset(self, X, y=None, fitting: bool = False) -> BagDataset:
        if hasattr(X, "to_numpy") and hasattr(X, "columns"):
            if self.bag_id_col is None:
                raise ValueError("bag_id_col is required for DataFrame inputs.")
            frame, encoded_cols = self._encode_frame(X, fitting)
            dataset = dataframe_to_bags(
                frame,
                bag_id_col=self.bag_id_col,
                feature_cols=encoded_cols,
                target_col=self.target_col if self.target_col in frame.columns else None,
                task=self.task,
                label_mode=self.label_mode,
                aggregation=self.aggregation,
                sort_cols=self.sort_cols,
                bag_label_map=self.bag_label_map,
                dropna=self.missing == "drop",
            )
        else:
            bags = ensure_bag_list(X, dtype=self.dtype, allow_nan=True)
            targets = None if y is None else np.asarray(y)
            dataset = BagDataset(bags=bags, targets=targets, metadata={"task": self.task})

        if dataset.targets is not None and len(dataset.targets) != len(dataset.bags):
            raise ValueError("Expected one target per bag.")
        bags = []
        targets = []
        bag_ids = [] if dataset.bag_ids is not None else None
        for i, bag in enumerate(dataset.bags):
            clean = self._handle_missing(np.asarray(bag, dtype=self.dtype))
            if clean.shape[0] < int(self.min_bag_size):
                continue
            clean = self._limit_bag_size(clean)
            bags.append(clean)
            if dataset.targets is not None:
                targets.append(dataset.targets[i])
            if bag_ids is not None:
                bag_ids.append(dataset.bag_ids[i])
        if not bags:
            raise ValueError("No valid bags remain after preprocessing.")
        target_array = None if dataset.targets is None else np.asarray(targets, dtype=dataset.targets.dtype)
        return BagDataset(bags=bags, targets=target_array, bag_ids=bag_ids, metadata=dict(dataset.metadata))

    def _encode_frame(self, X, fitting):
        if not X.columns.is_unique:
            raise ValueError("DataFrame column names must be unique.")
        if fitting:
            excluded = {self.bag_id_col, self.target_col, *(self.sort_cols or [])}
            self.feature_cols_ = (
                list(self.feature_cols)
                if self.feature_cols is not None
                else [c for c in X if c not in excluded]
            )
            if not self.feature_cols_ or any(
                c in {self.bag_id_col, self.target_col} for c in self.feature_cols_
            ):
                raise ValueError("Select features excluding bag ID and target.")
            self.categorical_cols_ = (
                list(self.categorical_cols)
                if self.categorical_cols is not None
                else [
                    c
                    for c in self.feature_cols_
                    if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])
                ]
            )
            if not set(self.categorical_cols_).issubset(self.feature_cols_):
                raise ValueError("categorical_cols must be selected feature columns.")
            self.numeric_cols_ = [c for c in self.feature_cols_ if c not in self.categorical_cols_]
        missing_cols = set(self.feature_cols_) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {sorted(missing_cols)}")
        frame = X.copy()
        if self.bag_id_col not in frame or frame[self.bag_id_col].isna().any():
            raise ValueError("Bag IDs must exist and cannot be missing.")
        if fitting and self.target_col is not None and self.target_col not in frame:
            raise ValueError("Training target column is missing.")
        if self.target_col in frame and frame[self.target_col].isna().any():
            raise ValueError("Targets cannot be missing.")
        if self.missing == "drop":
            frame = frame.dropna(subset=self.feature_cols_)
        elif self.missing == "error" and frame[self.feature_cols_].isna().any().any():
            raise ValueError("Missing feature values found.")
        if frame.empty:
            raise ValueError("No valid rows remain after preprocessing.")
        parts, names = [], []
        numeric_cols = self.feature_cols_ if self.categorical_encoding == "numeric" else self.numeric_cols_
        if numeric_cols:
            numeric = (
                frame[numeric_cols]
                .apply(pd.to_numeric, errors="raise")
                .to_numpy(dtype=float, na_value=np.nan)
            )
            if self.missing == "fill":
                numeric = np.where(np.isnan(numeric), self.fill_value, numeric)
            parts.append(numeric)
            names.extend(str(c) for c in numeric_cols)
        if self.categorical_cols_ and self.categorical_encoding != "numeric":
            # Prefix observed values to distinguish literal strings from the missing sentinel.
            cats = (
                frame[self.categorical_cols_]
                .apply(lambda col: col.map(lambda v: "missing:" if pd.isna(v) else "value:" + str(v)))
                .to_numpy(dtype=object)
            )
            if fitting:
                if self.categorical_encoding == "onehot":
                    self.encoder_ = OneHotEncoder(
                        sparse_output=False, handle_unknown=self.handle_unknown, dtype=self.dtype
                    )
                else:
                    self.encoder_ = OrdinalEncoder(
                        handle_unknown="error" if self.handle_unknown == "error" else "use_encoded_value",
                        unknown_value=None if self.handle_unknown == "error" else -1,
                        dtype=self.dtype,
                    )
                self.encoder_.fit(cats)
            parts.append(self.encoder_.transform(cats))
            names.extend(
                self.encoder_.get_feature_names_out([str(c) for c in self.categorical_cols_]).tolist()
            )
        values = np.column_stack(parts)
        if not np.isfinite(values).all():
            raise ValueError("Encoded features must be finite.")
        self.feature_names_out_ = np.asarray(names, dtype=object)
        cols = []
        for i in range(values.shape[1]):
            col = f"__radr_feature_{i}"
            while col in frame:
                col += "_"
            frame[col] = values[:, i]
            cols.append(col)
        return frame, cols

    def get_feature_names_out(self, input_features=None):
        """Return the encoded feature names learned from DataFrame inputs."""
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    def _handle_missing(self, bag: np.ndarray) -> np.ndarray:
        if np.isinf(bag).any():
            raise ValueError("Infinite feature values are not supported.")
        key = str(self.missing).lower()
        if key == "drop":
            return bag[~np.isnan(bag).any(axis=1)]
        if key == "fill":
            return np.nan_to_num(bag, nan=float(self.fill_value))
        if key == "error" and np.isnan(bag).any():
            raise ValueError("NaN values found in bag.")
        return bag

    def _limit_bag_size(self, bag: np.ndarray) -> np.ndarray:
        if self.max_bag_size is None or bag.shape[0] <= int(self.max_bag_size):
            return bag
        key = str(self.subsample).lower()
        n = int(self.max_bag_size)
        if key == "first":
            return bag[:n]
        if key == "last":
            return bag[-n:]
        if key == "random":
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(bag.shape[0], size=n, replace=False)
            return bag[np.sort(idx)]
        raise ValueError("subsample must be 'first', 'last', or 'random'.")

    def _transform_one_bag(self, bag: np.ndarray, index: int) -> np.ndarray:
        if self.scaler_ is not None:
            bag = self.scaler_.transform(bag)
        return np.asarray(bag, dtype=self.dtype)


def train_test_split_bags(
    bags,
    targets,
    *,
    bag_ids: Optional[Sequence[Any]] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: bool = False,
) -> tuple[BagDataset, BagDataset]:
    bags = ensure_bag_list(bags)
    y = None if targets is None else np.asarray(targets)
    ids = list(range(len(bags))) if bag_ids is None else list(bag_ids)
    if (y is not None and len(y) != len(bags)) or len(ids) != len(bags):
        raise ValueError("targets and bag_ids must match the number of bags.")
    stratify_y = y if stratify else None
    train_idx, test_idx = train_test_split(
        np.arange(len(bags)),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )

    def take(indices):
        indices = np.asarray(indices, dtype=int)
        return BagDataset(
            bags=[bags[i] for i in indices],
            targets=None if y is None else y[indices],
            bag_ids=[ids[i] for i in indices],
        )

    return take(train_idx), take(test_idx)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON.")


def save_npz(path, bags, targets=None, metadata=None, bag_ids=None):
    """Save numeric bags and JSON metadata without pickle/object arrays."""
    bags = ensure_bag_list(bags)
    if targets is not None and len(targets) != len(bags):
        raise ValueError("Expected one target per bag.")
    if bag_ids is not None and len(bag_ids) != len(bags):
        raise ValueError("Expected one ID per bag.")
    payload = json.dumps(
        {"targets": targets, "bag_ids": bag_ids, "metadata": metadata or {}}, default=_json_default
    )
    np.savez_compressed(
        path,
        values=np.concatenate(bags),
        offsets=np.r_[0, np.cumsum([len(b) for b in bags])],
        payload=np.asarray(payload),
    )


def load_npz(path) -> BagDataset:
    """Load numeric bag arrays and JSON metadata from a RaDR archive."""
    with np.load(path, allow_pickle=False) as data:
        values, offsets = data["values"], data["offsets"]
        if (
            offsets.ndim != 1
            or offsets.dtype.kind not in "iu"
            or len(offsets) < 2
            or offsets[0] != 0
            or offsets[-1] != len(values)
            or np.any(np.diff(offsets) <= 0)
        ):
            raise ValueError("Invalid bag offsets.")
        bags = ensure_bag_list([values[a:b] for a, b in zip(offsets[:-1], offsets[1:])])
        payload = json.loads(str(data["payload"].item()))
    targets = None if payload["targets"] is None else np.asarray(payload["targets"])
    ids = payload["bag_ids"]
    if (targets is not None and len(targets) != len(bags)) or (ids is not None and len(ids) != len(bags)):
        raise ValueError("Archive target/ID count differs from bag count.")
    return BagDataset(bags, targets, ids, payload["metadata"])
