"""Sklearn-style estimators for RaDR and adaptive RaDR."""

from .estimators import (
    ARaDRClassifier,
    ARaDRRegressor,
    RaDRClassifier,
    RaDRRegressor,
)
from .features import RandomFourierFeatures
from .preprocessing import (
    BagDataset,
    BagPreprocessor,
    dataframe_to_bags,
    load_npz,
    save_npz,
    torch_collate_variable_bags,
    torch_dataloader,
    train_test_split_bags,
)
from .selectors import (
    available_selectors,
    distribution_variance_scores,
    empirical_leverage_scores,
    label_alignment_scores,
)

__all__ = [
    "ARaDRClassifier",
    "ARaDRRegressor",
    "RaDRClassifier",
    "RaDRRegressor",
    "RandomFourierFeatures",
    "BagDataset",
    "BagPreprocessor",
    "dataframe_to_bags",
    "load_npz",
    "save_npz",
    "torch_collate_variable_bags",
    "torch_dataloader",
    "train_test_split_bags",
    "available_selectors",
    "distribution_variance_scores",
    "empirical_leverage_scores",
    "label_alignment_scores",
]

__version__ = "0.2.0"
