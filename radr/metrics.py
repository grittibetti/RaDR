from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


def classification_report_dict(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true.reshape(-1), y_pred.reshape(-1)]))
    average = "binary" if len(labels) <= 2 else "macro"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, pos_label=labels[-1], zero_division=0)),
        "f1_average": average,
    }


def regression_report_dict(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }
