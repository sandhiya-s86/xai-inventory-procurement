"""Evaluation metrics shared by every model so comparisons are apples-to-apples."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, MSE, RMSE, R2 and RMSLE.

    RMSLE is only defined for non-negative values. Demand predictions can go
    slightly negative under a linear model, so predictions are clipped at 0
    for the RMSLE term only -- the other metrics see the raw prediction.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    yt = np.clip(y_true, 0, None)
    yp = np.clip(y_pred, 0, None)
    rmsle = float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt)) ** 2)))

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "RMSLE": rmsle}


def classification_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    """Accuracy, precision, recall, F1 and (when scores are given) ROC-AUC.

    zero_division=0 keeps the table readable when a model predicts a single
    class -- that is a real result worth reporting, not an error to hide.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_score, dtype=float)))
    else:
        out["roc_auc"] = float("nan")

    return out


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Quantile (pinball) loss -- the correct scoring rule for quantile models."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    delta = y_true - y_pred
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1.0) * delta)))
