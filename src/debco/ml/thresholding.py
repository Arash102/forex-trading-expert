from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(fn, default: float = np.nan) -> float:
    try:
        value = float(fn())
        return value if np.isfinite(value) else default
    except Exception:
        return default


def binary_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": float(len(y)),
        "positive_rate": float(np.mean(y == 1)) if len(y) else np.nan,
        "predicted_positive_rate": float(np.mean(pred == 1)) if len(pred) else np.nan,
        "threshold": float(threshold),
        "accuracy": _safe_metric(lambda: accuracy_score(y, pred)),
        "precision": _safe_metric(lambda: precision_score(y, pred, zero_division=0)),
        "recall": _safe_metric(lambda: recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "f1": _safe_metric(lambda: f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": _safe_metric(lambda: balanced_accuracy_score(y, pred)),
        "mcc": _safe_metric(lambda: matthews_corrcoef(y, pred)),
        "roc_auc": _safe_metric(lambda: roc_auc_score(y, p)),
        "average_precision": _safe_metric(lambda: average_precision_score(y, p)),
        "log_loss": _safe_metric(lambda: log_loss(y, np.clip(p, 1e-6, 1-1e-6), labels=[0, 1])),
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
    }


def threshold_sweep(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Iterable[float],
    *,
    fold: str | None = None,
    probability_column: str = "y_prob",
) -> pd.DataFrame:
    rows = []
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    for thr in thresholds:
        m = binary_metrics(y, p, float(thr))
        m["fold"] = fold if fold is not None else "all"
        m["probability_column"] = probability_column
        m["signal_count"] = int(np.sum(p >= float(thr)))
        m["signal_rate"] = float(np.mean(p >= float(thr))) if len(p) else float("nan")
        rows.append(m)
    return pd.DataFrame(rows)


def threshold_sweep_from_predictions(
    predictions: pd.DataFrame,
    *,
    thresholds: Iterable[float],
    probability_columns: list[str],
    y_col: str = "y_true",
) -> pd.DataFrame:
    parts = []
    for col in probability_columns:
        if col not in predictions.columns:
            continue
        for fold, sub in predictions.groupby("fold", sort=False):
            parts.append(threshold_sweep(sub[y_col].to_numpy(), sub[col].to_numpy(), thresholds, fold=str(fold), probability_column=col))
        parts.append(threshold_sweep(predictions[y_col].to_numpy(), predictions[col].to_numpy(), thresholds, fold="ALL", probability_column=col))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
