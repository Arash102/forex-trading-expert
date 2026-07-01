from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


@dataclass
class ClassificationReport:
    precision: float
    recall: float
    specificity: float
    f1: float
    roc_auc: float | None


def classification_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> ClassificationReport:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    auc = None
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass
    return ClassificationReport(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        specificity=float(specificity),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=auc,
    )


def sample_xgb_params(trial: Any, search_space: dict[str, list[float | int]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, bounds in search_space.items():
        lo, hi = bounds
        if isinstance(lo, int) and isinstance(hi, int):
            params[name] = trial.suggest_int(name, lo, hi)
        else:
            params[name] = trial.suggest_float(name, float(lo), float(hi), log=(float(lo) > 0 and float(hi) / float(lo) > 20))
    return params


def train_xgb_with_optuna_placeholder(df: pd.DataFrame, config: dict) -> None:
    """Placeholder entry point.

    The clean project starts by locking data/features/labels. The Optuna training loop
    will be filled after the feature ground truth is fully confirmed.
    """
    raise NotImplementedError("Optuna/XGBoost loop will be implemented after feature ground truth is locked.")
