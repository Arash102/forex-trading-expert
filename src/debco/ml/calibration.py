from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class ProbabilityCalibrator:
    method: str = "none"
    model: Any | None = None
    fitted: bool = False
    reason: str = "not_fitted"

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)
        if not self.fitted or self.model is None or self.method == "none":
            return np.clip(p, 1e-6, 1.0 - 1e-6)
        if self.method == "sigmoid":
            out = self.model.predict_proba(p.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            out = self.model.predict(p)
        else:
            out = p
        return np.clip(np.asarray(out, dtype=float), 1e-6, 1.0 - 1e-6)


def fit_probability_calibrator(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    method: str = "sigmoid",
    fallback_to_raw_if_single_class: bool = True,
    random_seed: int = 42,
) -> ProbabilityCalibrator:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y_true, dtype=int)
    method_l = str(method).lower()
    if method_l in {"none", "raw", "identity", "disabled"}:
        return ProbabilityCalibrator(method="none", model=None, fitted=False, reason="disabled")
    finite = np.isfinite(p)
    p = p[finite]
    y = y[finite]
    if len(p) < 10:
        return ProbabilityCalibrator(method="none", model=None, fitted=False, reason="too_few_calibration_rows")
    if len(np.unique(y)) < 2:
        if fallback_to_raw_if_single_class:
            return ProbabilityCalibrator(method="none", model=None, fitted=False, reason="single_class_calibration")
        raise ValueError("Calibration labels contain only one class.")
    if method_l == "sigmoid":
        model = LogisticRegression(random_state=int(random_seed), solver="lbfgs")
        model.fit(p.reshape(-1, 1), y)
        return ProbabilityCalibrator(method="sigmoid", model=model, fitted=True, reason="ok")
    if method_l == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(p, y)
        return ProbabilityCalibrator(method="isotonic", model=model, fitted=True, reason="ok")
    raise ValueError(f"Unsupported calibration method: {method}")


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    finite = np.isfinite(p)
    y = y[finite]
    p = p[finite]
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y[mask] == 1))
        conf = float(np.mean(p[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return float(ece)


def calibration_metrics(y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 10) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    out = {
        "brier_score": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "ece": expected_calibration_error(y, p, bins=bins),
    }
    try:
        out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    except Exception:
        out["log_loss"] = float("nan")
    return out
