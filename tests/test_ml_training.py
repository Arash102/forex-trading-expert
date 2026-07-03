from __future__ import annotations

import numpy as np
import pandas as pd

from debco.ml.xgb_optuna import binary_classification_metrics
from debco.validation.cpcv import make_cpcv_splits
from debco.validation.walk_forward import make_walk_forward_splits


def test_binary_metrics_include_mcc() -> None:
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = binary_classification_metrics(y, proba, threshold=0.5)
    assert "mcc" in metrics
    assert metrics["mcc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_walk_forward_splits_are_causal() -> None:
    folds = make_walk_forward_splits(
        100,
        train_window_bars=40,
        test_window_bars=10,
        step_bars=10,
        expanding=False,
        min_train_bars=40,
    )
    assert len(folds) > 0
    first = folds[0]
    assert first.train_idx.max() < first.test_idx.min()
    assert len(first.train_idx) == 40
    assert len(first.test_idx) == 10


def test_cpcv_splits_do_not_overlap_train_test_rows() -> None:
    dates = pd.date_range("2025-01-01", periods=60, freq="15min")
    meta = pd.DataFrame({
        "date": dates,
        "entry_date": dates,
        "exit_date": dates + pd.Timedelta(minutes=45),
    })
    folds = make_cpcv_splits(meta, n_groups=5, n_test_groups=2, embargo_bars=1, purge=True)
    assert len(folds) > 0
    for fold in folds:
        assert set(fold.train_idx).isdisjoint(set(fold.test_idx))
        assert len(fold.test_groups) == 2
