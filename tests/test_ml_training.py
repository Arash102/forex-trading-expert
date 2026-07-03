from __future__ import annotations

import numpy as np
import pandas as pd

from debco.ml.calibration import expected_calibration_error, fit_probability_calibrator
from debco.ml.candidates import candidate_mask_for_job
from debco.ml.thresholding import threshold_sweep
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


def test_walk_forward_splits_are_causal_with_purge() -> None:
    folds = make_walk_forward_splits(
        100,
        train_window_bars=40,
        test_window_bars=10,
        step_bars=10,
        expanding=False,
        min_train_bars=30,
        purge_bars=5,
    )
    assert len(folds) > 0
    first = folds[0]
    assert first.train_idx.max() < first.test_idx.min()
    assert first.test_idx.min() - first.train_idx.max() == 6
    assert len(first.train_idx) == 35
    assert len(first.test_idx) == 10


def test_cpcv_splits_do_not_overlap_train_test_rows() -> None:
    dates = pd.date_range("2025-01-01", periods=60, freq="15min")
    meta = pd.DataFrame({"date": dates, "entry_date": dates, "exit_date": dates + pd.Timedelta(minutes=45)})
    folds = make_cpcv_splits(meta, n_groups=5, n_test_groups=2, embargo_bars=1, purge=True)
    assert len(folds) > 0
    for fold in folds:
        assert set(fold.train_idx).isdisjoint(set(fold.test_idx))
        assert len(fold.test_groups) == 2


def test_threshold_sweep_returns_signal_rates() -> None:
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.2, 0.6, 0.7, 0.9])
    out = threshold_sweep(y, proba, [0.5, 0.8], fold="x", probability_column="p")
    assert set(["threshold", "precision", "mcc", "signal_rate", "probability_column"]).issubset(out.columns)
    assert len(out) == 2
    assert out.loc[out["threshold"].eq(0.8), "signal_count"].iloc[0] == 1


def test_sigmoid_calibrator_and_ece_are_bounded() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.2, 0.35, 0.6, 0.8, 0.95])
    cal = fit_probability_calibrator(p, y, method="sigmoid")
    out = cal.predict(p)
    assert out.shape == p.shape
    assert np.all(out > 0)
    assert np.all(out < 1)
    ece = expected_calibration_error(y, out, bins=3)
    assert 0 <= ece <= 1


def test_candidate_filter_can_reduce_rows() -> None:
    x = pd.DataFrame({
        "session_block_id": [1, 3, 4, 6],
        "spread_pips": [1.0, 1.0, 2.0, 1.0],
        "gmma_distance": [0, 10, -30, 5],
        "atr_regime": [2, 2, 2, 5],
    })
    cfg = {
        "candidate_filter": {
            "enabled": True,
            "preset": "session_tradeable_v1",
            "session_block_ids": [3, 4, 6],
            "max_spread_pips_by_symbol": {"EURUSD": 1.5},
        }
    }
    mask = candidate_mask_for_job(x, symbol="EURUSD", side="long", config=cfg)
    assert mask.tolist() == [False, True, False, True]
