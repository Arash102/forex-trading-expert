from __future__ import annotations

import numpy as np
import pandas as pd

from debco.features.feature_builder import build_feature_outputs


def make_df(n: int = 600) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    close = pd.Series(1.10 + np.sin(np.arange(n) / 20) * 0.001 + np.arange(n) * 0.000001, dtype="float64")
    return pd.DataFrame({
        "date": dates,
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + 0.0003,
        "low": close - 0.0003,
        "close": close,
        "tick_volume": 100,
        "spread": 10,
        "real_volume": 0,
        "dxy_close": 101.0 + np.sin(np.arange(n) / 30) * 0.1,
        "dxy_inverse_close": 100.0 / (101.0 + np.sin(np.arange(n) / 30) * 0.1),
        "index_close": 100.0 / (101.0 + np.sin(np.arange(n) / 30) * 0.1),
    })


def test_build_symbol_aware_feature_outputs():
    cfg = {
        "defaults": {},
        "sessions": {
            "blocks": [
                {"name": "asia_pre_london", "id": 1, "start": "00:00", "end": "07:00"},
                {"name": "london_open", "id": 2, "start": "07:00", "end": "09:00"},
                {"name": "london_mid", "id": 3, "start": "09:00", "end": "12:00"},
                {"name": "overlap_early", "id": 4, "start": "12:00", "end": "14:00"},
                {"name": "overlap_late", "id": 5, "start": "14:00", "end": "16:00"},
                {"name": "ny_late", "id": 6, "start": "16:00", "end": "21:00"},
            ]
        },
        "symbols": {"EURUSD": {"pip_size": 0.0001, "point_size": 0.00001, "zigzag_deviation_pct": 0.13}},
        "model_features": {
            "max_features_with_lags": 10,
            "base_features": ["rsi", "gmma_distance", "market_x", "session_block_id"],
            "lags": {"rsi": [1], "gmma_distance": [1]},
        },
    }
    full, model = build_feature_outputs(make_df(), symbol="EURUSD", config=cfg)
    for c in ["spread_pips", "market_x", "market_y", "atr_14_pips", "session_block_id", "current_session_range_so_far_pips"]:
        assert c in full.columns
    for c in ["rsi_lag1", "gmma_distance_lag1"]:
        assert c in model.columns
    assert len(full) == len(model)
