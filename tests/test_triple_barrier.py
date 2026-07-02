from __future__ import annotations

import pandas as pd

from debco.labels.triple_barrier import (
    TripleBarrierParams,
    build_triple_barrier_labels,
    build_ml_ready_dataset,
    iter_label_jobs,
    make_params_from_config,
)


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=6, freq="15min"),
        "open": [1.1000, 1.1000, 1.1015, 1.1020, 1.1000, 1.0990],
        "high": [1.1005, 1.1010, 1.1032, 1.1025, 1.1002, 1.0995],
        "low": [1.0995, 1.0998, 1.1010, 1.0985, 1.0980, 1.0985],
        "close": [1.1000, 1.1008, 1.1028, 1.0990, 1.0988, 1.0992],
    })


def test_long_tp_binary_label_with_next_bar_entry() -> None:
    params = TripleBarrierParams(
        symbol="EURUSD",
        profile="runner_20_10_h24",
        side="long",
        pip_size=0.0001,
        tp_pips=20,
        sl_pips=10,
        max_horizon_bars=3,
        entry_offset_bars=1,
        entry_price_column="open",
        same_bar_policy="sl_first",
    )
    labels = build_triple_barrier_labels(_df(), params)
    first = labels.iloc[0]
    assert first["profile"] == "runner_20_10_h24"
    assert first["target_name"] == "long_target"
    assert first["entry_price"] == 1.1000
    assert first["label"] == 1
    assert first["outcome_label"] == 1
    assert first["event_type"] == "tp"
    assert first["bars_to_event"] == 2


def test_same_bar_sl_first_is_binary_zero_and_conservative() -> None:
    data = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=3, freq="15min"),
        "open": [1.1000, 1.1000, 1.1000],
        "high": [1.1000, 1.1025, 1.1000],
        "low": [1.1000, 1.0985, 1.1000],
        "close": [1.1000, 1.1000, 1.1000],
    })
    params = TripleBarrierParams(
        symbol="EURUSD",
        profile="runner_20_10_h24",
        side="long",
        pip_size=0.0001,
        tp_pips=20,
        sl_pips=10,
        max_horizon_bars=1,
        entry_offset_bars=1,
        entry_price_column="open",
        same_bar_policy="sl_first",
    )
    labels = build_triple_barrier_labels(data, params)
    assert labels.iloc[0]["label"] == 0
    assert labels.iloc[0]["outcome_label"] == -1
    assert labels.iloc[0]["event_type"] == "sl_same_bar"
    assert labels.iloc[0]["same_bar_hit"] == 1


def test_short_tp_binary_label() -> None:
    params = TripleBarrierParams(
        symbol="EURUSD",
        profile="fast_15_8_h16",
        side="short",
        pip_size=0.0001,
        tp_pips=15,
        sl_pips=10,
        max_horizon_bars=3,
        entry_offset_bars=1,
        entry_price_column="open",
        same_bar_policy="sl_first",
    )
    labels = build_triple_barrier_labels(_df(), params)
    row = labels.iloc[2]
    assert row["target_name"] == "short_target"
    assert row["label"] == 1
    assert row["outcome_label"] == 1
    assert row["event_type"] == "tp"


def test_config_profiles_create_eight_jobs_and_params() -> None:
    cfg = {
        "labeling": {"build_profiles": "all", "entry_offset_bars": 1},
        "symbols": {
            "EURUSD": {
                "enabled": True,
                "pip_size": 0.0001,
                "sides": ["long", "short"],
                "primary_profile": "runner_20_10_h24",
                "profiles": {
                    "runner_20_10_h24": {"tp_pips": 20, "sl_pips": 10, "max_horizon_bars": 24},
                    "fast_15_8_h16": {"tp_pips": 15, "sl_pips": 8, "max_horizon_bars": 16},
                },
            },
            "XAUUSD": {
                "enabled": True,
                "pip_size": 0.01,
                "sides": ["long", "short"],
                "primary_profile": "runner_2200_1100_h40",
                "profiles": {
                    "runner_2200_1100_h40": {"tp_pips": 2200, "sl_pips": 1100, "max_horizon_bars": 40},
                    "active_1500_800_h32": {"tp_pips": 1500, "sl_pips": 800, "max_horizon_bars": 32},
                },
            },
        },
    }
    jobs = iter_label_jobs(cfg)
    assert len(jobs) == 8
    params = make_params_from_config(cfg, "XAUUSD", "short", "active_1500_800_h32")
    assert params.pip_size == 0.01
    assert params.tp_pips == 1500
    assert params.sl_pips == 800
    assert params.max_horizon_bars == 32
    assert params.profile == "active_1500_800_h32"


def test_ml_ready_dataset_contains_only_features_and_label() -> None:
    features = _df().copy()
    features["symbol"] = "EURUSD"
    features["tick_volume"] = 1
    features["spread"] = 10
    features["real_volume"] = 0
    features["dxy_close"] = 101.0
    features["dxy_inverse_close"] = 0.99
    features["index_close"] = 0.99
    features["rsi"] = [50, 51, 52, 53, 54, 55]
    features["gmma_distance"] = [0, 1, 2, 3, 4, 5]

    params = TripleBarrierParams(
        symbol="EURUSD",
        profile="runner_20_10_h24",
        side="long",
        pip_size=0.0001,
        tp_pips=20,
        sl_pips=10,
        max_horizon_bars=3,
        entry_offset_bars=1,
        entry_price_column="open",
        same_bar_policy="sl_first",
    )
    labels = build_triple_barrier_labels(features, params)
    ml_ready = build_ml_ready_dataset(features, labels)
    assert "label" in ml_ready.columns
    assert "date" not in ml_ready.columns
    assert "open" not in ml_ready.columns
    assert "event_type" not in ml_ready.columns
    assert "entry_price" not in ml_ready.columns
    assert set(ml_ready.columns) == {"rsi", "gmma_distance", "label"}
