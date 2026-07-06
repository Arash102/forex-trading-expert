from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from debco.live.dxy import build_dxy_from_component_closes, rates_to_ohlc_frame
from debco.live.model_registry import LiveModelRegistry
from debco.live.signal_engine import LiveSignalEngine


class ConstantProbabilityModel:
    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, x):
        n = len(x)
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


def test_build_dxy_from_component_closes():
    dates = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    frames = {}
    for symbol in ["EURUSD", "USDJPY", "GBPUSD", "USDCAD", "USDSEK", "USDCHF"]:
        frames[symbol] = pd.DataFrame({"date": dates, "close": [1.1, 1.11, 1.12, 1.13]})
    dxy = build_dxy_from_component_closes(frames)
    assert len(dxy) == 4
    assert {"dxy_close", "dxy_inverse_close", "index_close"}.issubset(dxy.columns)
    assert dxy["dxy_inverse_close"].notna().all()


def test_rates_to_ohlc_frame_from_records():
    rows = [
        {"time": 1776695400, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "tick_volume": 10, "spread": 1, "real_volume": 0},
    ]
    df = rates_to_ohlc_frame(rows, symbol="EURUSD")
    assert df.loc[0, "symbol"] == "EURUSD"
    assert str(df.loc[0, "date"].tz) == "UTC"


def test_live_model_registry_constant_artifact(tmp_path):
    import joblib

    setup_dir = tmp_path / "SETUP_A"
    setup_dir.mkdir(parents=True)
    joblib.dump(ConstantProbabilityModel(0.72), setup_dir / "model.joblib")
    artifact = {
        "setup_id": "SETUP_A",
        "symbol": "EURUSD",
        "side": "long",
        "probability_column": "y_prob_raw",
        "live_probability_cutoff": 0.5,
        "feature_columns": ["x1", "x2"],
        "model_file": "model.joblib",
        "calibrator_file": None,
    }
    (setup_dir / "artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
    reg = LiveModelRegistry(tmp_path)
    p, art = reg.predict_probability("SETUP_A", pd.DataFrame({"x1": [1], "x2": [2]}))
    assert p == 0.72
    assert art.live_probability_cutoff == 0.5


def test_signal_engine_inference_missing_model_is_safe():
    spec = {"selected_setups": [{"setup_id": "SETUP_A", "symbol": "EURUSD", "side": "long"}]}
    engine = LiveSignalEngine(spec, {"SETUP_A": 130001}, dry_run=True, inference_engine=None)
    decisions = engine.evaluate_closed_bar(
        symbol="EURUSD",
        timeframe="M15",
        signal_bar_time_utc="2026-01-01T00:00:00Z",
        decision_bar_time_utc="2026-01-01T00:15:00Z",
    )
    assert decisions[0].action == "no_signal"


def test_router_feature_config_loader_skips_list_placeholder(tmp_path, monkeypatch):
    from debco.live.router import ForwardDemoRouter

    cfg_path = tmp_path / "live_router.json"
    spec_path = tmp_path / "live_execution_spec.json"
    bad_feature_path = tmp_path / "configs" / "features_config.example.json"
    good_feature_path = tmp_path / "configs" / "features_config.local.json"
    bad_feature_path.parent.mkdir(parents=True)
    bad_feature_path.write_text(json.dumps(["not", "a", "feature", "config"]), encoding="utf-8")
    good_feature_path.write_text(json.dumps({"defaults": {}, "symbols": {}, "model_features": {}}), encoding="utf-8")

    spec = {"selected_setups": [{"setup_id": "EUR_AH_ATR2_BUY", "symbol": "EURUSD", "side": "long"}], "risk_per_trade": 0.01}
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    cfg = {
        "router_id": "test",
        "live_execution_spec_path": str(spec_path),
        "state_db_path": str(tmp_path / "state.sqlite"),
        "chart_event_dir": str(tmp_path / "events"),
        "symbols": ["EURUSD"],
        "setup_magic_numbers": {"EUR_AH_ATR2_BUY": 130103},
        "execution": {"dry_run": True},
        "inference": {
            "enabled": True,
            "ml_config_path": str(tmp_path / "missing_ml.json"),
            "feature_config_path": str(bad_feature_path),
            "live_models_dir": str(tmp_path / "models"),
        },
    }
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    router = ForwardDemoRouter(cfg_path)
    assert isinstance(router.feature_config, dict)
    assert "model_features" in router.feature_config
