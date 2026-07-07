from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from debco.features.feature_builder import build_feature_outputs
from .dxy import merge_dxy_into_symbol_frame, rates_to_ohlc_frame


@dataclass(frozen=True)
class LiveFeatureSnapshot:
    symbol: str
    signal_bar_time_utc: str
    full_frame_tail_rows: int
    model_feature_row: pd.DataFrame
    raw_row: pd.Series
    missing_feature_columns: list[str]
    candidate_feature_row: pd.DataFrame | None = None


def _normalize_utc_iso(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_feature_config_for_live(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize feature config shapes commonly seen in local workspaces.

    The historical feature builder expects a dict with:
    defaults, symbols, sessions={timezone, blocks}, and model_features={...}.
    Some local/live configs may contain sessions or model_features as raw lists.
    Normalize those before calling the research feature builder so live inference
    fails with clear errors instead of opaque list.get AttributeErrors.
    """
    if not isinstance(config, Mapping):
        raise TypeError(f"Feature config must be a JSON object/dict, got {type(config).__name__}.")
    cfg: dict[str, Any] = dict(config)

    # Allow a wrapper object such as {"features": {...}} or
    # {"feature_engineering": {...}} if it contains the actual feature config.
    for key in ("features", "feature_config", "feature_engineering", "feature_builder"):
        nested = cfg.get(key)
        if isinstance(nested, Mapping) and ("symbols" in nested or "defaults" in nested or "model_features" in nested):
            cfg = dict(nested)
            break

    sessions = cfg.get("sessions", {})
    if isinstance(sessions, list):
        cfg["sessions"] = {
            "timezone": str(cfg.get("timezone", cfg.get("default_timezone", "UTC"))),
            "blocks": sessions,
            "build_named_session_features": True,
            "build_current_session_features": True,
        }
    elif isinstance(sessions, Mapping):
        sess = dict(sessions)
        if "blocks" not in sess:
            # If a dict-like session config exists without blocks, keep it valid.
            sess.setdefault("blocks", [])
        sess.setdefault("timezone", str(cfg.get("timezone", "UTC")))
        cfg["sessions"] = sess
    else:
        cfg["sessions"] = {"timezone": "UTC", "blocks": []}

    model_features = cfg.get("model_features", {})
    if isinstance(model_features, list):
        cfg["model_features"] = {
            "base_features": model_features,
            "lags": {},
            "max_features_with_lags": max(100, len(model_features)),
        }
    elif isinstance(model_features, Mapping):
        cfg["model_features"] = dict(model_features)
    else:
        cfg["model_features"] = {"base_features": [], "lags": {}, "max_features_with_lags": 100}

    if not isinstance(cfg.get("symbols"), Mapping):
        raise TypeError("Feature config must contain a 'symbols' object with EURUSD/XAUUSD settings.")
    if not isinstance(cfg.get("defaults", {}), Mapping):
        cfg["defaults"] = {}
    return cfg


def build_live_feature_snapshot(
    *,
    symbol: str,
    rates: Any,
    feature_config: Mapping[str, Any],
    signal_bar_time_utc: str,
    dxy_frame: pd.DataFrame | None = None,
    required_feature_columns: list[str] | None = None,
) -> LiveFeatureSnapshot:
    """Build the latest closed-bar model feature row for live inference.

    The router passes MT5 rates that include the current open bar. This function
    explicitly keeps rows up to ``signal_bar_time_utc`` so inference remains
    candle-close based and matches the historical research design.
    """
    symbol = str(symbol).upper()
    feature_config = normalize_feature_config_for_live(feature_config)
    raw = rates_to_ohlc_frame(rates, symbol=symbol)
    if raw.empty:
        raise ValueError(f"No MT5 rates available for {symbol}.")
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    signal_time = pd.Timestamp(signal_bar_time_utc)
    if signal_time.tzinfo is None:
        signal_time = signal_time.tz_localize("UTC")
    else:
        signal_time = signal_time.tz_convert("UTC")
    raw = raw.loc[raw["date"] <= signal_time].copy()
    if raw.empty:
        raise ValueError(f"No closed bars at or before {signal_bar_time_utc} for {symbol}.")
    raw = merge_dxy_into_symbol_frame(raw, dxy_frame)
    full, model = build_feature_outputs(raw, symbol=symbol, config=feature_config)
    if "date" in model.columns:
        model_dates = pd.to_datetime(model["date"], utc=True, errors="coerce")
        candidates = model.loc[model_dates <= signal_time].copy()
    else:
        candidates = model.copy()
    if candidates.empty:
        raise ValueError(f"No model feature row generated for {symbol} at {signal_bar_time_utc}.")
    candidate_row = candidates.tail(1).copy()
    row = candidate_row.copy()
    missing: list[str] = []
    if required_feature_columns:
        for c in required_feature_columns:
            if c not in row.columns:
                row[c] = pd.NA
                missing.append(c)
        row = row[required_feature_columns]
    return LiveFeatureSnapshot(
        symbol=symbol,
        signal_bar_time_utc=_normalize_utc_iso(signal_time),
        full_frame_tail_rows=int(len(full)),
        model_feature_row=row.reset_index(drop=True),
        raw_row=raw.tail(1).iloc[0],
        missing_feature_columns=missing,
        candidate_feature_row=candidate_row.reset_index(drop=True),
    )
