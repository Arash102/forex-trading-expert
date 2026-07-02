from __future__ import annotations

"""Profile-aware binary trade-outcome Triple Barrier labels.

Design decisions for the DebCo research engine:
- Python is the only source of truth for labels.
- Features at row i are known after candle i closes.
- Default entry is row i+1 open, controlled by entry_offset_bars=1.
- Every symbol can have multiple TP/SL/horizon profiles.
- Two binary targets are built per profile: long_target and short_target.

For each side-specific target:
- label = 1 means that side's TP barrier is reached first.
- label = 0 means SL, vertical barrier, no entry, or neutral same-bar outcome.
- outcome_label keeps the signed Lopez-style event: +1 TP, -1 SL, 0 vertical/neutral.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

import numpy as np
import pandas as pd

Side = Literal["long", "short"]
SameBarPolicy = Literal["sl_first", "tp_first", "neutral"]
VerticalLabelPolicy = Literal["zero", "signed"]


@dataclass(frozen=True)
class TripleBarrierParams:
    symbol: str
    side: Side
    pip_size: float
    tp_pips: float
    sl_pips: float
    max_horizon_bars: int
    profile: str = "default"
    barrier_cost_pips: float = 0.0
    entry_offset_bars: int = 1
    entry_price_column: str = "open"
    exit_price_column: str = "close"
    high_column: str = "high"
    low_column: str = "low"
    same_bar_policy: SameBarPolicy = "sl_first"
    vertical_label_policy: VerticalLabelPolicy = "zero"
    drop_no_entry_rows: bool = True


@dataclass(frozen=True)
class LabelJob:
    symbol: str
    profile: str
    side: Side


def target_name_for_side(side: Side) -> str:
    return "long_target" if side == "long" else "short_target"


def _as_float_array(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise ValueError(f"Required column {col!r} is missing.")
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _validate_params(params: TripleBarrierParams) -> None:
    if params.side not in {"long", "short"}:
        raise ValueError("side must be 'long' or 'short'.")
    if not params.profile:
        raise ValueError("profile name must be non-empty.")
    if params.pip_size <= 0:
        raise ValueError("pip_size must be positive.")
    if params.tp_pips <= 0:
        raise ValueError("tp_pips must be positive.")
    if params.sl_pips <= 0:
        raise ValueError("sl_pips must be positive.")
    if params.max_horizon_bars < 1:
        raise ValueError("max_horizon_bars must be >= 1.")
    if params.entry_offset_bars < 0:
        raise ValueError("entry_offset_bars must be >= 0.")
    if params.same_bar_policy not in {"sl_first", "tp_first", "neutral"}:
        raise ValueError("same_bar_policy must be sl_first, tp_first, or neutral.")
    if params.vertical_label_policy not in {"zero", "signed"}:
        raise ValueError("vertical_label_policy must be zero or signed.")


def _net_pips_for_side(price: float | np.ndarray, entry: float, side: Side, pip_size: float, cost_pips: float) -> float | np.ndarray:
    if side == "long":
        return (price - entry) / pip_size - cost_pips
    return (entry - price) / pip_size - cost_pips


def _hit_flags(high: float, low: float, entry: float, params: TripleBarrierParams) -> tuple[bool, bool]:
    if params.side == "long":
        best_pips = _net_pips_for_side(high, entry, "long", params.pip_size, params.barrier_cost_pips)
        worst_pips = _net_pips_for_side(low, entry, "long", params.pip_size, params.barrier_cost_pips)
    else:
        best_pips = _net_pips_for_side(low, entry, "short", params.pip_size, params.barrier_cost_pips)
        worst_pips = _net_pips_for_side(high, entry, "short", params.pip_size, params.barrier_cost_pips)
    tp_hit = bool(np.isfinite(best_pips) and best_pips >= params.tp_pips)
    sl_hit = bool(np.isfinite(worst_pips) and worst_pips <= -params.sl_pips)
    return tp_hit, sl_hit


def _vertical_outcome(realized_pips: float, policy: VerticalLabelPolicy) -> int:
    if policy == "zero":
        return 0
    if not np.isfinite(realized_pips) or abs(realized_pips) < 1e-12:
        return 0
    return 1 if realized_pips > 0 else -1


def _profile_names_for_symbol(config: Mapping[str, Any], symbol: str) -> list[str]:
    symbol_cfg = dict(config.get("symbols", {}).get(symbol, {}))
    profiles = dict(symbol_cfg.get("profiles", {}))
    if not profiles:
        # Backward compatibility with older flat configs.
        if {"tp_pips", "sl_pips", "max_horizon_bars"}.issubset(symbol_cfg):
            return [str(symbol_cfg.get("primary_profile", "default"))]
        raise KeyError(f"Symbol {symbol!r} has no configured profiles.")

    build_profiles = config.get("labeling", {}).get("build_profiles", "all")
    symbol_build_profiles = symbol_cfg.get("build_profiles", None)
    selected: Any = symbol_build_profiles if symbol_build_profiles is not None else build_profiles

    if selected == "all":
        return list(profiles.keys())
    if selected == "primary":
        return [str(symbol_cfg.get("primary_profile", next(iter(profiles.keys()))))]
    if isinstance(selected, str):
        return [selected]
    if isinstance(selected, Iterable):
        return [str(x) for x in selected]
    raise ValueError("build_profiles must be 'all', 'primary', a profile name, or a list of profile names.")


def iter_label_jobs(config: Mapping[str, Any]) -> list[LabelJob]:
    jobs: list[LabelJob] = []
    for symbol, symbol_cfg in config.get("symbols", {}).items():
        if not symbol_cfg.get("enabled", True):
            continue
        sides = [str(s).lower() for s in symbol_cfg.get("sides", ["long", "short"])]
        for profile in _profile_names_for_symbol(config, symbol):
            for side in sides:
                if side not in {"long", "short"}:
                    raise ValueError(f"Invalid side {side!r} for {symbol}.")
                jobs.append(LabelJob(symbol=symbol, profile=profile, side=side))  # type: ignore[arg-type]
    return jobs


def make_params_from_config(config: Mapping[str, Any], symbol: str, side: Side, profile: str | None = None) -> TripleBarrierParams:
    symbol_cfg = dict(config.get("symbols", {}).get(symbol, {}))
    if not symbol_cfg:
        raise KeyError(f"Symbol {symbol!r} is not configured in labels config.")
    labeling_cfg = dict(config.get("labeling", {}))
    profiles = dict(symbol_cfg.get("profiles", {}))

    profile_name = profile or str(symbol_cfg.get("primary_profile", "default"))
    if profiles:
        if profile_name not in profiles:
            raise KeyError(f"Profile {profile_name!r} is not configured for symbol {symbol!r}.")
        profile_cfg = dict(profiles[profile_name])
    else:
        # Backward compatibility with older flat configs.
        profile_cfg = symbol_cfg

    return TripleBarrierParams(
        symbol=symbol,
        side=side,
        profile=profile_name,
        pip_size=float(symbol_cfg["pip_size"]),
        tp_pips=float(profile_cfg["tp_pips"]),
        sl_pips=float(profile_cfg["sl_pips"]),
        max_horizon_bars=int(profile_cfg["max_horizon_bars"]),
        barrier_cost_pips=float(profile_cfg.get("barrier_cost_pips", symbol_cfg.get("barrier_cost_pips", 0.0))),
        entry_offset_bars=int(labeling_cfg.get("entry_offset_bars", 1)),
        entry_price_column=str(labeling_cfg.get("entry_price_column", "open")),
        exit_price_column=str(labeling_cfg.get("exit_price_column", "close")),
        high_column=str(labeling_cfg.get("high_column", "high")),
        low_column=str(labeling_cfg.get("low_column", "low")),
        same_bar_policy=str(labeling_cfg.get("same_bar_policy", "sl_first")),  # type: ignore[arg-type]
        vertical_label_policy=str(labeling_cfg.get("vertical_label_policy", "zero")),  # type: ignore[arg-type]
        drop_no_entry_rows=bool(labeling_cfg.get("drop_no_entry_rows", True)),
    )


def build_triple_barrier_labels(df: pd.DataFrame, params: TripleBarrierParams) -> pd.DataFrame:
    """Return one binary side-specific target row per feature row."""
    _validate_params(params)
    if "date" not in df.columns:
        raise ValueError("Input dataframe must include a date column.")

    data = df.copy().reset_index(drop=True)
    dates = pd.to_datetime(data["date"], errors="raise")
    high = _as_float_array(data, params.high_column)
    low = _as_float_array(data, params.low_column)
    entry_source = _as_float_array(data, params.entry_price_column)
    exit_source = _as_float_array(data, params.exit_price_column)
    n = len(data)

    records: list[dict[str, Any]] = []
    for signal_i in range(n):
        entry_i = signal_i + params.entry_offset_bars
        base = {
            "date": dates.iloc[signal_i],
            "symbol": params.symbol,
            "profile": params.profile,
            "side": params.side,
            "target_name": target_name_for_side(params.side),
            "tp_pips": float(params.tp_pips),
            "sl_pips": float(params.sl_pips),
            "max_horizon_bars": int(params.max_horizon_bars),
            "barrier_cost_pips": float(params.barrier_cost_pips),
            "same_bar_policy": params.same_bar_policy,
            "vertical_label_policy": params.vertical_label_policy,
        }
        if entry_i >= n or not np.isfinite(entry_source[entry_i]):
            rec = {
                **base,
                "label": 0,
                "outcome_label": 0,
                "event_type": "no_entry",
                "entry_index": np.nan,
                "exit_index": np.nan,
                "entry_date": pd.NaT,
                "exit_date": pd.NaT,
                "entry_price": np.nan,
                "exit_price": np.nan,
                "horizon_bars": 0,
                "bars_to_event": np.nan,
                "realized_pips": np.nan,
                "mfe_pips": np.nan,
                "mae_pips": np.nan,
                "same_bar_hit": 0,
            }
            if not params.drop_no_entry_rows:
                records.append(rec)
            continue

        entry_price = float(entry_source[entry_i])
        end_i = min(n - 1, entry_i + params.max_horizon_bars - 1)
        outcome_label = 0
        event_type = "vertical"
        exit_i = end_i
        same_bar_hit = 0

        for j in range(entry_i, end_i + 1):
            tp_hit, sl_hit = _hit_flags(float(high[j]), float(low[j]), entry_price, params)
            if tp_hit and sl_hit:
                same_bar_hit = 1
                exit_i = j
                if params.same_bar_policy == "sl_first":
                    outcome_label = -1
                    event_type = "sl_same_bar"
                elif params.same_bar_policy == "tp_first":
                    outcome_label = 1
                    event_type = "tp_same_bar"
                else:
                    outcome_label = 0
                    event_type = "both_same_bar_neutral"
                break
            if tp_hit:
                outcome_label = 1
                event_type = "tp"
                exit_i = j
                break
            if sl_hit:
                outcome_label = -1
                event_type = "sl"
                exit_i = j
                break

        if outcome_label == 1:
            exit_price = (
                entry_price + (params.tp_pips + params.barrier_cost_pips) * params.pip_size
                if params.side == "long"
                else entry_price - (params.tp_pips + params.barrier_cost_pips) * params.pip_size
            )
            realized_pips = params.tp_pips
        elif outcome_label == -1:
            exit_price = (
                entry_price - (params.sl_pips - params.barrier_cost_pips) * params.pip_size
                if params.side == "long"
                else entry_price + (params.sl_pips - params.barrier_cost_pips) * params.pip_size
            )
            realized_pips = -params.sl_pips
        else:
            exit_price = float(exit_source[exit_i]) if np.isfinite(exit_source[exit_i]) else np.nan
            realized_pips = (
                float(_net_pips_for_side(exit_price, entry_price, params.side, params.pip_size, params.barrier_cost_pips))
                if np.isfinite(exit_price)
                else np.nan
            )
            if event_type == "vertical":
                outcome_label = _vertical_outcome(realized_pips, params.vertical_label_policy)

        binary_label = 1 if outcome_label == 1 else 0

        window_high = high[entry_i : end_i + 1]
        window_low = low[entry_i : end_i + 1]
        if params.side == "long":
            mfe_pips = float(np.nanmax((window_high - entry_price) / params.pip_size - params.barrier_cost_pips))
            mae_pips = float(np.nanmin((window_low - entry_price) / params.pip_size - params.barrier_cost_pips))
        else:
            mfe_pips = float(np.nanmax((entry_price - window_low) / params.pip_size - params.barrier_cost_pips))
            mae_pips = float(np.nanmin((entry_price - window_high) / params.pip_size - params.barrier_cost_pips))

        records.append({
            **base,
            "label": int(binary_label),
            "outcome_label": int(outcome_label),
            "event_type": event_type,
            "entry_index": int(entry_i),
            "exit_index": int(exit_i),
            "entry_date": dates.iloc[entry_i],
            "exit_date": dates.iloc[exit_i],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "horizon_bars": int(params.max_horizon_bars),
            "bars_to_event": int(exit_i - entry_i + 1),
            "realized_pips": realized_pips,
            "mfe_pips": mfe_pips,
            "mae_pips": mae_pips,
            "same_bar_hit": int(same_bar_hit),
        })

    labels = pd.DataFrame.from_records(records)
    if not labels.empty:
        labels["date"] = pd.to_datetime(labels["date"])
        labels["entry_date"] = pd.to_datetime(labels["entry_date"])
        labels["exit_date"] = pd.to_datetime(labels["exit_date"])
    return labels


MODEL_INPUT_EXCLUDE_COLUMNS = [
    "date", "symbol", "open", "high", "low", "close", "tick_volume", "spread", "real_volume",
    "dxy_close", "dxy_inverse_close", "index_close",
]


def model_feature_columns(features: pd.DataFrame, exclude_columns: list[str] | None = None) -> list[str]:
    """Return only configured model feature columns from a model-feature CSV.

    The model-feature CSV intentionally keeps some identifier/raw columns such as
    date, OHLC, spread, and DXY for audit and label construction. Those columns
    must not be sent to XGBoost as model inputs.
    """
    excluded = set(exclude_columns or MODEL_INPUT_EXCLUDE_COLUMNS)
    return [c for c in features.columns if c not in excluded]


def join_features_with_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join binary labels back to model features by feature-row date.

    This is an audit/debug join and may include label metadata columns. It is not
    the strict ML input table.
    """
    left = features.copy()
    right = labels.copy()
    left["date"] = pd.to_datetime(left["date"])
    right["date"] = pd.to_datetime(right["date"])
    return left.merge(right, on="date", how="inner", suffixes=("", "_label"))


def build_ml_ready_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    target_column: str = "label",
    output_target_column: str = "label",
    exclude_feature_columns: list[str] | None = None,
    dropna: bool = False,
) -> pd.DataFrame:
    """Build a strict ML-ready dataset: configured model features + one binary label.

    Output contains no OHLC, DXY raw columns, entry/exit prices, event metadata,
    or other audit columns. It is the table to train XGBoost on.
    """
    joined = join_features_with_labels(features, labels)
    feature_cols = model_feature_columns(features, exclude_feature_columns)
    missing = [c for c in feature_cols if c not in joined.columns]
    if missing:
        raise ValueError(f"Feature columns missing after label join: {missing}")
    if target_column not in joined.columns:
        raise ValueError(f"Target column {target_column!r} is missing after label join.")
    out = joined[feature_cols + [target_column]].copy()
    if output_target_column != target_column:
        out = out.rename(columns={target_column: output_target_column})
    if dropna:
        out = out.dropna(axis=0, how="any").reset_index(drop=True)
    return out


def build_label_metadata(labels: pd.DataFrame) -> pd.DataFrame:
    """Build a non-model metadata table for temporal validation and audits.

    These columns are useful for CPCV/walk-forward splitting, realized outcome
    analysis, and debugging, but must not be used as model input features.
    """
    keep = [
        "date", "symbol", "profile", "side", "target_name",
        "label", "outcome_label", "event_type",
        "entry_date", "exit_date", "entry_price", "exit_price",
        "tp_pips", "sl_pips", "max_horizon_bars",
        "bars_to_event", "realized_pips", "mfe_pips", "mae_pips", "same_bar_hit",
    ]
    return labels[[c for c in keep if c in labels.columns]].copy()
