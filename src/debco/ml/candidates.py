from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import copy
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateExperiment:
    name: str
    config: dict[str, Any]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _symbol_value(value: Any, symbol: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        if symbol in value:
            return value[symbol]
        if "default" in value:
            return value["default"]
        return default
    return value if value is not None else default


def _clean_name(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name).strip())
    return out.strip("_") or "candidate_set"


def list_candidate_experiments(config: Mapping[str, Any]) -> list[CandidateExperiment]:
    """Return enabled candidate experiment configs.

    Each experiment is a full ML config copy with one candidate_filter injected and
    a distinct output.experiment_name. This keeps the all-candles baseline and
    candidate-based runs fully separated on disk.
    """
    exp_cfg = config.get("candidate_experiments", {})
    if not bool(exp_cfg.get("enabled", False)):
        name = str(config.get("candidate_filter", {}).get("set_name", "current_config"))
        return [CandidateExperiment(name=_clean_name(name), config=copy.deepcopy(dict(config)))]

    base_name = str(exp_cfg.get("base_experiment_name", config.get("output", {}).get("experiment_name", "candidate_experiment")))
    out: list[CandidateExperiment] = []
    for item in exp_cfg.get("sets", []):
        if not bool(item.get("enabled", False)):
            continue
        name = _clean_name(str(item.get("name", "candidate_set")))
        cfg = copy.deepcopy(dict(config))
        cf = copy.deepcopy(dict(item.get("candidate_filter", {})))
        cf["enabled"] = bool(cf.get("enabled", True))
        cf["set_name"] = name
        cfg["candidate_filter"] = cf
        cfg.setdefault("output", {})
        cfg["output"]["experiment_name"] = f"{base_name}_{name}"
        out.append(CandidateExperiment(name=name, config=cfg))
    return out


def _base_tradeable_mask(x: pd.DataFrame, *, symbol: str, cfg: Mapping[str, Any]) -> pd.Series:
    base = cfg.get("base", {})
    mask = pd.Series(True, index=x.index)

    session_ids = base.get("session_block_ids", cfg.get("session_block_ids", None))
    if session_ids is not None and "session_block_id" in x.columns:
        mask &= _num(x, "session_block_id").isin([int(v) for v in session_ids])

    max_spread_by_symbol = base.get("max_spread_pips_by_symbol", cfg.get("max_spread_pips_by_symbol", {}))
    max_spread = _symbol_value(max_spread_by_symbol, symbol, None)
    if max_spread is not None and "spread_pips" in x.columns:
        mask &= _num(x, "spread_pips") <= float(max_spread)

    min_atr_pct = _symbol_value(base.get("min_atr_percentile", None), symbol, None)
    max_atr_pct = _symbol_value(base.get("max_atr_percentile", None), symbol, None)
    if "atr_percentile_240" in x.columns:
        atr_pct = _num(x, "atr_percentile_240")
        if min_atr_pct is not None:
            mask &= atr_pct >= float(min_atr_pct)
        if max_atr_pct is not None:
            mask &= atr_pct <= float(max_atr_pct)

    min_session_vol = _symbol_value(base.get("min_session_volatility_percentile", None), symbol, None)
    max_session_vol = _symbol_value(base.get("max_session_volatility_percentile", None), symbol, None)
    if "session_volatility_percentile_240" in x.columns:
        sv = _num(x, "session_volatility_percentile_240")
        if min_session_vol is not None:
            mask &= sv >= float(min_session_vol)
        if max_session_vol is not None:
            mask &= sv <= float(max_session_vol)

    min_atr_regime = base.get("min_atr_regime", None)
    max_atr_regime = base.get("max_atr_regime", None)
    if "atr_regime" in x.columns:
        ar = _num(x, "atr_regime")
        if min_atr_regime is not None:
            mask &= ar >= float(min_atr_regime)
        if max_atr_regime is not None:
            mask &= ar <= float(max_atr_regime)

    max_bar_vol = _symbol_value(base.get("max_bar_volatility_vs_atr", None), symbol, None)
    if max_bar_vol is not None and "bar_volatility_vs_atr" in x.columns:
        mask &= _num(x, "bar_volatility_vs_atr") <= float(max_bar_vol)

    min_current_range = _symbol_value(base.get("min_current_session_range_so_far_pips", None), symbol, None)
    if min_current_range is not None and "current_session_range_so_far_pips" in x.columns:
        mask &= _num(x, "current_session_range_so_far_pips") >= float(min_current_range)

    exclude_days = base.get("exclude_day_of_week", [])
    if exclude_days and "day_of_week" in x.columns:
        mask &= ~_num(x, "day_of_week").isin([int(v) for v in exclude_days])

    return mask


def _directional_trend_mask(x: pd.DataFrame, *, symbol: str, side: str, cfg: Mapping[str, Any]) -> pd.Series:
    dc = cfg.get("directional_trend_v1", cfg.get("directional_context_v1", {}))
    mask = pd.Series(True, index=x.index)
    gmma = _num(x, "gmma_distance")
    gmma_slope = _num(x, "gmma_distance_slope")
    h1_trend = _num(x, "h1_trend_direction")
    h1_spread = _num(x, "h1_ema_spread_pips")
    rsi = _num(x, "rsi")

    if side == "long":
        if "long_min_gmma_distance" in dc:
            mask &= gmma >= float(_symbol_value(dc["long_min_gmma_distance"], symbol, dc["long_min_gmma_distance"]))
        if "long_min_gmma_slope" in dc:
            mask &= gmma_slope >= float(_symbol_value(dc["long_min_gmma_slope"], symbol, dc["long_min_gmma_slope"]))
        if bool(dc.get("require_h1_direction", True)) and "h1_trend_direction" in x.columns:
            mask &= h1_trend >= 0
        if "long_min_h1_spread_pips" in dc:
            mask &= h1_spread >= float(_symbol_value(dc["long_min_h1_spread_pips"], symbol, dc["long_min_h1_spread_pips"]))
        mask &= rsi.between(float(dc.get("long_min_rsi", 45)), float(dc.get("long_max_rsi", 75)), inclusive="both")
    elif side == "short":
        if "short_max_gmma_distance" in dc:
            mask &= gmma <= float(_symbol_value(dc["short_max_gmma_distance"], symbol, dc["short_max_gmma_distance"]))
        if "short_max_gmma_slope" in dc:
            mask &= gmma_slope <= float(_symbol_value(dc["short_max_gmma_slope"], symbol, dc["short_max_gmma_slope"]))
        if bool(dc.get("require_h1_direction", True)) and "h1_trend_direction" in x.columns:
            mask &= h1_trend <= 0
        if "short_max_h1_spread_pips" in dc:
            mask &= h1_spread <= float(_symbol_value(dc["short_max_h1_spread_pips"], symbol, dc["short_max_h1_spread_pips"]))
        mask &= rsi.between(float(dc.get("short_min_rsi", 25)), float(dc.get("short_max_rsi", 55)), inclusive="both")
    return mask


def _session_breakout_mask(x: pd.DataFrame, *, symbol: str, side: str, cfg: Mapping[str, Any]) -> pd.Series:
    bc = cfg.get("session_breakout_v1", {})
    mask = pd.Series(True, index=x.index)
    expansion = _num(x, "london_expansion_vs_asia")
    min_expansion = bc.get("min_london_expansion_vs_asia", None)
    max_expansion = bc.get("max_london_expansion_vs_asia", None)
    if min_expansion is not None:
        mask &= expansion >= float(min_expansion)
    if max_expansion is not None:
        mask &= expansion <= float(max_expansion)

    breakout_pips = float(_symbol_value(bc.get("breakout_pips_by_symbol", {"default": 0.0}), symbol, 0.0))
    if side == "long":
        mask &= _num(x, "distance_from_asia_high_pips") >= breakout_pips
        if "min_current_session_return_pips_by_symbol" in bc:
            mask &= _num(x, "current_session_return_so_far_pips") >= float(_symbol_value(bc["min_current_session_return_pips_by_symbol"], symbol, 0.0))
    elif side == "short":
        mask &= _num(x, "distance_from_asia_low_pips") <= -breakout_pips
        if "min_current_session_return_pips_by_symbol" in bc:
            mask &= _num(x, "current_session_return_so_far_pips") <= -float(_symbol_value(bc["min_current_session_return_pips_by_symbol"], symbol, 0.0))
    return mask


def _trend_pullback_mask(x: pd.DataFrame, *, symbol: str, side: str, cfg: Mapping[str, Any]) -> pd.Series:
    pc = cfg.get("trend_pullback_v1", {})
    mask = _directional_trend_mask(x, symbol=symbol, side=side, cfg={"directional_trend_v1": pc.get("trend", {})})
    day_pos = _num(x, "day_position_pct")
    min_day_pos = pc.get("min_day_position_pct", None)
    max_day_pos = pc.get("max_day_position_pct", None)
    if min_day_pos is not None:
        mask &= day_pos >= float(min_day_pos)
    if max_day_pos is not None:
        mask &= day_pos <= float(max_day_pos)

    near_pips = float(_symbol_value(pc.get("near_zigzag_pips_by_symbol", {"default": 999999}), symbol, 999999))
    if side == "long":
        mask &= _num(x, "distance_from_confirmed_zigzag_low_pips") <= near_pips
        mask &= _num(x, "rsi") <= float(pc.get("long_max_rsi", 62))
    elif side == "short":
        # distance_from_confirmed_zigzag_high_pips is usually negative when price is below the high.
        mask &= _num(x, "distance_from_confirmed_zigzag_high_pips") >= -near_pips
        mask &= _num(x, "rsi") >= float(pc.get("short_min_rsi", 38))
    return mask


def _mean_reversion_mask(x: pd.DataFrame, *, symbol: str, side: str, cfg: Mapping[str, Any]) -> pd.Series:
    rc = cfg.get("mean_reversion_v1", {})
    mask = pd.Series(True, index=x.index)
    near_pips = float(_symbol_value(rc.get("near_zigzag_pips_by_symbol", {"default": 999999}), symbol, 999999))
    rsi = _num(x, "rsi")
    if side == "long":
        mask &= _num(x, "distance_from_confirmed_zigzag_low_pips") <= near_pips
        mask &= rsi <= float(rc.get("long_max_rsi", 42))
        if "min_distance_from_day_high_pips_by_symbol" in rc:
            # avoid buying directly into day high
            mask &= _num(x, "distance_from_day_high_so_far_pips") <= -float(_symbol_value(rc["min_distance_from_day_high_pips_by_symbol"], symbol, 0.0))
    elif side == "short":
        mask &= _num(x, "distance_from_confirmed_zigzag_high_pips") >= -near_pips
        mask &= rsi >= float(rc.get("short_min_rsi", 58))
        if "min_distance_from_day_low_pips_by_symbol" in rc:
            # avoid selling directly into day low
            mask &= _num(x, "distance_from_day_low_so_far_pips") >= float(_symbol_value(rc["min_distance_from_day_low_pips_by_symbol"], symbol, 0.0))
    return mask



def _series_between(s: pd.Series, low: float | None, high: float | None) -> pd.Series:
    mask = pd.Series(True, index=s.index)
    if low is not None:
        mask &= s >= float(low)
    if high is not None:
        mask &= s <= float(high)
    return mask


def _eurusd_ah_atr2_buy_mask(x: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    c = cfg.get("eurusd_ah_atr2_buy", {})
    mask = pd.Series(True, index=x.index)
    mask &= _num(x, "session_block_id") == int(c.get("session_block_id", 4))
    mask &= _num(x, "market_regime").isin([int(v) for v in c.get("market_regimes", [1, 3, 5])])
    mask &= _num(x, "distance_from_asia_high_pips") >= float(c.get("min_distance_from_asia_high_pips", 3.0))
    mask &= _num(x, "london_expansion_vs_asia") >= float(c.get("min_london_expansion_vs_asia", 0.8))
    mask &= _num(x, "gmma_distance") >= float(c.get("min_gmma_distance", 20.0))
    mask &= _num(x, "body_ratio") >= float(c.get("min_body_ratio", 0.05))
    mask &= _series_between(
        _num(x, "day_return_from_open_pips"),
        c.get("min_day_return_from_open_pips", -8.0),
        c.get("max_day_return_from_open_pips", 30.0),
    )
    mask &= _num(x, "atr_regime") == int(c.get("atr_regime", 2))
    return mask


def _eurusd_l3_r52_buy_mask(x: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    c = cfg.get("eurusd_l3_r52_buy", {})
    mask = pd.Series(True, index=x.index)
    mask &= _num(x, "session_block_id") == int(c.get("session_block_id", 3))
    mask &= _num(x, "market_regime").isin([int(v) for v in c.get("market_regimes", [0, 1, 3, 5])])
    mask &= _series_between(
        _num(x, "distance_from_asia_low_pips"),
        c.get("min_distance_from_asia_low_pips", 0.0),
        c.get("max_distance_from_asia_low_pips", 8.0),
    )
    mask &= _num(x, "rsi") <= float(c.get("max_rsi", 52.0))
    mask &= _num(x, "gmma_distance") >= float(c.get("min_gmma_distance", -10.0))
    mask &= _num(x, "body_ratio") >= float(c.get("min_body_ratio", 0.10))
    return mask


def _eurusd_l4_not2_buy_mask(x: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    c = cfg.get("eurusd_l4_not2_buy", {})
    mask = pd.Series(True, index=x.index)
    mask &= _num(x, "session_block_id") == int(c.get("session_block_id", 4))
    mask &= _series_between(
        _num(x, "distance_from_confirmed_zigzag_low_pips"),
        c.get("min_distance_from_zigzag_low_pips", 0.0),
        c.get("max_distance_from_zigzag_low_pips", 15.0),
    )
    mask &= _num(x, "rsi") <= float(c.get("max_rsi", 60.0))
    mask &= _num(x, "gmma_distance") >= float(c.get("min_gmma_distance", -30.0))
    mask &= _num(x, "h1_trend_direction") == int(c.get("h1_trend_direction", 1))
    mask &= _num(x, "body_ratio") >= float(c.get("min_body_ratio", 0.05))
    if c.get("excluded_market_regime", 2) is not None:
        mask &= _num(x, "market_regime") != int(c.get("excluded_market_regime", 2))
    return mask


def _eurusd_buy_playbook_mask(x: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    return (
        _eurusd_ah_atr2_buy_mask(x, cfg)
        | _eurusd_l3_r52_buy_mask(x, cfg)
        | _eurusd_l4_not2_buy_mask(x, cfg)
    )


def _eurusd_lateny_tuefri_short_mask(x: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    c = cfg.get("eurusd_lateny_tuefri_short", {})
    weak = (_num(x, "body_ratio") <= float(c.get("max_body_ratio", 0.05))) | (_num(x, "gmma_distance_slope") <= float(c.get("max_gmma_slope", 0.0)))
    mask = pd.Series(True, index=x.index)
    mask &= _num(x, "session_block_id") == int(c.get("session_block_id", 6))
    mask &= _num(x, "rsi") >= float(c.get("min_rsi", 60.0))
    mask &= _num(x, "gmma_distance") >= float(c.get("min_gmma_distance", 10.0))
    mask &= weak
    mask &= _num(x, "day_of_week").isin([int(v) for v in c.get("day_of_week", [1, 4])])
    return mask


def _eurusd_lateny_atrhi_short_mask(x: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    c = cfg.get("eurusd_lateny_atrhi_short", {})
    weak = (_num(x, "body_ratio") <= float(c.get("max_body_ratio", 0.05))) | (_num(x, "gmma_distance_slope") <= float(c.get("max_gmma_slope", 0.0)))
    mask = pd.Series(True, index=x.index)
    mask &= _num(x, "session_block_id") == int(c.get("session_block_id", 6))
    mask &= _num(x, "rsi") >= float(c.get("min_rsi", 65.0))
    mask &= _num(x, "gmma_distance") >= float(c.get("min_gmma_distance", 10.0))
    mask &= weak
    mask &= _num(x, "atr_regime").isin([int(v) for v in c.get("atr_regimes", [3, 4])])
    return mask


def _eurusd_london_weak_short_mask(x: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    c = cfg.get("eurusd_london_weak_short", {})
    mask = pd.Series(True, index=x.index)
    mask &= _num(x, "session_block_id") == int(c.get("session_block_id", 3))
    mask &= _num(x, "market_regime").isin([int(v) for v in c.get("market_regimes", [0, 1, 2])])
    mask &= _series_between(_num(x, "gmma_distance"), c.get("min_gmma_distance", 5.0), c.get("max_gmma_distance", 60.0))
    mask &= _num(x, "gmma_distance_slope") <= float(c.get("max_gmma_slope", -4.0))
    mask &= _num(x, "body_ratio") > float(c.get("min_body_ratio", -0.30))
    mask &= _num(x, "atr_regime") == int(c.get("atr_regime", 2))
    return mask


def _eurusd_sell_core_mask(x: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    return (
        _eurusd_lateny_tuefri_short_mask(x, cfg)
        | _eurusd_lateny_atrhi_short_mask(x, cfg)
        | _eurusd_london_weak_short_mask(x, cfg)
    )


def _xau_asial_reject_buy_mask(x: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    c = cfg.get("xau_asial_reject_buy", {})
    mask = pd.Series(True, index=x.index)
    mask &= _num(x, "session_block_id").isin([int(v) for v in c.get("session_block_ids", [4, 5])])
    mask &= _num(x, "distance_from_asia_low_pips") <= float(c.get("max_distance_from_asia_low_pips", 0.0))
    mask &= _num(x, "rsi") <= float(c.get("max_rsi", 38.0))
    mask &= _num(x, "body_ratio") >= float(c.get("min_body_ratio", 0.0))
    mask &= _num(x, "gmma_distance_slope") >= float(c.get("min_gmma_slope", 0.0))
    if c.get("max_gmma_distance", None) is not None:
        mask &= _num(x, "gmma_distance") <= float(c["max_gmma_distance"])
    return mask


def _xau_h1up_buy_mask(x: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    c = cfg.get("xau_h1up_buy", {})
    mask = pd.Series(True, index=x.index)
    mask &= _num(x, "session_block_id").isin([int(v) for v in c.get("session_block_ids", [4, 5])])
    mask &= _num(x, "h1_trend_direction") == int(c.get("h1_trend_direction", 1))
    mask &= _num(x, "atr_regime").isin([int(v) for v in c.get("atr_regimes", [3, 4])])
    mask &= _num(x, "gmma_distance") >= float(c.get("min_gmma_distance", 40.0))
    mask &= _num(x, "gmma_distance_slope") >= float(c.get("min_gmma_slope", -1.0))
    mask &= _series_between(_num(x, "rsi"), c.get("min_rsi", 45.0), c.get("max_rsi", 75.0))
    return mask


def _xau_buy_final_mask(x: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    return _xau_asial_reject_buy_mask(x, cfg) | _xau_h1up_buy_mask(x, cfg)


def _xau_dc_nofri_short_mask(x: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    c = cfg.get("xau_dc_nofri_short", {})
    mask = pd.Series(True, index=x.index)
    mask &= _num(x, "session_block_id") == int(c.get("session_block_id", 5))
    mask &= _num(x, "session_volatility_percentile_240") <= float(c.get("max_session_volatility_percentile", 0.25))
    mask &= _num(x, "gmma_distance") <= float(c.get("max_gmma_distance", -5.0))
    mask &= _num(x, "gmma_distance_slope") <= float(c.get("max_gmma_slope", 0.0))
    mask &= _num(x, "rsi") <= float(c.get("max_rsi", 55.0))
    if bool(c.get("exclude_friday", True)):
        mask &= _num(x, "day_of_week") != 4
    if bool(c.get("require_below_prev_day_high", False)):
        mask &= _num(x, "distance_from_prev_day_high_pips") < 0
    return mask


def _rule_inspired_mask(x: pd.DataFrame, *, symbol: str, side: str, cfg: Mapping[str, Any]) -> pd.Series:
    symbol_u = symbol.upper()
    if symbol_u == "EURUSD" and side == "long":
        return _eurusd_buy_playbook_mask(x, cfg)
    if symbol_u == "EURUSD" and side == "short":
        return _eurusd_sell_core_mask(x, cfg)
    if symbol_u == "XAUUSD" and side == "long":
        return _xau_buy_final_mask(x, cfg)
    if symbol_u == "XAUUSD" and side == "short":
        return _xau_dc_nofri_short_mask(x, cfg)
    return pd.Series(False, index=x.index)



def _eurusd_buy_context_mask(x: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    """Broader EURUSD buy context for meta-labeling.

    This is intentionally wider than the exact research setup. The goal is to
    create enough same-direction candidates for XGBoost to learn which setup
    contexts to accept or reject.
    """
    c = cfg.get("eurusd_buy_context", {})
    mask = pd.Series(True, index=x.index)
    mask &= _num(x, "session_block_id").isin([int(v) for v in c.get("session_block_ids", [3, 4, 5])])
    mask &= _series_between(_num(x, "rsi"), c.get("min_rsi", 35.0), c.get("max_rsi", 66.0))
    mask &= _num(x, "gmma_distance") >= float(c.get("min_gmma_distance", -35.0))
    if c.get("require_h1_non_negative", False):
        mask &= _num(x, "h1_trend_direction") >= 0
    if c.get("min_atr_regime", None) is not None:
        mask &= _num(x, "atr_regime") >= float(c.get("min_atr_regime"))
    if c.get("max_atr_regime", None) is not None:
        mask &= _num(x, "atr_regime") <= float(c.get("max_atr_regime"))
    if c.get("min_london_expansion_vs_asia", None) is not None:
        mask &= _num(x, "london_expansion_vs_asia") >= float(c.get("min_london_expansion_vs_asia"))
    if c.get("min_body_ratio", None) is not None:
        mask &= _num(x, "body_ratio") >= float(c.get("min_body_ratio"))
    return mask


def _eurusd_sell_context_mask(x: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    c = cfg.get("eurusd_sell_context", {})
    mask = pd.Series(True, index=x.index)
    mask &= _num(x, "session_block_id").isin([int(v) for v in c.get("session_block_ids", [3, 6])])
    mask &= _series_between(_num(x, "rsi"), c.get("min_rsi", 45.0), c.get("max_rsi", 78.0))
    # EUR short core was mostly late-NY/london weakness after an upside move;
    # keep the context broad enough and let the meta model score quality.
    mask &= _num(x, "gmma_distance") >= float(c.get("min_gmma_distance", -10.0))
    mask &= _num(x, "gmma_distance_slope") <= float(c.get("max_gmma_slope", 4.0))
    if c.get("exclude_friday", False):
        mask &= _num(x, "day_of_week") != 4
    if c.get("min_atr_regime", None) is not None:
        mask &= _num(x, "atr_regime") >= float(c.get("min_atr_regime"))
    if c.get("max_atr_regime", None) is not None:
        mask &= _num(x, "atr_regime") <= float(c.get("max_atr_regime"))
    return mask


def _xau_buy_context_mask(x: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    c = cfg.get("xau_buy_context", {})
    mask = pd.Series(True, index=x.index)
    mask &= _num(x, "session_block_id").isin([int(v) for v in c.get("session_block_ids", [4, 5])])
    mask &= _series_between(_num(x, "rsi"), c.get("min_rsi", 30.0), c.get("max_rsi", 68.0))
    if c.get("require_h1_non_negative", True):
        mask &= _num(x, "h1_trend_direction") >= 0
    if c.get("min_gmma_distance", None) is not None:
        mask &= _num(x, "gmma_distance") >= float(c.get("min_gmma_distance"))
    if c.get("min_gmma_slope", None) is not None:
        mask &= _num(x, "gmma_distance_slope") >= float(c.get("min_gmma_slope"))
    if c.get("min_atr_regime", None) is not None:
        mask &= _num(x, "atr_regime") >= float(c.get("min_atr_regime"))
    if c.get("max_atr_regime", None) is not None:
        mask &= _num(x, "atr_regime") <= float(c.get("max_atr_regime"))
    return mask


def _xau_sell_context_mask(x: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    c = cfg.get("xau_sell_context", {})
    mask = pd.Series(True, index=x.index)
    mask &= _num(x, "session_block_id").isin([int(v) for v in c.get("session_block_ids", [5])])
    mask &= _num(x, "rsi") <= float(c.get("max_rsi", 62.0))
    mask &= _num(x, "gmma_distance_slope") <= float(c.get("max_gmma_slope", 3.0))
    if c.get("max_gmma_distance", None) is not None:
        mask &= _num(x, "gmma_distance") <= float(c.get("max_gmma_distance"))
    if c.get("max_session_volatility_percentile", None) is not None:
        mask &= _num(x, "session_volatility_percentile_240") <= float(c.get("max_session_volatility_percentile"))
    if bool(c.get("exclude_friday", True)):
        mask &= _num(x, "day_of_week") != 4
    return mask


def _rule_context_mask(x: pd.DataFrame, *, symbol: str, side: str, cfg: Mapping[str, Any]) -> pd.Series:
    symbol_u = symbol.upper()
    if symbol_u == "EURUSD" and side == "long":
        return _eurusd_buy_context_mask(x, cfg)
    if symbol_u == "EURUSD" and side == "short":
        return _eurusd_sell_context_mask(x, cfg)
    if symbol_u == "XAUUSD" and side == "long":
        return _xau_buy_context_mask(x, cfg)
    if symbol_u == "XAUUSD" and side == "short":
        return _xau_sell_context_mask(x, cfg)
    return pd.Series(False, index=x.index)

def candidate_mask_for_job(x: pd.DataFrame, *, symbol: str, side: str, config: Mapping[str, Any]) -> pd.Series:
    """Return candidate rows for candidate-based/meta-label training.

    The mask is built only from same-bar model features. It never uses labels,
    realized_pips, entry/exit outcomes, or future information.
    """
    cfg = config.get("candidate_filter", {})
    mask = pd.Series(True, index=x.index)
    if not bool(cfg.get("enabled", False)):
        return mask

    mask &= _base_tradeable_mask(x, symbol=symbol, cfg=cfg)
    preset = str(cfg.get("preset", "session_volatility_v1"))

    if preset in {"session_volatility_v1", "session_tradeable_v1"}:
        pass
    elif preset in {"directional_context_v1", "directional_trend_v1"}:
        mask &= _directional_trend_mask(x, symbol=symbol, side=side, cfg=cfg)
    elif preset == "session_breakout_v1":
        mask &= _session_breakout_mask(x, symbol=symbol, side=side, cfg=cfg)
    elif preset == "trend_pullback_v1":
        mask &= _trend_pullback_mask(x, symbol=symbol, side=side, cfg=cfg)
    elif preset == "mean_reversion_v1":
        mask &= _mean_reversion_mask(x, symbol=symbol, side=side, cfg=cfg)
    elif preset == "rule_inspired_core_v1":
        mask &= _rule_inspired_mask(x, symbol=symbol, side=side, cfg=cfg)
    elif preset == "rule_inspired_context_v1":
        mask &= _rule_context_mask(x, symbol=symbol, side=side, cfg=cfg)
    elif preset == "rule_inspired_core_or_context_v1":
        mask &= (_rule_inspired_mask(x, symbol=symbol, side=side, cfg=cfg) | _rule_context_mask(x, symbol=symbol, side=side, cfg=cfg))
    elif preset == "eurusd_buy_playbook_v1":
        mask &= _eurusd_buy_playbook_mask(x, cfg) if symbol.upper() == "EURUSD" and side == "long" else False
    elif preset == "eurusd_sell_core_v1":
        mask &= _eurusd_sell_core_mask(x, cfg) if symbol.upper() == "EURUSD" and side == "short" else False
    elif preset == "xau_buy_final_v1":
        mask &= _xau_buy_final_mask(x, cfg) if symbol.upper() == "XAUUSD" and side == "long" else False
    elif preset == "xau_sell_dc_nofri_v1":
        mask &= _xau_dc_nofri_short_mask(x, cfg) if symbol.upper() == "XAUUSD" and side == "short" else False
    else:
        raise ValueError(f"Unsupported candidate_filter preset: {preset}")

    min_keep_ratio = cfg.get("min_keep_ratio", None)
    if min_keep_ratio is not None and len(mask):
        keep_ratio = float(mask.fillna(False).mean())
        if keep_ratio < float(min_keep_ratio):
            # Do not silently train on a tiny, accidental subset.
            raise ValueError(
                f"Candidate filter {cfg.get('set_name', preset)!r} kept only {keep_ratio:.4f}; "
                f"min_keep_ratio={float(min_keep_ratio):.4f}. Relax the filter or lower min_keep_ratio."
            )
    return mask.fillna(False).astype(bool)


def apply_candidate_filter(
    x: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    *,
    symbol: str,
    side: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, float]]:
    mask = candidate_mask_for_job(x, symbol=symbol, side=side, config=config)
    before = len(x)
    x2 = x.loc[mask].reset_index(drop=True)
    y2 = y.loc[mask].reset_index(drop=True)
    m2 = metadata.loc[mask].reset_index(drop=True)
    after = len(x2)
    stats = {
        "candidate_set": str(config.get("candidate_filter", {}).get("set_name", "all_candles")),
        "candidate_preset": str(config.get("candidate_filter", {}).get("preset", "all_candles")),
        "candidate_filter_enabled": float(bool(config.get("candidate_filter", {}).get("enabled", False))),
        "candidate_rows_before": float(before),
        "candidate_rows_after": float(after),
        "candidate_keep_ratio": float(after / before) if before else np.nan,
        "candidate_positive_rate": float((y2 == 1).mean()) if after else np.nan,
        "candidate_positive_count": float((y2 == 1).sum()) if after else 0.0,
        "candidate_negative_count": float((y2 == 0).sum()) if after else 0.0,
    }
    return x2, y2, m2, stats
