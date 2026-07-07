from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_PIP_SIZE: dict[str, float] = {
    "EURUSD": 0.0001,
    "XAUUSD": 0.01,
}


@dataclass(frozen=True)
class TradeProfile:
    tp_pips: float
    sl_pips: float
    horizon_bars: int | None = None


def parse_trade_profile_from_job(job: str | None) -> TradeProfile | None:
    """Extract TP/SL/horizon from job names used by the research pipeline.

    Examples:
    - EURUSD_fast_15_8_h16_long -> TP=15 pips, SL=8 pips, H=16 bars
    - XAUUSD_runner_2200_1100_h40_long -> TP=2200 pips, SL=1100 pips, H=40 bars
    """
    if not job:
        return None
    m = re.search(r"_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_h(\d+)(?:_|$)", str(job))
    if not m:
        return None
    return TradeProfile(tp_pips=float(m.group(1)), sl_pips=float(m.group(2)), horizon_bars=int(m.group(3)))


def pip_size_for_symbol(symbol: str, *, config: Mapping[str, Any] | None = None) -> float:
    symbol = str(symbol).upper()
    cfg = config or {}
    by_symbol = cfg.get("pip_size", {}) if isinstance(cfg, Mapping) else {}
    if isinstance(by_symbol, Mapping) and symbol in by_symbol:
        return float(by_symbol[symbol])
    return float(DEFAULT_PIP_SIZE.get(symbol, 0.0001))


def risk_weight_for_decision(live_spec: Mapping[str, Any], *, symbol: str, side: str, setup_id: str) -> float:
    """Resolve the final portfolio risk weight for a live decision.

    Supports the risk_plan_weights object emitted by the v0.1.12 final report.
    Precedence: component/setup > symbol|side > symbol > side > 1.0.
    """
    weights = live_spec.get("risk_plan_weights", {}) or {}
    if not isinstance(weights, Mapping):
        return 1.0
    symbol = str(symbol).upper()
    side = str(side).lower()
    setup_id = str(setup_id)

    component_weights = weights.get("component_weights", {}) or {}
    if isinstance(component_weights, Mapping) and setup_id in component_weights:
        return float(component_weights[setup_id])

    symbol_side_weights = weights.get("symbol_side_weights", {}) or {}
    key = f"{symbol}|{side}"
    if isinstance(symbol_side_weights, Mapping) and key in symbol_side_weights:
        return float(symbol_side_weights[key])

    symbol_weights = weights.get("symbol_weights", {}) or {}
    if isinstance(symbol_weights, Mapping) and symbol in symbol_weights:
        return float(symbol_weights[symbol])

    side_weights = weights.get("side_weights", {}) or {}
    if isinstance(side_weights, Mapping) and side in side_weights:
        return float(side_weights[side])

    return 1.0


def normalize_volume(raw_volume: float, *, volume_min: float, volume_max: float, volume_step: float) -> float:
    if raw_volume <= 0 or not math.isfinite(raw_volume):
        return 0.0
    volume_step = float(volume_step or 0.01)
    volume_min = float(volume_min or volume_step)
    volume_max = float(volume_max or raw_volume)
    steps = math.floor(raw_volume / volume_step)
    vol = steps * volume_step
    if vol < volume_min:
        vol = volume_min
    if vol > volume_max:
        vol = volume_max
    # Most retail FX/CFD brokers use two decimals for lot sizes. If the step is
    # smaller, preserve enough decimals without floating-point noise.
    decimals = max(2, int(abs(math.log10(volume_step))) + 1 if volume_step < 1 else 2)
    return round(vol, decimals)
