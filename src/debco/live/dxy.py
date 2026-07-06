from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


DXY_COMPONENT_WEIGHTS: dict[str, float] = {
    "EURUSD": -0.576,
    "USDJPY": 0.136,
    "GBPUSD": -0.119,
    "USDCAD": 0.091,
    "USDSEK": 0.042,
    "USDCHF": 0.036,
}
DXY_CONSTANT = 50.14348112


@dataclass(frozen=True)
class DXYComponentConfig:
    symbol_map: Mapping[str, str]
    enabled: bool = True


def rates_to_ohlc_frame(rates: Any, *, symbol: str) -> pd.DataFrame:
    """Convert MT5 structured rates or dict/list rows to a normalized OHLC frame."""
    if rates is None:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    if df.empty:
        return df
    if "time" not in df.columns:
        raise ValueError(f"MT5 rates for {symbol} do not include a time column.")
    df = df.copy()
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    required = ["date", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    for col in required:
        if col not in df.columns:
            if col in {"tick_volume", "spread", "real_volume"}:
                df[col] = 0
            else:
                raise ValueError(f"MT5 rates for {symbol} missing required column {col!r}.")
    out = df[required].copy()
    out["symbol"] = str(symbol).upper()
    return out.sort_values("date").reset_index(drop=True)


def build_dxy_from_component_closes(component_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build DXY and inverse-DXY from component OHLC frames.

    The canonical DXY formula is:
    DXY = 50.14348112 * EURUSD^-0.576 * USDJPY^0.136 * GBPUSD^-0.119
          * USDCAD^0.091 * USDSEK^0.042 * USDCHF^0.036
    """
    aligned: pd.DataFrame | None = None
    for dxy_symbol, weight in DXY_COMPONENT_WEIGHTS.items():
        frame = component_frames.get(dxy_symbol)
        if frame is None or frame.empty:
            raise ValueError(f"Missing DXY component frame: {dxy_symbol}")
        if "date" not in frame.columns or "close" not in frame.columns:
            raise ValueError(f"DXY component {dxy_symbol} must contain date and close columns.")
        part = frame[["date", "close"]].copy()
        part["date"] = pd.to_datetime(part["date"], utc=True, errors="coerce")
        part = part.dropna(subset=["date"]).rename(columns={"close": dxy_symbol})
        part[dxy_symbol] = pd.to_numeric(part[dxy_symbol], errors="coerce")
        part = part.dropna(subset=[dxy_symbol]).sort_values("date")
        aligned = part if aligned is None else aligned.merge(part, on="date", how="inner")
    if aligned is None or aligned.empty:
        return pd.DataFrame(columns=["date", "dxy_close", "dxy_inverse_close", "index_close"])
    dxy = pd.Series(DXY_CONSTANT, index=aligned.index, dtype=float)
    for dxy_symbol, weight in DXY_COMPONENT_WEIGHTS.items():
        dxy = dxy * np.power(pd.to_numeric(aligned[dxy_symbol], errors="coerce"), float(weight))
    out = pd.DataFrame({"date": aligned["date"], "dxy_close": dxy})
    out["dxy_inverse_close"] = 100.0 / out["dxy_close"]
    out["index_close"] = out["dxy_inverse_close"]
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "dxy_inverse_close"]).reset_index(drop=True)


def merge_dxy_into_symbol_frame(symbol_frame: pd.DataFrame, dxy_frame: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join DXY inverse columns onto a symbol OHLC frame by candle time."""
    out = symbol_frame.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    if dxy_frame is None or dxy_frame.empty:
        return out
    dxy = dxy_frame[["date", "dxy_close", "dxy_inverse_close", "index_close"]].copy()
    dxy["date"] = pd.to_datetime(dxy["date"], utc=True, errors="coerce")
    out = out.merge(dxy, on="date", how="left")
    return out
