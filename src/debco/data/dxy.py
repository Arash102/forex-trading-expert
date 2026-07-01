from __future__ import annotations

from functools import reduce
from typing import Mapping

import numpy as np
import pandas as pd

DXY_DEFAULT_CONSTANT = 50.14348112
DXY_DEFAULT_WEIGHTS: dict[str, float] = {
    "EURUSD": -0.576,
    "USDJPY": 0.136,
    "GBPUSD": -0.119,
    "USDCAD": 0.091,
    "USDSEK": 0.042,
    "USDCHF": 0.036,
}


def _close_frame(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{name} dataframe must include date and close columns")
    out = df[["date", "close"]].copy()
    out = out.rename(columns={"close": f"{name}_close"})
    return out.sort_values("date")


def align_component_closes(component_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [_close_frame(name, df) for name, df in component_frames.items()]
    if not frames:
        raise ValueError("No DXY components provided")
    merged = reduce(lambda left, right: pd.merge(left, right, on="date", how="inner"), frames)
    return merged.sort_values("date").reset_index(drop=True)


def build_dxy_from_components(
    component_frames: Mapping[str, pd.DataFrame],
    *,
    constant: float = DXY_DEFAULT_CONSTANT,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build DXY in Python from component OHLC closes.

    Formula used in the project:
    DXY = constant * EURUSD^-0.576 * USDJPY^0.136 * GBPUSD^-0.119
          * USDCAD^0.091 * USDSEK^0.042 * USDCHF^0.036
    """
    weights = dict(weights or DXY_DEFAULT_WEIGHTS)
    missing = sorted(set(weights) - set(component_frames))
    if missing:
        raise ValueError(f"Missing DXY components: {missing}")

    aligned = align_component_closes({k: component_frames[k] for k in weights})
    dxy = np.full(len(aligned), constant, dtype=float)
    for symbol, weight in weights.items():
        close = aligned[f"{symbol}_close"].astype(float).to_numpy()
        if np.any(close <= 0):
            raise ValueError(f"Non-positive close found in DXY component: {symbol}")
        dxy *= np.power(close, weight)

    out = aligned[["date"]].copy()
    out["dxy_close"] = dxy
    out["dxy_inverse_close"] = 100.0 / out["dxy_close"]
    out["index_close"] = out["dxy_inverse_close"]
    return out


def merge_symbol_with_dxy(symbol_df: pd.DataFrame, dxy_df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "dxy_close", "dxy_inverse_close", "index_close"}
    if not required.issubset(dxy_df.columns):
        raise ValueError(f"dxy_df missing columns: {sorted(required - set(dxy_df.columns))}")
    return pd.merge(symbol_df.sort_values("date"), dxy_df.sort_values("date"), on="date", how="inner")
