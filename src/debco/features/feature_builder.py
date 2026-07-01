from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)


def session_block_id(date: pd.Series) -> pd.Series:
    dt = pd.to_datetime(date)
    minutes = dt.dt.hour * 60 + dt.dt.minute
    out = pd.Series(0, index=date.index, dtype="int64")
    out[(minutes >= 0) & (minutes < 420)] = 1
    out[(minutes >= 420) & (minutes < 540)] = 2
    out[(minutes >= 540) & (minutes < 720)] = 3
    out[(minutes >= 720) & (minutes < 840)] = 4
    out[(minutes >= 840) & (minutes < 960)] = 5
    out[(minutes >= 960) & (minutes < 1260)] = 6
    return out


def add_asia_london_context(df: pd.DataFrame, pip_size: float) -> pd.DataFrame:
    out = df.copy()
    out["day"] = pd.to_datetime(out["date"]).dt.floor("D")
    out["minute"] = pd.to_datetime(out["date"]).dt.hour * 60 + pd.to_datetime(out["date"]).dt.minute

    asia = out[out["minute"].between(0, 419)].groupby("day").agg(
        asia_high=("high", "max"),
        asia_low=("low", "min"),
        asia_range=("high", lambda x: np.nan),
    )
    asia["asia_range"] = asia["asia_high"] - asia["asia_low"]
    out = out.merge(asia, left_on="day", right_index=True, how="left")

    london_mask = out["minute"].between(420, 959)
    london_high = out.where(london_mask).groupby("day")["high"].cummax()
    london_low = out.where(london_mask).groupby("day")["low"].cummin()
    out["london_high_so_far"] = london_high.groupby(out["day"]).ffill()
    out["london_low_so_far"] = london_low.groupby(out["day"]).ffill()
    out["london_range_so_far"] = out["london_high_so_far"] - out["london_low_so_far"]

    out["distance_from_asia_high_pips"] = (out["close"] - out["asia_high"]) / pip_size
    out["distance_from_asia_low_pips"] = (out["close"] - out["asia_low"]) / pip_size
    out["london_expansion_vs_asia"] = out["london_range_so_far"] / out["asia_range"].replace(0, np.nan)

    day_open = out.groupby("day")["open"].transform("first")
    out["day_return_from_open_pips"] = (out["close"] - day_open) / pip_size
    return out.drop(columns=["day", "minute"], errors="ignore")


def add_gmma(df: pd.DataFrame, short_periods: list[int], long_periods: list[int], slope_lag: int = 5) -> pd.DataFrame:
    out = df.copy()
    for p in short_periods + long_periods:
        out[f"ema_{p}"] = ema(out["close"], p)
    short_mean = out[[f"ema_{p}" for p in short_periods]].mean(axis=1)
    long_mean = out[[f"ema_{p}" for p in long_periods]].mean(axis=1)
    raw = short_mean - long_mean

    # Bounded normalized GMMA distance, consistent with the project convention:
    # positive = short-term EMAs above long-term EMAs; negative = bearish structure.
    mean_abs = raw.abs().rolling(30, min_periods=30).mean()
    z = raw.abs() / mean_abs.replace(0, np.nan)
    bounded = 100.0 - (100.0 / (1.0 + z))
    out["gmma_distance"] = np.where(raw < 0, -bounded, bounded)
    out["gmma_distance_slope"] = (out["gmma_distance"] - out["gmma_distance"].shift(slope_lag)) / slope_lag
    return out


def add_market_features(df: pd.DataFrame, window: int = 20, momentum_lag: int = 20) -> pd.DataFrame:
    if "index_close" not in df.columns:
        raise ValueError("index_close is required for market features. Build and merge DXY first.")
    out = df.copy()
    out["relative_price"] = out["close"] / out["index_close"]
    out["relative_strength_20"] = out["relative_price"] / out["relative_price"].rolling(window, min_periods=window).mean()
    out["relative_momentum_20"] = out["relative_strength_20"] - out["relative_strength_20"].shift(momentum_lag)
    out["market_x"] = zscore(out["relative_strength_20"], window)
    out["market_y"] = zscore(out["relative_momentum_20"], window)

    # Initial transparent regime mapping. This must be compared with prior clean_feature_engineering.py
    # before model research is finalized.
    conditions = [
        (out["market_x"] > 0.5) & (out["market_y"] > 0),
        (out["market_x"] < -0.5) & (out["market_y"] < 0),
        (out["market_x"].abs() <= 0.5),
        (out["market_x"] < -0.5) & (out["market_y"] > 0),
    ]
    choices = [1, 2, 3, 5]
    out["market_regime"] = np.select(conditions, choices, default=0).astype(int)
    return out


def add_atr_regime(df: pd.DataFrame, pip_size: float, period: int = 14, lookback: int = 240) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = atr(out, period)
    out["atr_pips"] = out["atr"] / pip_size
    pct = out["atr_pips"].rolling(lookback, min_periods=max(50, lookback // 3)).rank(pct=True)
    out["atr_percentile_240"] = pct
    out["atr_regime"] = np.select(
        [pct >= 0.90, pct >= 0.75, pct <= 0.25],
        [4, 3, 1],
        default=2,
    ).astype(int)
    return out


def add_h1_trend_direction(df: pd.DataFrame, pip_size: float) -> pd.DataFrame:
    out = df.copy()
    tmp = out.set_index(pd.to_datetime(out["date"])).sort_index()
    h1 = tmp["close"].resample("1h").last().dropna().to_frame("h1_close")
    h1["h1_ema8"] = ema(h1["h1_close"], 8)
    h1["h1_ema21"] = ema(h1["h1_close"], 21)
    h1["spread_pips"] = (h1["h1_ema8"] - h1["h1_ema21"]) / pip_size
    h1["slope_pips"] = (h1["h1_ema8"] - h1["h1_ema8"].shift(3)) / (3 * pip_size)
    h1["h1_trend_direction"] = np.select(
        [(h1["spread_pips"] > 3) & (h1["slope_pips"] > 0), (h1["spread_pips"] < -3) & (h1["slope_pips"] < 0)],
        [1, -1],
        default=0,
    )
    aligned = pd.merge_asof(
        out.sort_values("date"),
        h1[["h1_trend_direction"]].reset_index().rename(columns={"index": "date"}).sort_values("date"),
        on="date",
        direction="backward",
    )
    return aligned


def build_features(df: pd.DataFrame, *, symbol: str, pip_size: float, config: dict) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    fcfg = config.get("features", config)
    out["session_block_id"] = session_block_id(out["date"])
    out["day_of_week"] = pd.to_datetime(out["date"]).dt.dayofweek
    out["body_ratio"] = (out["close"] - out["open"]) / (out["high"] - out["low"]).replace(0, np.nan)
    out["rsi"] = rsi(out["close"], int(fcfg.get("rsi_period", 14)))
    out = add_atr_regime(out, pip_size, int(fcfg.get("atr_period", 14)), int(fcfg.get("atr_regime_lookback", 240)))
    out = add_gmma(out, list(fcfg.get("gmma_short_periods", [3,5,8,10,12,15])), list(fcfg.get("gmma_long_periods", [30,35,40,45,50,60])))
    out = add_market_features(out, int(fcfg.get("market_window", 20)), int(fcfg.get("market_momentum_lag", 20)))
    out = add_asia_london_context(out, pip_size)
    out = add_h1_trend_direction(out, pip_size)
    return out
