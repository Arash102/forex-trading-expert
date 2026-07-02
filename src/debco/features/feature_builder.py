from __future__ import annotations

"""Symbol-aware, causal feature engineering for the DebCo research engine.

Design rules:
- Python is the only source of truth for all features.
- All price-scale parameters are symbol-aware through ``features_config``.
- Full feature output is kept for research/debug/rule-base.
- Model feature output is a controlled subset plus explicitly configured lags.
- No future bars are used for rolling/session features.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


# -----------------------------
# Config helpers
# -----------------------------

@dataclass(frozen=True)
class FeatureParams:
    symbol: str = "EURUSD"
    pip_size: float = 0.0001
    point_size: float = 0.00001
    timezone: str = "UTC"

    zigzag_depth: int = 10
    zigzag_deviation_pct: float = 0.13

    fracdiff_order: float = 0.4
    fracdiff_window: int = 50

    atr_period: int = 14
    rsi_period: int = 14
    rsi_slope_window: int = 5

    volume_ratio_window: int = 30

    gmma_short_spans: tuple[int, ...] = (3, 5, 8, 10, 12, 15)
    gmma_long_spans: tuple[int, ...] = (30, 35, 40, 45, 50, 60)
    gmma_normalize_window: int = 30
    gmma_distance_slope_window: int = 5

    market_rs_window_short: int = 20
    market_rs_window_long: int = 60
    market_momentum_lag: int = 20
    market_z_window: int = 20
    market_regime_vol_window: int = 120
    market_regime_vol_quantile: float = 0.85

    atr_percentile_window: int = 240
    session_volatility_window: int = 240

    prev_session_bias_pips: float = 8.0
    day_bias_pips: float = 8.0
    h1_trend_spread_threshold_pips: float = 3.0


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_symbol_config(config: Mapping[str, Any], symbol: str) -> dict[str, Any]:
    defaults = dict(config.get("defaults", {}))
    symbol_cfg = dict(config.get("symbols", {}).get(symbol, {}))
    if not symbol_cfg:
        raise KeyError(f"Symbol {symbol!r} is not configured in features config.")
    return deep_merge(defaults, symbol_cfg)


def feature_params_from_config(config: Mapping[str, Any], symbol: str) -> FeatureParams:
    cfg = resolve_symbol_config(config, symbol)
    bias = cfg.get("bias_thresholds", {})
    sessions = config.get("sessions", {})
    return FeatureParams(
        symbol=symbol,
        pip_size=float(cfg.get("pip_size", 0.0001)),
        point_size=float(cfg.get("point_size", cfg.get("pip_size", 0.0001))),
        timezone=str(cfg.get("timezone", sessions.get("timezone", "UTC"))),
        zigzag_depth=int(cfg.get("zigzag_depth", 10)),
        zigzag_deviation_pct=float(cfg.get("zigzag_deviation_pct", 0.13)),
        fracdiff_order=float(cfg.get("fracdiff_order", 0.4)),
        fracdiff_window=int(cfg.get("fracdiff_window", 50)),
        atr_period=int(cfg.get("atr_period", 14)),
        rsi_period=int(cfg.get("rsi_period", 14)),
        rsi_slope_window=int(cfg.get("rsi_slope_window", 5)),
        volume_ratio_window=int(cfg.get("volume_ratio_window", 30)),
        gmma_short_spans=tuple(int(x) for x in cfg.get("gmma_short_spans", [3, 5, 8, 10, 12, 15])),
        gmma_long_spans=tuple(int(x) for x in cfg.get("gmma_long_spans", [30, 35, 40, 45, 50, 60])),
        gmma_normalize_window=int(cfg.get("gmma_normalize_window", 30)),
        gmma_distance_slope_window=int(cfg.get("gmma_distance_slope_window", 5)),
        market_rs_window_short=int(cfg.get("market_rs_window_short", 20)),
        market_rs_window_long=int(cfg.get("market_rs_window_long", 60)),
        market_momentum_lag=int(cfg.get("market_momentum_lag", 20)),
        market_z_window=int(cfg.get("market_z_window", 20)),
        market_regime_vol_window=int(cfg.get("market_regime_vol_window", 120)),
        market_regime_vol_quantile=float(cfg.get("market_regime_vol_quantile", 0.85)),
        atr_percentile_window=int(cfg.get("atr_percentile_window", 240)),
        session_volatility_window=int(cfg.get("session_volatility_window", 240)),
        prev_session_bias_pips=float(bias.get("prev_session_bias_pips", 8.0)),
        day_bias_pips=float(bias.get("day_bias_pips", 8.0)),
        h1_trend_spread_threshold_pips=float(cfg.get("h1_trend_spread_threshold_pips", 3.0)),
    )


# -----------------------------
# Math helpers
# -----------------------------

def safe_div(a: Any, b: Any, fill_value: float = np.nan) -> Any:
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        idx = a.index if isinstance(a, pd.Series) else b.index
        left = a.astype(float) if isinstance(a, pd.Series) else pd.Series(a, index=idx, dtype=float)
        right = b.astype(float) if isinstance(b, pd.Series) else pd.Series(b, index=idx, dtype=float)
        out = left.div(right)
        return out.where(np.isfinite(out), fill_value)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.true_divide(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    if np.ndim(out) == 0:
        return float(out) if np.isfinite(out) else fill_value
    out = np.asarray(out, dtype=float)
    out[~np.isfinite(out)] = fill_value
    return out


def calc_true_zscore(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    roll = s.rolling(window, min_periods=window)
    return safe_div(s - roll.mean(), roll.std(ddof=1))


def bounded_normalize(series: pd.Series, window: int = 30) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    base = s.abs().rolling(window, min_periods=window).mean()
    z = safe_div(s.abs(), base)
    return 100.0 - (100.0 / (1.0 + z))


def calc_slope(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return safe_div(s - s.shift(window), float(window))


def fractional_difference_weights(d: float, window: int) -> np.ndarray:
    if int(window) < 2:
        raise ValueError("fracdiff_window must be >= 2")
    weights = np.empty(int(window), dtype=float)
    weights[0] = 1.0
    for k in range(1, int(window)):
        weights[k] = -weights[k - 1] * (float(d) - k + 1.0) / k
    return weights


def fractional_difference(series: pd.Series, d: float = 0.4, window: int = 50) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float).ffill().to_numpy(dtype=float)
    weights = fractional_difference_weights(d, int(window))
    transformed = np.convolve(values, weights, mode="full")[: len(values)]
    transformed[: int(window) - 1] = np.nan
    return pd.Series(transformed, index=series.index, name=series.name)


def _wilder_ewm(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(alpha=1.0 / int(window), adjust=False, min_periods=int(window)).mean()


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _wilder_ewm(gain, window)
    avg_loss = _wilder_ewm(loss, window)
    rs = safe_div(avg_gain, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return rsi


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return _wilder_ewm(true_range(high, low, close), int(window))


def _rolling_percentile_against_past(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    window = int(window)
    min_periods = int(min_periods)
    for i, x in enumerate(values):
        if not np.isfinite(x):
            continue
        start = max(0, i - window)
        hist = values[start:i]
        hist = hist[np.isfinite(hist)]
        if len(hist) < min_periods:
            continue
        out[i] = float(np.mean(hist <= x))
    return pd.Series(out, index=series.index, dtype="float64")


# -----------------------------
# Time/session helpers
# -----------------------------

def _parse_hhmm(value: str) -> tuple[int, int]:
    h, m = str(value).split(":")[:2]
    return int(h), int(m)


def _time_to_minutes(value: str) -> int:
    h, m = _parse_hhmm(value)
    return h * 60 + m


def _time_in_minutes(minutes: pd.Series, start: int, end: int) -> pd.Series:
    if start <= end:
        return minutes.ge(start) & minutes.lt(end)
    return minutes.ge(start) | minutes.lt(end)


def validate_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input data is missing OHLC columns: {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    for c in ["open", "high", "low", "close", "tick_volume", "real_volume", "spread", "index_close", "dxy_close", "dxy_inverse_close"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "tick_volume" not in out.columns:
        out["tick_volume"] = 0.0
    if "real_volume" not in out.columns:
        out["real_volume"] = 0.0
    if "spread" not in out.columns:
        out["spread"] = np.nan
    out["symbol"] = symbol
    prices = out[["open", "high", "low", "close"]]
    bad = (~np.isfinite(prices).all(axis=1)) | (prices <= 0).any(axis=1)
    if bad.any():
        raise ValueError(f"Invalid OHLC rows: {out.index[bad].tolist()[:10]}")
    if (out["high"] < out["low"]).any():
        raise ValueError("Found high < low.")
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


# -----------------------------
# Feature families
# -----------------------------

def add_confirmed_zigzag_features(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    out = df.copy()
    high = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    n = len(out)
    right = max(1, int(params.zigzag_depth) // 2)
    left = right
    deviation = abs(float(params.zigzag_deviation_pct)) / 100.0

    last_high = np.full(n, np.nan)
    last_low = np.full(n, np.nan)
    current_high = np.nan
    current_low = np.nan

    for i in range(n):
        pivot_idx = i - right
        if pivot_idx >= left:
            hs = high[pivot_idx - left : pivot_idx + right + 1]
            ls = low[pivot_idx - left : pivot_idx + right + 1]
            if len(hs) == left + right + 1 and np.isfinite(hs).all() and high[pivot_idx] >= np.max(hs):
                if np.isnan(current_low) or high[pivot_idx] >= current_low * (1.0 + deviation):
                    current_high = high[pivot_idx]
            if len(ls) == left + right + 1 and np.isfinite(ls).all() and low[pivot_idx] <= np.min(ls):
                if np.isnan(current_high) or low[pivot_idx] <= current_high * (1.0 - deviation):
                    current_low = low[pivot_idx]
        last_high[i] = current_high
        last_low[i] = current_low

    out["last_confirmed_zigzag_high"] = last_high
    out["last_confirmed_zigzag_low"] = last_low
    out["distance_from_confirmed_zigzag_high_pips"] = safe_div(close - last_high, params.pip_size)
    out["distance_from_confirmed_zigzag_low_pips"] = safe_div(close - last_low, params.pip_size)
    return out


def add_gmma_features(out: pd.DataFrame, close: pd.Series, params: FeatureParams) -> pd.DataFrame:
    short_cols: list[str] = []
    long_cols: list[str] = []
    all_spans = sorted(set(params.gmma_short_spans + params.gmma_long_spans))
    for span in all_spans:
        col = f"ema_{span}"
        out[col] = close.ewm(span=span, adjust=False, min_periods=span).mean()
        if span in params.gmma_short_spans:
            short_cols.append(col)
        if span in params.gmma_long_spans:
            long_cols.append(col)

    short_emas = out[short_cols].astype(float)
    long_emas = out[long_cols].astype(float)
    short_mean = short_emas.mean(axis=1, skipna=False)
    long_mean = long_emas.mean(axis=1, skipna=False)
    short_spread = short_emas.max(axis=1) - short_emas.min(axis=1)
    long_spread = long_emas.max(axis=1) - long_emas.min(axis=1)
    ema_distance = short_mean - long_mean

    out["gmma_short_mean"] = short_mean
    out["gmma_long_mean"] = long_mean
    out["gmma_green"] = bounded_normalize(short_spread, window=params.gmma_normalize_window)
    out["gmma_red"] = bounded_normalize(long_spread, window=params.gmma_normalize_window)
    out["gmma_distance"] = bounded_normalize(ema_distance.abs(), window=params.gmma_normalize_window) * np.sign(ema_distance)
    out["gmma_distance_slope"] = calc_slope(out["gmma_distance"], params.gmma_distance_slope_window)

    short_min = short_emas.min(axis=1, skipna=False)
    short_max = short_emas.max(axis=1, skipna=False)
    long_min = long_emas.min(axis=1, skipna=False)
    long_max = long_emas.max(axis=1, skipna=False)
    gmma_state = pd.Series(0, index=out.index, dtype="int8")
    gmma_state.loc[short_min > long_max] = 1
    gmma_state.loc[short_max < long_min] = -1
    out["gmma"] = gmma_state.astype("int8")
    return out


def add_market_features(out: pd.DataFrame, close: pd.Series, params: FeatureParams) -> pd.DataFrame:
    if "index_close" not in out.columns:
        if "dxy_inverse_close" in out.columns:
            out["index_close"] = out["dxy_inverse_close"]
        else:
            out["index_close"] = close
    index_close = pd.to_numeric(out["index_close"], errors="coerce")
    relative_price = safe_div(close, index_close)

    rs_short = params.market_rs_window_short
    rs_long = params.market_rs_window_long
    lag = params.market_momentum_lag
    zwin = params.market_z_window

    out["relative_strength_20"] = safe_div(relative_price, relative_price.rolling(rs_short, min_periods=rs_short).mean())
    out["relative_strength_60"] = safe_div(relative_price, relative_price.rolling(rs_long, min_periods=rs_long).mean())
    out["relative_momentum_20"] = out["relative_strength_20"] - out["relative_strength_20"].shift(lag)
    out["market_x"] = calc_true_zscore(out["relative_strength_20"], zwin)
    out["market_y"] = calc_true_zscore(out["relative_momentum_20"], zwin)

    conditions = [
        (out["market_x"] > 0) & (out["market_y"] > 0),
        (out["market_x"] < 0) & (out["market_y"] > 0),
        (out["market_x"] < 0) & (out["market_y"] < 0),
        (out["market_x"] > 0) & (out["market_y"] < 0),
    ]
    out["market_quadrant"] = np.select(conditions, [1, 2, 3, 4], default=0).astype("int8")

    symbol_return = close.pct_change(fill_method=None)
    index_return = index_close.pct_change(fill_method=None)
    cov20 = symbol_return.rolling(20, min_periods=20).cov(index_return)
    var20 = index_return.rolling(20, min_periods=20).var(ddof=1)
    out["rolling_beta_20"] = safe_div(cov20, var20)
    out["rolling_corr_20"] = symbol_return.rolling(20, min_periods=20).corr(index_return)
    out["market_volatility"] = safe_div(
        symbol_return.rolling(20, min_periods=20).std(ddof=1),
        index_return.rolling(20, min_periods=20).std(ddof=1),
    )

    trend_raw = out["relative_strength_20"] * out["relative_momentum_20"]
    market_trend_component = (bounded_normalize(trend_raw.abs(), 30) / 100.0).clip(-1.0, 1.0) * np.sign(trend_raw)
    market_x_component = np.tanh(out["market_x"] / 2.0)
    market_y_component = np.tanh(out["market_y"] / 2.0)
    out["market_regime_score"] = 0.40 * market_x_component + 0.35 * market_y_component + 0.25 * market_trend_component

    score = out["market_regime_score"]
    regime = pd.Series(0, index=out.index, dtype="int8")
    regime.loc[score.abs() <= 0.10] = 3
    regime.loc[score >= 0.30] = 1
    regime.loc[score <= -0.30] = 2
    recovery = score.notna() & (score > 0) & (out["market_x"] < -0.5) & (out["market_y"] > 0.5)
    regime.loc[recovery] = 5
    vol_thr = out["market_volatility"].rolling(
        params.market_regime_vol_window,
        min_periods=max(20, params.market_regime_vol_window // 2),
    ).quantile(params.market_regime_vol_quantile)
    high_vol = out["market_volatility"].notna() & vol_thr.notna() & (out["market_volatility"] > 1.5) & (out["market_volatility"] > vol_thr)
    regime.loc[high_vol & regime.isin([0, 3])] = 4
    out["market_regime"] = regime.astype("int8")

    if "dxy_inverse_close" in out.columns:
        dxy_inv = pd.to_numeric(out["dxy_inverse_close"], errors="coerce")
        out["dxy_inverse_return_1"] = np.log(safe_div(dxy_inv, dxy_inv.shift(1))).replace([np.inf, -np.inf], np.nan)
        out["dxy_inverse_return_4"] = np.log(safe_div(dxy_inv, dxy_inv.shift(4))).replace([np.inf, -np.inf], np.nan)
        out["dxy_inverse_zscore_20"] = calc_true_zscore(dxy_inv, 20)
    else:
        out["dxy_inverse_return_1"] = np.nan
        out["dxy_inverse_return_4"] = np.nan
        out["dxy_inverse_zscore_20"] = np.nan
    return out


def add_session_context_features(out: pd.DataFrame, params: FeatureParams, config: Mapping[str, Any]) -> pd.DataFrame:
    out = out.copy()
    dt = pd.to_datetime(out["date"], errors="coerce")
    out["trade_date"] = dt.dt.date.astype(str)
    minutes = dt.dt.hour * 60 + dt.dt.minute
    out["minutes_of_day"] = minutes.astype(float)

    blocks = config.get("sessions", {}).get("blocks", [
        {"name": "asia_pre_london", "id": 1, "start": "00:00", "end": "07:00"},
        {"name": "london_open", "id": 2, "start": "07:00", "end": "09:00"},
        {"name": "london_mid", "id": 3, "start": "09:00", "end": "12:00"},
        {"name": "overlap_early", "id": 4, "start": "12:00", "end": "14:00"},
        {"name": "overlap_late", "id": 5, "start": "14:00", "end": "16:00"},
        {"name": "ny_late", "id": 6, "start": "16:00", "end": "21:00"},
    ])

    block = pd.Series("outside", index=out.index, dtype="object")
    block_id = pd.Series(0, index=out.index, dtype="int8")
    for spec in blocks:
        start = _time_to_minutes(str(spec["start"]))
        end = _time_to_minutes(str(spec["end"]))
        mask = _time_in_minutes(minutes, start, end)
        block.loc[mask] = str(spec["name"])
        block_id.loc[mask] = int(spec["id"])
    out["session_block"] = block
    out["session_block_id"] = block_id.astype("int8")
    out["session_code"] = out["session_block_id"].astype("int8")

    grouped = out.groupby("trade_date", sort=False)
    out["day_open"] = grouped["open"].transform("first")
    out["day_high_so_far"] = grouped["high"].cummax()
    out["day_low_so_far"] = grouped["low"].cummin()
    out["day_return_from_open_pips"] = safe_div(out["close"] - out["day_open"], params.pip_size)
    out["day_range_so_far_pips"] = safe_div(out["day_high_so_far"] - out["day_low_so_far"], params.pip_size)
    out["distance_from_day_high_so_far_pips"] = safe_div(out["close"] - out["day_high_so_far"], params.pip_size)
    out["distance_from_day_low_so_far_pips"] = safe_div(out["close"] - out["day_low_so_far"], params.pip_size)
    out["day_position_pct"] = safe_div(out["close"] - out["day_low_so_far"], out["day_high_so_far"] - out["day_low_so_far"])

    block_table = out.groupby(["trade_date", "session_block_id"], as_index=False).agg(
        session_open=("open", "first"),
        session_high=("high", "max"),
        session_low=("low", "min"),
        session_close=("close", "last"),
        session_bars=("date", "count"),
    ).sort_values(["trade_date", "session_block_id"])
    block_table["prev_session_open"] = block_table.groupby("trade_date")["session_open"].shift(1)
    block_table["prev_session_high"] = block_table.groupby("trade_date")["session_high"].shift(1)
    block_table["prev_session_low"] = block_table.groupby("trade_date")["session_low"].shift(1)
    block_table["prev_session_close"] = block_table.groupby("trade_date")["session_close"].shift(1)
    block_table["prev_session_bars"] = block_table.groupby("trade_date")["session_bars"].shift(1)
    keep = ["trade_date", "session_block_id", "prev_session_open", "prev_session_high", "prev_session_low", "prev_session_close", "prev_session_bars"]
    out = out.merge(block_table[keep], on=["trade_date", "session_block_id"], how="left")
    out["prev_session_return_pips"] = safe_div(out["prev_session_close"] - out["prev_session_open"], params.pip_size)
    out["prev_session_range_pips"] = safe_div(out["prev_session_high"] - out["prev_session_low"], params.pip_size)
    prev_ret = pd.to_numeric(out["prev_session_return_pips"], errors="coerce")
    out["prev_session_bias"] = np.select(
        [prev_ret >= params.prev_session_bias_pips, prev_ret <= -params.prev_session_bias_pips],
        [1, -1],
        default=0,
    ).astype("int8")

    day_ret = pd.to_numeric(out["day_return_from_open_pips"], errors="coerce")
    out["day_bias"] = np.select(
        [day_ret >= params.day_bias_pips, day_ret <= -params.day_bias_pips],
        [1, -1],
        default=0,
    ).astype("int8")
    return out


def add_h1_trend_context(out: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    out = out.copy()
    dt = pd.to_datetime(out["date"], errors="coerce")
    hour = dt.dt.floor("h")
    h1 = out.assign(_h1_hour=hour).groupby("_h1_hour", as_index=True).agg(
        h1_open=("open", "first"),
        h1_high=("high", "max"),
        h1_low=("low", "min"),
        h1_close=("close", "last"),
    ).sort_index()
    close_h1 = pd.to_numeric(h1["h1_close"], errors="coerce")
    h1["h1_return_1h_pips"] = safe_div(close_h1 - close_h1.shift(1), params.pip_size)
    h1["h1_return_4h_pips"] = safe_div(close_h1 - close_h1.shift(4), params.pip_size)
    h1["h1_range_pips"] = safe_div(h1["h1_high"] - h1["h1_low"], params.pip_size)
    h1["h1_ema_fast"] = close_h1.ewm(span=8, adjust=False, min_periods=8).mean()
    h1["h1_ema_slow"] = close_h1.ewm(span=21, adjust=False, min_periods=21).mean()
    h1["h1_ema_spread_pips"] = safe_div(h1["h1_ema_fast"] - h1["h1_ema_slow"], params.pip_size)
    h1["h1_ema_fast_slope_3h_pips"] = safe_div(h1["h1_ema_fast"] - h1["h1_ema_fast"].shift(3), params.pip_size * 3.0)
    spread = pd.to_numeric(h1["h1_ema_spread_pips"], errors="coerce")
    slope = pd.to_numeric(h1["h1_ema_fast_slope_3h_pips"], errors="coerce")
    thr = params.h1_trend_spread_threshold_pips
    h1["h1_trend_direction"] = np.select(
        [(spread > thr) & (slope > 0.0), (spread < -thr) & (slope < 0.0)],
        [1, -1],
        default=0,
    ).astype("int8")
    h1_features = h1[[
        "h1_return_1h_pips", "h1_return_4h_pips", "h1_range_pips",
        "h1_ema_spread_pips", "h1_ema_fast_slope_3h_pips", "h1_trend_direction",
    ]].shift(1).reset_index().rename(columns={"_h1_hour": "_bar_hour"})
    out["_bar_hour"] = hour
    out = out.merge(h1_features, on="_bar_hour", how="left").drop(columns=["_bar_hour"])
    return out


def _merge_shifted_daily_features(out: pd.DataFrame, daily: pd.DataFrame, prefix: str, value_cols: list[str]) -> pd.DataFrame:
    shifted = daily.copy().sort_values("trade_date")
    for col in value_cols:
        shifted[f"{prefix}_{col}"] = shifted[col].shift(1)
    keep = ["trade_date"] + [f"{prefix}_{col}" for col in value_cols]
    return out.merge(shifted[keep], on="trade_date", how="left")


def add_prior_day_context(out: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    out = out.copy()
    daily = out.groupby("trade_date", as_index=False).agg(
        high=("high", "max"),
        low=("low", "min"),
        open=("open", "first"),
        close=("close", "last"),
    ).sort_values("trade_date")
    daily["range_pips"] = safe_div(daily["high"] - daily["low"], params.pip_size)
    daily["return_pips"] = safe_div(daily["close"] - daily["open"], params.pip_size)
    out = _merge_shifted_daily_features(out, daily, "prev_day", ["high", "low", "open", "close", "range_pips", "return_pips"])
    close = pd.to_numeric(out["close"], errors="coerce")
    out["distance_from_prev_day_high_pips"] = safe_div(close - out["prev_day_high"], params.pip_size)
    out["distance_from_prev_day_low_pips"] = safe_div(close - out["prev_day_low"], params.pip_size)
    out["prev_day_position_pct"] = safe_div(close - out["prev_day_low"], out["prev_day_high"] - out["prev_day_low"])
    return out


def add_named_session_features(out: pd.DataFrame, params: FeatureParams, config: Mapping[str, Any]) -> pd.DataFrame:
    """Completed named-session features + current-session so-far features.

    Completed session features are only available after the configured end time.
    This avoids leaking final session high/low into bars inside that same session.
    """
    out = out.copy()
    minutes = pd.to_numeric(out["minutes_of_day"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    blocks = config.get("sessions", {}).get("blocks", [])

    current_hi = out.groupby(["trade_date", "session_block_id"], sort=False)["high"].cummax()
    current_lo = out.groupby(["trade_date", "session_block_id"], sort=False)["low"].cummin()
    current_open = out.groupby(["trade_date", "session_block_id"], sort=False)["open"].transform("first")
    out["current_session_high_so_far"] = current_hi
    out["current_session_low_so_far"] = current_lo
    out["current_session_range_so_far_pips"] = safe_div(current_hi - current_lo, params.pip_size)
    out["current_session_return_so_far_pips"] = safe_div(close - current_open, params.pip_size)
    out["distance_from_current_session_high_so_far_pips"] = safe_div(close - current_hi, params.pip_size)
    out["distance_from_current_session_low_so_far_pips"] = safe_div(close - current_lo, params.pip_size)

    for spec in blocks:
        name = str(spec["name"])
        sid = int(spec["id"])
        end_min = _time_to_minutes(str(spec["end"]))
        prefix = name
        mask = out["session_block_id"].eq(sid)
        sess = out.loc[mask].groupby("trade_date", as_index=False).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            bars=("date", "count"),
        ).sort_values("trade_date")
        if sess.empty:
            for col in [
                f"{prefix}_range_pips", f"{prefix}_return_pips", f"{prefix}_range_percentile_60d",
                f"distance_from_{prefix}_high_pips", f"distance_from_{prefix}_low_pips",
            ]:
                out[col] = np.nan
            continue
        sess[f"{prefix}_open"] = sess["open"]
        sess[f"{prefix}_high"] = sess["high"]
        sess[f"{prefix}_low"] = sess["low"]
        sess[f"{prefix}_close"] = sess["close"]
        sess[f"{prefix}_bars"] = sess["bars"]
        sess[f"{prefix}_range_pips"] = safe_div(sess["high"] - sess["low"], params.pip_size)
        sess[f"{prefix}_return_pips"] = safe_div(sess["close"] - sess["open"], params.pip_size)
        sess[f"{prefix}_range_percentile_60d"] = _rolling_percentile_against_past(sess[f"{prefix}_range_pips"], 60, 20).to_numpy()
        keep = ["trade_date", f"{prefix}_open", f"{prefix}_high", f"{prefix}_low", f"{prefix}_close", f"{prefix}_bars", f"{prefix}_range_pips", f"{prefix}_return_pips", f"{prefix}_range_percentile_60d"]
        out = out.merge(sess[keep], on="trade_date", how="left")
        available = minutes.ge(end_min) if end_min > 0 else pd.Series(True, index=out.index)
        for col in keep[1:]:
            out.loc[~available, col] = np.nan
        out[f"distance_from_{prefix}_high_pips"] = safe_div(close - out[f"{prefix}_high"], params.pip_size)
        out[f"distance_from_{prefix}_low_pips"] = safe_div(close - out[f"{prefix}_low"], params.pip_size)

    # Backward-compatible aliases used by earlier rule research.
    if "asia_pre_london_range_pips" in out.columns:
        out["asia_range_pips"] = out["asia_pre_london_range_pips"]
        out["asia_return_pips"] = out["asia_pre_london_return_pips"]
        out["asia_range_percentile_60d"] = out["asia_pre_london_range_percentile_60d"]
        out["asia_high"] = out.get("asia_pre_london_high", np.nan)
        out["asia_low"] = out.get("asia_pre_london_low", np.nan)
        out["distance_from_asia_high_pips"] = out.get("distance_from_asia_pre_london_high_pips", np.nan)
        out["distance_from_asia_low_pips"] = out.get("distance_from_asia_pre_london_low_pips", np.nan)
    return out


def add_london_range_expansion_context(out: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    out = out.copy()
    minutes = pd.to_numeric(out.get("minutes_of_day"), errors="coerce")
    london_active_or_after = minutes.ge(7 * 60)
    london_regular = minutes.ge(7 * 60) & minutes.lt(16 * 60)
    hi = pd.to_numeric(out["high"], errors="coerce").where(london_regular)
    lo = pd.to_numeric(out["low"], errors="coerce").where(london_regular)
    out["london_high_so_far"] = hi.groupby(out["trade_date"], sort=False).cummax().groupby(out["trade_date"], sort=False).ffill()
    out["london_low_so_far"] = lo.groupby(out["trade_date"], sort=False).cummin().groupby(out["trade_date"], sort=False).ffill()
    out.loc[~london_active_or_after, ["london_high_so_far", "london_low_so_far"]] = np.nan
    out["london_range_so_far_pips"] = safe_div(out["london_high_so_far"] - out["london_low_so_far"], params.pip_size)
    out["london_expansion_vs_asia"] = safe_div(out["london_range_so_far_pips"], out.get("asia_range_pips", np.nan))
    close = pd.to_numeric(out["close"], errors="coerce")
    out["london_close_vs_asia_high_pips"] = safe_div(close - out.get("asia_high", np.nan), params.pip_size)
    out["london_close_vs_asia_low_pips"] = safe_div(close - out.get("asia_low", np.nan), params.pip_size)
    return out


def add_volatility_regime_context(out: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    out = out.copy()
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    out["true_range_pips"] = safe_div(true_range(high, low, close), params.pip_size)
    out["candle_range_pips"] = safe_div(high - low, params.pip_size)
    out["atr_percentile_240"] = _rolling_percentile_against_past(out["atr_14_pips"], params.atr_percentile_window, max(80, params.atr_percentile_window // 3))
    atr_pct = pd.to_numeric(out["atr_percentile_240"], errors="coerce")
    out["atr_regime"] = np.select(
        [atr_pct >= 0.90, atr_pct >= 0.75, atr_pct <= 0.25],
        [4, 3, 1],
        default=2,
    ).astype("int8")
    out.loc[atr_pct.isna(), "atr_regime"] = 0

    parts = []
    for _, sub in out.groupby("session_block_id", sort=False):
        pct = _rolling_percentile_against_past(sub["true_range_pips"], params.session_volatility_window, max(50, params.session_volatility_window // 5))
        parts.append(pd.Series(pct.to_numpy(), index=sub.index))
    out["session_volatility_percentile_240"] = pd.concat(parts).sort_index() if parts else np.nan
    out["bar_volatility_vs_atr"] = safe_div(out["true_range_pips"], out["atr_14_pips"])
    return out


def add_time_of_week_context(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    dt = pd.to_datetime(out["date"], errors="coerce")
    out["day_of_week"] = dt.dt.dayofweek.astype("int8")
    out["hour_of_day"] = dt.dt.hour.astype("int8")
    out["is_monday"] = out["day_of_week"].eq(0).astype("int8")
    out["is_friday"] = out["day_of_week"].eq(4).astype("int8")
    out["is_week_start"] = out["day_of_week"].isin([0, 1]).astype("int8")
    out["is_week_end"] = out["day_of_week"].isin([3, 4]).astype("int8")
    out["week_part"] = np.select(
        [out["day_of_week"].eq(0), out["day_of_week"].between(1, 3), out["day_of_week"].eq(4)],
        [1, 2, 3],
        default=0,
    ).astype("int8")
    return out


# -----------------------------
# Model feature projection
# -----------------------------

DEFAULT_ID_COLUMNS = [
    "date", "symbol", "open", "high", "low", "close", "tick_volume", "spread", "real_volume",
    "dxy_close", "dxy_inverse_close", "index_close",
]


def add_lag_features(df: pd.DataFrame, lags: Mapping[str, list[int]]) -> pd.DataFrame:
    out = df.copy()
    for col, lag_list in (lags or {}).items():
        if col not in out.columns:
            continue
        for lag in lag_list:
            lag_i = int(lag)
            out[f"{col}_lag{lag_i}"] = out[col].shift(lag_i)
    return out


def build_model_feature_frame(full: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    model_cfg = config.get("model_features", {})
    base_features = list(model_cfg.get("base_features", []))
    lags = model_cfg.get("lags", {})
    max_features = int(model_cfg.get("max_features_with_lags", 100))
    id_cols = [c for c in DEFAULT_ID_COLUMNS if c in full.columns]

    missing = [c for c in base_features if c not in full.columns]
    if missing:
        raise ValueError(f"Configured model features are missing from full feature frame: {missing}")

    model = full[id_cols + base_features].copy()
    model = add_lag_features(model, lags)
    feature_cols = [c for c in model.columns if c not in id_cols]
    if len(feature_cols) > max_features:
        raise ValueError(
            f"Model feature count is {len(feature_cols)}, above max_features_with_lags={max_features}. "
            "Reduce base_features or lags in features config."
        )
    return model.replace([np.inf, -np.inf], np.nan)


# -----------------------------
# Public API
# -----------------------------

FULL_RESEARCH_COLUMNS = [
    "distance_from_confirmed_zigzag_high_pips",
    "distance_from_confirmed_zigzag_low_pips",
    "fracdiff_price",
    "log_return",
    "return_1bar_pips",
    "return_4bar_pips",
    "atr_14_pips",
    "volume_ratio",
    "rsi",
    "rsi_slope",
    "gmma",
    "gmma_green",
    "gmma_red",
    "gmma_distance",
    "gmma_distance_slope",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "market_x",
    "market_y",
    "market_quadrant",
    "market_regime",
    "session_code",
    "session_block_id",
    "day_return_from_open_pips",
    "day_range_so_far_pips",
    "distance_from_day_high_so_far_pips",
    "distance_from_day_low_so_far_pips",
    "day_position_pct",
    "prev_session_return_pips",
    "prev_session_range_pips",
    "prev_session_return_to_atr",
    "prev_session_bias",
    "day_return_to_atr",
    "day_bias",
    "h1_return_1h_pips",
    "h1_return_4h_pips",
    "h1_range_pips",
    "h1_ema_spread_pips",
    "h1_ema_fast_slope_3h_pips",
    "h1_trend_direction",
    "prev_day_range_pips",
    "prev_day_return_pips",
    "distance_from_prev_day_high_pips",
    "distance_from_prev_day_low_pips",
    "prev_day_position_pct",
    "asia_range_pips",
    "asia_return_pips",
    "asia_range_percentile_60d",
    "distance_from_asia_high_pips",
    "distance_from_asia_low_pips",
    "london_range_so_far_pips",
    "london_expansion_vs_asia",
    "london_close_vs_asia_high_pips",
    "london_close_vs_asia_low_pips",
    "true_range_pips",
    "candle_range_pips",
    "atr_percentile_240",
    "atr_regime",
    "session_volatility_percentile_240",
    "bar_volatility_vs_atr",
    "day_of_week",
    "hour_of_day",
    "is_monday",
    "is_friday",
    "is_week_start",
    "is_week_end",
    "week_part",
    "spread_pips",
    "dxy_inverse_return_1",
    "dxy_inverse_return_4",
    "dxy_inverse_zscore_20",
    "current_session_range_so_far_pips",
    "current_session_return_so_far_pips",
    "distance_from_current_session_high_so_far_pips",
    "distance_from_current_session_low_so_far_pips",
]


def build_features(df: pd.DataFrame, *, symbol: str, config: Mapping[str, Any], pip_size: float | None = None) -> pd.DataFrame:
    """Build full causal feature frame for one symbol.

    ``pip_size`` is accepted only for backward compatibility; the symbol-aware
    config is the source of truth when both are supplied.
    """
    params = feature_params_from_config(config, symbol)
    out = validate_ohlcv(df, symbol=symbol)
    out["volume"] = pd.to_numeric(out.get("tick_volume", 0.0), errors="coerce").fillna(0.0)

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_ = out["open"].astype(float)
    volume = out["volume"].astype(float)

    out = add_confirmed_zigzag_features(out, params)
    out["log_return"] = np.log(safe_div(close, close.shift(1))).replace([np.inf, -np.inf], np.nan)
    out["return_1bar_pips"] = safe_div(close - close.shift(1), params.pip_size)
    out["return_4bar_pips"] = safe_div(close - close.shift(4), params.pip_size)
    out["fracdiff_price"] = fractional_difference(close, d=params.fracdiff_order, window=params.fracdiff_window)

    atr_price = calculate_atr(high, low, close, params.atr_period)
    out["atr_14_pips"] = safe_div(atr_price, params.pip_size)
    out["volume_ratio"] = safe_div(volume, volume.rolling(params.volume_ratio_window, min_periods=params.volume_ratio_window).mean())
    out["rsi"] = calculate_rsi(close, params.rsi_period)
    out["rsi_slope"] = calc_slope(out["rsi"], params.rsi_slope_window)

    out = add_gmma_features(out, close, params)

    candle_range = high - low
    out["body_ratio"] = safe_div(close - open_, candle_range)
    out["upper_wick_ratio"] = safe_div(high - pd.concat([open_, close], axis=1).max(axis=1), candle_range)
    out["lower_wick_ratio"] = safe_div(pd.concat([open_, close], axis=1).min(axis=1) - low, candle_range)
    for c in ["body_ratio", "upper_wick_ratio", "lower_wick_ratio"]:
        out.loc[candle_range.abs() < 1e-12, c] = 0.0

    out = add_market_features(out, close, params)
    out = add_session_context_features(out, params, config)
    out = add_h1_trend_context(out, params)
    out = add_prior_day_context(out, params)
    out = add_named_session_features(out, params, config)
    out = add_london_range_expansion_context(out, params)
    out = add_volatility_regime_context(out, params)
    out = add_time_of_week_context(out)

    out["prev_session_return_to_atr"] = safe_div(out["prev_session_return_pips"], out["atr_14_pips"])
    out["day_return_to_atr"] = safe_div(out["day_return_from_open_pips"], out["atr_14_pips"])
    out["spread_pips"] = safe_div(pd.to_numeric(out["spread"], errors="coerce") * params.point_size, params.pip_size)

    return out.replace([np.inf, -np.inf], np.nan)


def build_feature_outputs(df: pd.DataFrame, *, symbol: str, config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = build_features(df, symbol=symbol, config=config)
    model = build_model_feature_frame(full, config)
    return full, model


def clean_feature_columns() -> list[str]:
    return list(FULL_RESEARCH_COLUMNS)
