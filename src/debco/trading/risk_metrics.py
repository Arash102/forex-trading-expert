from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrawdownStats:
    max_drawdown: float
    max_drawdown_duration_periods: int
    peak_index: int
    trough_index: int
    recovery_index: int | None


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def profit_factor(pnl: Iterable[float]) -> float:
    arr = _as_float_array(pnl)
    gross_profit = float(arr[arr > 0].sum())
    gross_loss = float(-arr[arr < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else float("nan")
    return gross_profit / gross_loss


def payoff_ratio(pnl: Iterable[float]) -> float:
    arr = _as_float_array(pnl)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    if len(wins) == 0 or len(losses) == 0:
        return float("nan")
    avg_win = float(wins.mean())
    avg_loss = float(-losses.mean())
    return avg_win / avg_loss if avg_loss else float("inf")


def max_drawdown_stats(equity: Iterable[float]) -> DrawdownStats:
    eq = np.asarray(list(equity), dtype=float)
    if len(eq) == 0:
        return DrawdownStats(float("nan"), 0, -1, -1, None)
    peaks = np.maximum.accumulate(eq)
    dd = peaks - eq
    trough = int(np.nanargmax(dd))
    max_dd = float(dd[trough])
    peak = int(np.where(eq[: trough + 1] == peaks[trough])[0][0])
    recovery = None
    for i in range(trough + 1, len(eq)):
        if eq[i] >= peaks[trough]:
            recovery = int(i)
            break
    duration = (len(eq) - 1 - peak) if recovery is None else (recovery - peak)
    return DrawdownStats(max_dd, int(duration), peak, trough, recovery)


def longest_losing_streak(pnl: Iterable[float]) -> int:
    arr = np.asarray(list(pnl), dtype=float)
    best = cur = 0
    for v in arr:
        if np.isfinite(v) and v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def risk_of_ruin_monte_carlo(
    r_multiples: Iterable[float],
    *,
    risk_per_trade: float,
    ruin_drawdown: float,
    n_sims: int = 5000,
    random_seed: int = 42,
) -> float:
    """Estimate probability of hitting a drawdown limit by bootstrapping trades.

    `r_multiples` are trade outcomes in R. If risk_per_trade is 0.02, a -1R
    trade loses 2% of current equity and a +2R trade gains 4%.
    """
    r = _as_float_array(r_multiples)
    if len(r) == 0 or int(n_sims) <= 0:
        return float("nan")
    if risk_per_trade <= 0:
        return float("nan")
    rng = np.random.default_rng(int(random_seed))
    ruin_count = 0
    n = len(r)
    for _ in range(int(n_sims)):
        sample = rng.choice(r, size=n, replace=True)
        equity = 1.0
        peak = 1.0
        ruined = False
        for outcome_r in sample:
            equity *= max(0.0, 1.0 + float(outcome_r) * float(risk_per_trade))
            peak = max(peak, equity)
            dd = 1.0 - (equity / peak if peak else 0.0)
            if dd >= float(ruin_drawdown) or equity <= 0:
                ruined = True
                break
        ruin_count += int(ruined)
    return float(ruin_count / int(n_sims))


def summarize_trade_pnl(
    trades: pd.DataFrame,
    *,
    pnl_col: str = "pnl_pips",
    r_col: str = "pnl_R",
    date_col: str = "date",
    initial_capital: float = 1000.0,
    risk_per_trade: float = 0.02,
    ruin_drawdowns: list[float] | None = None,
    n_ruin_sims: int = 5000,
    random_seed: int = 42,
) -> dict[str, float]:
    ruin_drawdowns = ruin_drawdowns or [0.25, 0.30]
    if trades.empty:
        out = {
            "trade_count": 0.0,
            "win_rate": float("nan"),
            "label_tp_rate": float("nan"),
            "avg_win_pips": float("nan"),
            "avg_loss_pips": float("nan"),
            "payoff_ratio": float("nan"),
            "profit_factor": float("nan"),
            "avg_win_R": float("nan"),
            "avg_loss_R": float("nan"),
            "payoff_ratio_R": float("nan"),
            "profit_factor_R": float("nan"),
            "gross_profit_R": 0.0,
            "gross_loss_R": 0.0,
            "expectancy_pips": float("nan"),
            "expectancy_R": float("nan"),
            "net_pips": 0.0,
            "net_R": 0.0,
            "net_dollars": 0.0,
            "max_drawdown_pips": 0.0,
            "max_drawdown_R": 0.0,
            "max_drawdown_pct": 0.0,
            "drawdown_duration_trades": 0.0,
            "drawdown_duration_days": float("nan"),
            "drawdown_duration_trades_R": 0.0,
            "drawdown_duration_days_R": float("nan"),
            "longest_losing_streak": 0.0,
            "trades_per_month": 0.0,
            "risk_per_trade_pct": float(risk_per_trade * 100.0),
            "initial_capital": float(initial_capital),
        }
        for dd in ruin_drawdowns:
            out[f"risk_of_ruin_dd_{int(dd * 100)}pct"] = float("nan")
        return out

    df = trades.copy()
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
    r = pd.to_numeric(df[r_col], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else float("nan"))
    avg_win = float(wins.mean()) if len(wins) else float("nan")
    avg_loss = float(-losses.mean()) if len(losses) else float("nan")
    pr = avg_win / avg_loss if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss > 0 else float("nan")

    r_wins = r[r > 0]
    r_losses = r[r < 0]
    gross_profit_R = float(r_wins.sum())
    gross_loss_R = float(-r_losses.sum())
    profit_factor_R = gross_profit_R / gross_loss_R if gross_loss_R > 0 else (float("inf") if gross_profit_R > 0 else float("nan"))
    avg_win_R = float(r_wins.mean()) if len(r_wins) else float("nan")
    avg_loss_R = float(-r_losses.mean()) if len(r_losses) else float("nan")
    payoff_ratio_R = avg_win_R / avg_loss_R if np.isfinite(avg_win_R) and np.isfinite(avg_loss_R) and avg_loss_R > 0 else float("nan")

    equity_pips = pnl.cumsum().to_numpy(dtype=float)
    equity_r = r.cumsum().to_numpy(dtype=float)
    equity_dollars = float(initial_capital) + (r * float(initial_capital) * float(risk_per_trade)).cumsum().to_numpy(dtype=float)
    dd_pips = max_drawdown_stats(equity_pips)
    dd_r = max_drawdown_stats(equity_r)
    dd_dollars = max_drawdown_stats(equity_dollars)

    duration_days = float("nan")
    duration_days_R = float("nan")
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        if dd_pips.peak_index >= 0 and dd_pips.trough_index >= 0:
            end_idx = dd_pips.recovery_index if dd_pips.recovery_index is not None else len(df) - 1
            if pd.notna(dates.iloc[dd_pips.peak_index]) and pd.notna(dates.iloc[end_idx]):
                duration_days = float((dates.iloc[end_idx] - dates.iloc[dd_pips.peak_index]).total_seconds() / 86400.0)
        if dd_dollars.peak_index >= 0 and dd_dollars.trough_index >= 0:
            end_idx_r = dd_dollars.recovery_index if dd_dollars.recovery_index is not None else len(df) - 1
            if pd.notna(dates.iloc[dd_dollars.peak_index]) and pd.notna(dates.iloc[end_idx_r]):
                duration_days_R = float((dates.iloc[end_idx_r] - dates.iloc[dd_dollars.peak_index]).total_seconds() / 86400.0)

    months = float("nan")
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if len(dates) >= 2:
            months = max(float((dates.max() - dates.min()).days) / 30.4375, 1e-9)

    out = {
        "trade_count": float(len(df)),
        "win_rate": float((pnl > 0).mean()),
        "label_tp_rate": float((pd.to_numeric(df.get("y_true", pd.Series(index=df.index, dtype=float)), errors="coerce") == 1).mean()) if "y_true" in df.columns else float("nan"),
        "avg_win_pips": avg_win,
        "avg_loss_pips": avg_loss,
        "payoff_ratio": pr,
        "profit_factor": pf,
        "gross_profit_pips": gross_profit,
        "gross_loss_pips": gross_loss,
        "avg_win_R": avg_win_R,
        "avg_loss_R": avg_loss_R,
        "payoff_ratio_R": payoff_ratio_R,
        "profit_factor_R": profit_factor_R,
        "gross_profit_R": gross_profit_R,
        "gross_loss_R": gross_loss_R,
        "expectancy_pips": float(pnl.mean()),
        "expectancy_R": float(r.mean()),
        "net_pips": float(pnl.sum()),
        "net_R": float(r.sum()),
        "net_dollars": float(r.sum() * float(initial_capital) * float(risk_per_trade)),
        "return_pct_on_initial": float(r.sum() * float(risk_per_trade) * 100.0),
        "max_drawdown_pips": float(dd_pips.max_drawdown),
        "max_drawdown_R": float(dd_r.max_drawdown),
        "max_drawdown_dollars": float(dd_dollars.max_drawdown),
        "max_drawdown_pct": float(dd_dollars.max_drawdown / float(initial_capital)) if initial_capital else float("nan"),
        "drawdown_duration_trades": float(dd_pips.max_drawdown_duration_periods),
        "drawdown_duration_days": duration_days,
        "drawdown_duration_trades_R": float(dd_dollars.max_drawdown_duration_periods),
        "drawdown_duration_days_R": duration_days_R,
        "longest_losing_streak": float(longest_losing_streak(pnl)),
        "trades_per_month": float(len(df) / months) if np.isfinite(months) else float("nan"),
        "risk_per_trade_pct": float(risk_per_trade * 100.0),
        "initial_capital": float(initial_capital),
    }
    for dd in ruin_drawdowns:
        out[f"risk_of_ruin_dd_{int(dd * 100)}pct"] = risk_of_ruin_monte_carlo(
            r,
            risk_per_trade=risk_per_trade,
            ruin_drawdown=dd,
            n_sims=n_ruin_sims,
            random_seed=random_seed,
        )
    return out
