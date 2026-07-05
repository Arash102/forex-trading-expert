from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


DEFAULT_RISK_PER_TRADE_GRID = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]


@dataclass(frozen=True)
class DecisionFilters:
    min_trades: float = 60.0
    min_profit_factor_R: float = 1.50
    max_risk_of_ruin_dd_25pct: float = 0.01
    min_positive_folds: float = 6.0
    require_all_components_loaded: bool = True
    require_side_complete_3x3: bool = False


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _max_drawdown(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    equity = np.concatenate([[0.0], np.cumsum(values)])
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def _drawdown_duration(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    equity = np.concatenate([[0.0], np.cumsum(values)])
    peak = np.maximum.accumulate(equity)
    underwater = equity < peak
    longest = 0
    current = 0
    for flag in underwater:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _profit_factor(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    gross_profit = float(values[values > 0].sum())
    gross_loss = float(-values[values < 0].sum())
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _payoff_ratio(values: np.ndarray) -> float:
    wins = values[values > 0]
    losses = values[values < 0]
    if wins.size == 0 or losses.size == 0:
        return 0.0
    return float(wins.mean() / abs(losses.mean()))


def risk_multiplier_for_trade(row: Mapping[str, Any], plan: Mapping[str, Any]) -> float:
    multiplier = 1.0
    symbol = str(row.get("symbol", "")).upper()
    side = str(row.get("side", "")).lower()
    component = str(row.get("component_id", row.get("setup_id", "")))

    symbol_weights = plan.get("symbol_weights", {}) or {}
    side_weights = plan.get("side_weights", {}) or {}
    symbol_side_weights = plan.get("symbol_side_weights", {}) or {}
    component_weights = plan.get("component_weights", {}) or {}

    if symbol in symbol_weights:
        multiplier *= _to_float(symbol_weights[symbol], 1.0)
    if side in side_weights:
        multiplier *= _to_float(side_weights[side], 1.0)
    ss_key = f"{symbol}|{side}"
    if ss_key in symbol_side_weights:
        multiplier *= _to_float(symbol_side_weights[ss_key], 1.0)
    if component in component_weights:
        multiplier *= _to_float(component_weights[component], 1.0)
    return max(0.0, float(multiplier))


def add_risk_multipliers(trades: pd.DataFrame, plan: Mapping[str, Any]) -> pd.DataFrame:
    out = trades.copy()
    out["risk_multiplier"] = [risk_multiplier_for_trade(row, plan) for row in out.to_dict("records")]
    out["weighted_pnl_R"] = pd.to_numeric(out["pnl_R"], errors="coerce").fillna(0.0) * out["risk_multiplier"]
    return out


def bootstrap_risk_of_ruin(
    returns_pct: np.ndarray,
    *,
    threshold: float,
    simulations: int,
    seed: int,
) -> float:
    if returns_pct.size == 0 or simulations <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    n = returns_pct.size
    ruin = 0
    for _ in range(int(simulations)):
        sample = rng.choice(returns_pct, size=n, replace=True)
        if _max_drawdown(sample) >= float(threshold):
            ruin += 1
    return float(ruin / int(simulations))


def summarize_weighted_trades(
    trades: pd.DataFrame,
    *,
    portfolio: str,
    risk_policy: str,
    risk_plan: Mapping[str, Any],
    risk_per_trade: float,
    initial_capital: float,
    ruin_drawdowns: list[float],
    simulations: int,
    seed: int,
    source_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if trades.empty:
        return {
            "portfolio": portfolio,
            "risk_policy": risk_policy,
            "risk_plan": str(risk_plan.get("name", "risk_plan")),
            "risk_per_trade_pct": float(risk_per_trade),
            "trade_count": 0.0,
            "profit_factor_R": 0.0,
            "net_R_weighted": 0.0,
            "portfolio_pass": False,
        }

    df = add_risk_multipliers(trades, risk_plan)
    if "entry_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
        df = df.sort_values(["entry_date", "component_id"], kind="mergesort")

    weighted_r = pd.to_numeric(df["weighted_pnl_R"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    returns_pct = weighted_r * float(risk_per_trade)
    pnl_dollars = returns_pct * float(initial_capital)

    out: dict[str, Any] = {
        "portfolio": portfolio,
        "risk_policy": risk_policy,
        "risk_plan": str(risk_plan.get("name", "risk_plan")),
        "risk_plan_description": str(risk_plan.get("description", "")),
        "risk_per_trade_pct": float(risk_per_trade),
        "initial_capital": float(initial_capital),
        "trade_count": float(len(df)),
        "win_rate": float((weighted_r > 0).mean()) if len(weighted_r) else 0.0,
        "payoff_ratio_R": _payoff_ratio(weighted_r),
        "profit_factor_R": _profit_factor(weighted_r),
        "gross_profit_R": float(weighted_r[weighted_r > 0].sum()),
        "gross_loss_R": float(-weighted_r[weighted_r < 0].sum()),
        "expectancy_R_weighted": float(weighted_r.mean()) if len(weighted_r) else 0.0,
        "net_R_weighted": float(weighted_r.sum()),
        "net_return_pct_on_initial": float(returns_pct.sum()),
        "net_dollars": float(pnl_dollars.sum()),
        "max_drawdown_R_weighted": _max_drawdown(weighted_r),
        "max_drawdown_pct": _max_drawdown(returns_pct),
        "max_drawdown_dollars": _max_drawdown(pnl_dollars),
        "drawdown_duration_trades": _drawdown_duration(returns_pct),
        "avg_risk_multiplier": float(df["risk_multiplier"].mean()),
        "min_risk_multiplier": float(df["risk_multiplier"].min()),
        "max_risk_multiplier": float(df["risk_multiplier"].max()),
    }

    for dd in ruin_drawdowns:
        key = f"risk_of_ruin_dd_{int(round(float(dd) * 100))}pct"
        out[key] = bootstrap_risk_of_ruin(
            returns_pct,
            threshold=float(dd),
            simulations=int(simulations),
            seed=int(seed + round(float(risk_per_trade) * 1_000_000) + len(str(risk_plan.get("name", "")))),
        )

    if "fold" in df.columns:
        fold_net = df.groupby("fold", dropna=False)["weighted_pnl_R"].sum()
        out["folds_with_trades"] = float(fold_net.shape[0])
        out["positive_folds"] = float((fold_net > 0).sum())
        out["negative_folds"] = float((fold_net < 0).sum())
        out["worst_fold_net_R_weighted"] = float(fold_net.min()) if len(fold_net) else 0.0
    else:
        out["folds_with_trades"] = 0.0
        out["positive_folds"] = 0.0
        out["negative_folds"] = 0.0
        out["worst_fold_net_R_weighted"] = 0.0

    if source_summary:
        for key in [
            "configured_component_count",
            "loaded_component_count",
            "configured_setup_count",
            "eurusd_long_setup_count",
            "eurusd_short_setup_count",
            "xauusd_long_setup_count",
            "xauusd_short_setup_count",
            "side_complete_3x3_configured",
            "all_components_loaded",
            "candidate_trade_count_before_controls",
            "rejected_trade_count",
        ]:
            if key in source_summary:
                out[key] = source_summary[key]
    return out


def component_risk_summary(trades: pd.DataFrame, plan: Mapping[str, Any]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    df = add_risk_multipliers(trades, plan)
    rows = []
    for component_id, g in df.groupby("component_id", dropna=False):
        w = pd.to_numeric(g["weighted_pnl_R"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        rows.append(
            {
                "risk_plan": str(plan.get("name", "risk_plan")),
                "component_id": component_id,
                "symbol": ";".join(sorted({str(x) for x in g.get("symbol", pd.Series(dtype=str)).dropna().unique()})),
                "side": ";".join(sorted({str(x) for x in g.get("side", pd.Series(dtype=str)).dropna().unique()})),
                "trade_count": float(len(g)),
                "risk_multiplier_mean": float(g["risk_multiplier"].mean()),
                "profit_factor_R_weighted": _profit_factor(w),
                "net_R_weighted": float(w.sum()),
                "max_drawdown_R_weighted": _max_drawdown(w),
                "win_rate": float((w > 0).mean()) if len(w) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["net_R_weighted", "profit_factor_R_weighted"], ascending=[True, True])


def apply_decision_filters(df: pd.DataFrame, filters: DecisionFilters) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "trade_count",
        "profit_factor_R",
        "risk_of_ruin_dd_25pct",
        "positive_folds",
        "loaded_component_count",
        "configured_component_count",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["portfolio_pass"] = (
        (out.get("trade_count", 0) >= float(filters.min_trades))
        & (out.get("profit_factor_R", 0) >= float(filters.min_profit_factor_R))
        & (out.get("risk_of_ruin_dd_25pct", 1) <= float(filters.max_risk_of_ruin_dd_25pct))
        & (out.get("positive_folds", 0) >= float(filters.min_positive_folds))
    )
    if filters.require_all_components_loaded and "loaded_component_count" in out.columns and "configured_component_count" in out.columns:
        out["portfolio_pass"] &= out["loaded_component_count"] == out["configured_component_count"]
    if filters.require_side_complete_3x3 and "side_complete_3x3_configured" in out.columns:
        out["portfolio_pass"] &= out["side_complete_3x3_configured"].astype(str).str.lower().isin(["true", "1"])
    return out


def discover_selected_trade_files(source_dir: str | Path) -> list[Path]:
    root = Path(source_dir)
    return sorted(root.glob("*/**/portfolio_selected_trades.csv"))


def load_source_summary(source_dir: str | Path) -> pd.DataFrame:
    path = Path(source_dir) / "all_inventory_portfolio_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
