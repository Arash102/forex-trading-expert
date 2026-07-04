from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from debco.trading.risk_metrics import summarize_trade_pnl


NO_LIMIT_INT = 10**9
NO_LIMIT_FLOAT = 10**9


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["date", "entry_date", "exit_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _float_equal(series: pd.Series, value: float, *, tol: float = 1e-9) -> pd.Series:
    return np.isclose(pd.to_numeric(series, errors="coerce"), float(value), atol=tol, rtol=0.0)


def _limit_int(value: Any, default: int = NO_LIMIT_INT) -> int:
    if value is None:
        return default
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    return default if iv < 0 else iv


def _limit_float(value: Any, default: float = NO_LIMIT_FLOAT) -> float:
    if value is None:
        return default
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return default
    return default if fv < 0 else fv


def infer_session_label(ts: Any) -> str:
    """Infer a UTC M15 trading session label from an entry timestamp.

    The project feature config uses UTC blocks:
    00-07 Asia/pre-London, 07-09 London open, 09-12 London mid,
    12-14 early overlap, 14-16 late overlap, 16-21 NY late.
    """
    if pd.isna(ts):
        return "unknown"
    t = pd.Timestamp(ts)
    hour = t.hour + t.minute / 60.0
    if 0 <= hour < 7:
        return "asia_pre_london"
    if 7 <= hour < 9:
        return "london_open"
    if 9 <= hour < 12:
        return "london_mid"
    if 12 <= hour < 14:
        return "overlap_early"
    if 14 <= hour < 16:
        return "overlap_late"
    if 16 <= hour < 21:
        return "ny_late"
    return "off_hours"


def load_component_trades(source_root: Path, component: Mapping[str, Any]) -> pd.DataFrame:
    exp = str(component["experiment"])
    job = str(component["job"])
    path = source_root / exp / job / "selected_trades.csv"
    required = bool(component.get("required", True))
    if not path.exists():
        if required:
            raise FileNotFoundError(f"selected_trades.csv not found for component {component.get('component_id')}: {path}")
        return pd.DataFrame()
    df = _parse_dates(pd.read_csv(path))
    mask = pd.Series(True, index=df.index)
    for col in ["policy", "probability_column"]:
        if col in component:
            if col not in df.columns:
                raise ValueError(f"Missing {col} in {path}")
            mask &= df[col].astype(str).eq(str(component[col]))
    if "threshold" in component and component.get("threshold") is not None:
        if "threshold" not in df.columns:
            raise ValueError(f"Missing threshold in {path}")
        mask &= _float_equal(df["threshold"], float(component["threshold"]))
    else:
        if "threshold" in df.columns:
            mask &= df["threshold"].isna()
    if "top_percentile" in component and component.get("top_percentile") is not None:
        if "top_percentile" not in df.columns:
            raise ValueError(f"Missing top_percentile in {path}")
        mask &= _float_equal(df["top_percentile"], float(component["top_percentile"]))
    else:
        if "top_percentile" in df.columns:
            mask &= df["top_percentile"].isna()
    selected = df.loc[mask].copy()
    if selected.empty and not required:
        return selected
    selected["component_id"] = str(component.get("component_id", job))
    selected["component_priority"] = int(component.get("priority", 100))
    selected["component_required"] = required
    selected["source_experiment"] = exp
    selected["source_job"] = job
    selected["strategy_id"] = (
        selected["component_id"].astype(str)
        + "|" + selected.get("policy", "").astype(str)
        + "|" + selected.get("probability_column", "").astype(str)
    )

    rank_col = str(component.get("rank_column", component.get("probability_column", "y_prob_raw")))
    if rank_col in selected.columns:
        selected["portfolio_rank_score"] = pd.to_numeric(selected[rank_col], errors="coerce")
    elif "portfolio_rank_score" not in selected.columns:
        selected["portfolio_rank_score"] = np.nan
    return selected


def load_portfolio_candidate_trades(source_root: Path, components: list[Mapping[str, Any]]) -> pd.DataFrame:
    frames = [load_component_trades(source_root, c) for c in components]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    trades = pd.concat(frames, ignore_index=True, sort=False)
    if trades.empty:
        return trades
    date_col = "entry_date" if "entry_date" in trades.columns else "date"
    sort_cols = [date_col, "component_priority", "portfolio_rank_score"]
    ascending = [True, True, False]
    return trades.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _entry_day(row: pd.Series) -> Any:
    dt = row.get("entry_date", row.get("date"))
    if pd.isna(dt):
        return None
    return pd.Timestamp(dt).date()


def _session_key(row: pd.Series) -> str:
    for col in ["session", "current_session", "session_label", "entry_session"]:
        if col in row.index and pd.notna(row.get(col)):
            return str(row.get(col))
    return infer_session_label(row.get("entry_date", row.get("date")))


def _active_open_trades(accepted: list[pd.Series], entry_time: pd.Timestamp) -> list[pd.Series]:
    active = []
    for r in accepted:
        x = r.get("exit_date")
        if pd.isna(x):
            continue
        if pd.Timestamp(x) > entry_time:
            active.append(r)
    return active


def _same_symbol_active(active: list[pd.Series], symbol: str) -> bool:
    for r in active:
        if str(r.get("symbol", "")).upper() == symbol:
            return True
    return False


def apply_portfolio_risk_controls(
    trades: pd.DataFrame,
    controls: Mapping[str, Any],
    *,
    risk_per_trade: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply live-feasible portfolio risk controls to time-ordered candidate trades.

    Selection is chronological. Only same-timestamp conflicts can be rank-aware via
    component priority and portfolio_rank_score. No future same-day information is used.
    """
    if trades.empty:
        return trades.copy(), pd.DataFrame()
    df = _parse_dates(trades).copy()
    date_col = "entry_date" if "entry_date" in df.columns else "date"
    if date_col not in df.columns:
        raise ValueError("Portfolio trades must include entry_date or date.")
    if "portfolio_rank_score" not in df.columns:
        df["portfolio_rank_score"] = np.nan
    if "component_priority" not in df.columns:
        df["component_priority"] = 100
    df = df.sort_values([date_col, "component_priority", "portfolio_rank_score"], ascending=[True, True, False]).reset_index(drop=True)

    max_open_risk = _limit_float(controls.get("max_open_risk", None))
    max_open_trades = _limit_int(controls.get("max_open_trades", None))
    max_daily_risk = _limit_float(controls.get("max_daily_risk", None))
    max_trades_per_day = _limit_int(controls.get("max_trades_per_day", None))
    max_trades_per_symbol_per_day = _limit_int(controls.get("max_trades_per_symbol_per_day", None))
    max_trades_per_session = _limit_int(controls.get("max_trades_per_session", None))
    max_trades_per_symbol_per_session = _limit_int(controls.get("max_trades_per_symbol_per_session", None))
    stop_after_daily_losses = _limit_int(controls.get("stop_after_daily_losses", None))
    no_opposite_same_symbol_open = bool(controls.get("no_opposite_same_symbol_open", True))
    no_overlap_same_symbol = bool(controls.get("no_overlap_same_symbol", False))
    dedupe_same_symbol_entry = bool(controls.get("dedupe_same_symbol_entry", True))

    accepted_rows: list[pd.Series] = []
    audit_rows: list[dict[str, Any]] = []
    accepted_by_day: dict[Any, int] = {}
    accepted_by_symbol_day: dict[tuple[Any, str], int] = {}
    accepted_by_session_day: dict[tuple[Any, str], int] = {}
    accepted_by_symbol_session_day: dict[tuple[Any, str, str], int] = {}
    used_entry_keys: set[tuple[Any, str]] = set()

    for _, row in df.iterrows():
        entry_time = pd.Timestamp(row[date_col])
        day = _entry_day(row)
        symbol = str(row.get("symbol", row.get("source_job", ""))).upper()
        side = str(row.get("side", "")).lower()
        session = _session_key(row)
        reason = "accepted"
        active = _active_open_trades(accepted_rows, entry_time)
        open_risk = len(active) * float(risk_per_trade)
        day_count = accepted_by_day.get(day, 0)
        symbol_day_count = accepted_by_symbol_day.get((day, symbol), 0)
        session_day_count = accepted_by_session_day.get((day, session), 0)
        symbol_session_count = accepted_by_symbol_session_day.get((day, symbol, session), 0)

        if dedupe_same_symbol_entry:
            key = (entry_time, symbol)
            if key in used_entry_keys:
                reason = "reject_duplicate_same_symbol_entry"
        if reason == "accepted" and day_count >= max_trades_per_day:
            reason = "reject_max_trades_per_day"
        if reason == "accepted" and symbol_day_count >= max_trades_per_symbol_per_day:
            reason = "reject_max_trades_per_symbol_per_day"
        if reason == "accepted" and session_day_count >= max_trades_per_session:
            reason = "reject_max_trades_per_session"
        if reason == "accepted" and symbol_session_count >= max_trades_per_symbol_per_session:
            reason = "reject_max_trades_per_symbol_per_session"
        if reason == "accepted" and (day_count + 1) * float(risk_per_trade) > max_daily_risk + 1e-12:
            reason = "reject_max_daily_risk"
        if reason == "accepted" and len(active) >= max_open_trades:
            reason = "reject_max_open_trades"
        if reason == "accepted" and open_risk + float(risk_per_trade) > max_open_risk + 1e-12:
            reason = "reject_max_open_risk"
        if reason == "accepted" and no_overlap_same_symbol and _same_symbol_active(active, symbol):
            reason = "reject_same_symbol_overlap"
        if reason == "accepted" and no_opposite_same_symbol_open:
            for r in active:
                if str(r.get("symbol", "")).upper() == symbol and str(r.get("side", "")).lower() and str(r.get("side", "")).lower() != side:
                    reason = "reject_opposite_same_symbol_open"
                    break
        if reason == "accepted" and stop_after_daily_losses < NO_LIMIT_INT:
            known_losses = 0
            for r in accepted_rows:
                if _entry_day(r) != day:
                    continue
                if pd.notna(r.get("exit_date")) and pd.Timestamp(r.get("exit_date")) <= entry_time:
                    if float(r.get("pnl_R", 0.0)) < 0:
                        known_losses += 1
            if known_losses >= stop_after_daily_losses:
                reason = "reject_stop_after_daily_losses"

        audit = {
            "accepted": reason == "accepted",
            "reject_reason": reason,
            "entry_date": row.get(date_col),
            "exit_date": row.get("exit_date"),
            "symbol": symbol,
            "side": side,
            "session": session,
            "component_id": row.get("component_id"),
            "source_job": row.get("source_job"),
            "portfolio_rank_score": row.get("portfolio_rank_score"),
            "pnl_R": row.get("pnl_R"),
            "pnl_pips": row.get("pnl_pips"),
            "open_trades_before": len(active),
            "open_risk_before": open_risk,
            "day_trade_count_before": day_count,
            "symbol_day_trade_count_before": symbol_day_count,
            "session_day_trade_count_before": session_day_count,
            "symbol_session_trade_count_before": symbol_session_count,
        }
        audit_rows.append(audit)
        if reason == "accepted":
            accepted_rows.append(row)
            accepted_by_day[day] = day_count + 1
            accepted_by_symbol_day[(day, symbol)] = symbol_day_count + 1
            accepted_by_session_day[(day, session)] = session_day_count + 1
            accepted_by_symbol_session_day[(day, symbol, session)] = symbol_session_count + 1
            if dedupe_same_symbol_entry:
                used_entry_keys.add((entry_time, symbol))

    accepted = pd.DataFrame(accepted_rows).reset_index(drop=True) if accepted_rows else pd.DataFrame(columns=df.columns)
    return accepted, pd.DataFrame(audit_rows)


def fold_stability_metrics(trades: pd.DataFrame, *, initial_capital: float, risk_per_trade: float, ruin_drawdowns: list[float]) -> dict[str, float]:
    if trades.empty or "fold" not in trades.columns:
        return {"folds_with_trades": 0.0, "positive_folds": 0.0, "negative_folds": 0.0, "worst_fold_net_R": float("nan"), "worst_fold_profit_factor": float("nan"), "worst_fold_profit_factor_R": float("nan"), "worst_fold_win_rate": float("nan")}
    rows = []
    for fold, sub in trades.groupby("fold", sort=False):
        s = summarize_trade_pnl(sub, initial_capital=initial_capital, risk_per_trade=risk_per_trade, ruin_drawdowns=ruin_drawdowns, n_ruin_sims=0, date_col="entry_date")
        s["fold"] = fold
        rows.append(s)
    fdf = pd.DataFrame(rows)
    if fdf.empty:
        return {"folds_with_trades": 0.0, "positive_folds": 0.0, "negative_folds": 0.0, "worst_fold_net_R": float("nan"), "worst_fold_profit_factor": float("nan"), "worst_fold_profit_factor_R": float("nan"), "worst_fold_win_rate": float("nan")}
    worst = fdf.sort_values("net_R", ascending=True).iloc[0]
    return {
        "folds_with_trades": float(len(fdf)),
        "positive_folds": float((pd.to_numeric(fdf["net_R"], errors="coerce") > 0).sum()),
        "negative_folds": float((pd.to_numeric(fdf["net_R"], errors="coerce") < 0).sum()),
        "worst_fold_net_R": float(worst.get("net_R", np.nan)),
        "worst_fold_profit_factor": float(worst.get("profit_factor", np.nan)),
        "worst_fold_profit_factor_R": float(worst.get("profit_factor_R", np.nan)),
        "worst_fold_win_rate": float(worst.get("win_rate", np.nan)),
    }


def component_contribution(trades: pd.DataFrame, *, initial_capital: float, risk_per_trade: float, ruin_drawdowns: list[float]) -> pd.DataFrame:
    if trades.empty or "component_id" not in trades.columns:
        return pd.DataFrame()
    rows = []
    total_r = float(pd.to_numeric(trades["pnl_R"], errors="coerce").fillna(0.0).sum()) if "pnl_R" in trades.columns else 0.0
    for component_id, sub in trades.groupby("component_id", sort=False):
        s = summarize_trade_pnl(sub, initial_capital=initial_capital, risk_per_trade=risk_per_trade, ruin_drawdowns=ruin_drawdowns, n_ruin_sims=0, date_col="entry_date")
        s["component_id"] = component_id
        s["net_R_share"] = float(s["net_R"] / total_r) if total_r != 0 else float("nan")
        rows.append(s)
    return pd.DataFrame(rows)


def reject_reason_summary(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty or "reject_reason" not in audit.columns:
        return pd.DataFrame()
    rows = []
    for reason, sub in audit.groupby("reject_reason", sort=False):
        rows.append(
            {
                "reject_reason": reason,
                "count": float(len(sub)),
                "accepted": bool(reason == "accepted"),
                "net_R": float(pd.to_numeric(sub.get("pnl_R", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
                "net_pips": float(pd.to_numeric(sub.get("pnl_pips", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _merge_controls(base: Mapping[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(base or {})
    if override:
        out.update(dict(override))
    return out


def risk_policies_from_config(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    pe = config.get("portfolio_eval", {})
    base_controls = pe.get("risk_controls", {})
    sweep = pe.get("risk_policy_sweep", {})
    if bool(sweep.get("enabled", False)):
        policies = []
        for p in sweep.get("policies", []):
            if not bool(p.get("enabled", True)):
                continue
            name = str(p.get("name", "risk_policy"))
            policies.append(
                {
                    "name": name,
                    "description": str(p.get("description", name)),
                    "controls": _merge_controls(base_controls, p.get("controls", {})),
                }
            )
        if policies:
            return policies
    return [
        {
            "name": "default",
            "description": "portfolio_eval.risk_controls",
            "controls": dict(base_controls),
        }
    ]


def evaluate_portfolio(config: Mapping[str, Any], portfolio: Mapping[str, Any], risk_policy: Mapping[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    pe = config.get("portfolio_eval", {})
    source_root = Path(pe.get("source_trading_eval_dir", "data/trading_eval"))
    initial_capital = float(pe.get("initial_capital", config.get("trading_eval", {}).get("initial_capital", 1000.0)))
    risk_per_trade = float(pe.get("risk_per_trade", config.get("trading_eval", {}).get("risk_per_trade", 0.02)))
    ruin_drawdowns = [float(x) for x in pe.get("ruin_drawdowns", config.get("trading_eval", {}).get("ruin_drawdowns", [0.25, 0.30]))]
    n_ruin_sims = int(pe.get("risk_of_ruin_simulations", config.get("trading_eval", {}).get("risk_of_ruin_simulations", 5000)))
    random_seed = int(pe.get("random_seed", config.get("trading_eval", {}).get("random_seed", 42)))
    if risk_policy is None:
        controls = pe.get("risk_controls", {})
        risk_policy_name = "default"
        risk_policy_desc = "portfolio_eval.risk_controls"
    else:
        controls = risk_policy.get("controls", {})
        risk_policy_name = str(risk_policy.get("name", "risk_policy"))
        risk_policy_desc = str(risk_policy.get("description", risk_policy_name))

    candidates = load_portfolio_candidate_trades(source_root, list(portfolio.get("components", [])))
    accepted, audit = apply_portfolio_risk_controls(candidates, controls, risk_per_trade=risk_per_trade)
    if not accepted.empty:
        accepted = accepted.copy()
        accepted["risk_policy"] = risk_policy_name
    if not candidates.empty:
        candidates = candidates.copy()
        candidates["risk_policy"] = risk_policy_name
    if not audit.empty:
        audit = audit.copy()
        audit["risk_policy"] = risk_policy_name

    summary = summarize_trade_pnl(
        accepted,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        ruin_drawdowns=ruin_drawdowns,
        n_ruin_sims=n_ruin_sims,
        random_seed=random_seed,
        date_col="entry_date",
    )
    summary.update(fold_stability_metrics(accepted, initial_capital=initial_capital, risk_per_trade=risk_per_trade, ruin_drawdowns=ruin_drawdowns))
    summary.update({
        "portfolio": str(portfolio.get("name", "portfolio")),
        "risk_policy": risk_policy_name,
        "risk_policy_description": risk_policy_desc,
        "configured_component_count": float(len(portfolio.get("components", []))),
        "loaded_component_count": float(candidates["component_id"].nunique()) if not candidates.empty and "component_id" in candidates.columns else 0.0,
        "candidate_trade_count_before_controls": float(len(candidates)),
        "rejected_trade_count": float(len(candidates) - len(accepted)),
        "max_open_trades": _limit_int(controls.get("max_open_trades", None)),
        "max_trades_per_symbol_per_day": _limit_int(controls.get("max_trades_per_symbol_per_day", None)),
        "max_trades_per_day": _limit_int(controls.get("max_trades_per_day", None)),
        "max_open_risk": _limit_float(controls.get("max_open_risk", None)),
        "max_daily_risk": _limit_float(controls.get("max_daily_risk", None)),
    })

    fold_rows = []
    if not accepted.empty and "fold" in accepted.columns:
        for fold, sub in accepted.groupby("fold", sort=False):
            s = summarize_trade_pnl(sub, initial_capital=initial_capital, risk_per_trade=risk_per_trade, ruin_drawdowns=ruin_drawdowns, n_ruin_sims=0, date_col="entry_date")
            s["fold"] = fold
            s["portfolio"] = str(portfolio.get("name", "portfolio"))
            s["risk_policy"] = risk_policy_name
            fold_rows.append(s)

    return {
        "portfolio_summary": pd.DataFrame([summary]),
        "portfolio_selected_trades": accepted,
        "portfolio_candidate_trades_before_controls": candidates,
        "portfolio_risk_control_audit": audit,
        "portfolio_risk_control_reject_summary": reject_reason_summary(audit),
        "portfolio_fold_metrics": pd.DataFrame(fold_rows),
        "portfolio_component_contribution": component_contribution(accepted, initial_capital=initial_capital, risk_per_trade=risk_per_trade, ruin_drawdowns=ruin_drawdowns),
    }
