from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


CORE_12_SETUP_IDS: list[str] = [
    "EUR_BUY_MOMENTUM_OVERLAP",
    "EUR_L4_NOT2_BUY",
    "EUR_AH_ATR2_BUY",
    "EUR_LONDON_WEAK_SHORT",
    "EUR_SELL_H1DOWN_CONT",
    "EUR_SELL_LONDON_BREAKDOWN",
    "XAU_H1UP_BUY",
    "XAU_BUY_DXY_TREND",
    "XAU_BUY_ASIA_HIGH_RECLAIM_DXY",
    "XAU_SHORT_REVERSAL",
    "XAU_SELL_LONDON_REJECTION",
    "XAU_SELL_H1DOWN_CONT",
]

BACKUP_SETUP_IDS: list[str] = ["XAU_SELL_DXY_PRESSURE"]


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def load_best_policy_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing setup best-policy file: {p}")
    df = pd.read_csv(p)
    required = {"setup_id", "experiment", "job", "policy", "probability_column"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Best-policy table is missing required columns: {sorted(missing)}")
    df = df.copy()
    for col in ["trade_count", "profit_factor_R", "net_R", "risk_of_ruin_dd_25pct", "candidate_pass_soft"]:
        if col in df.columns and col != "candidate_pass_soft":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "candidate_pass_soft" in df.columns:
        df["candidate_pass_soft"] = df["candidate_pass_soft"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def component_from_policy_row(row: Mapping[str, Any], *, priority: int = 100, required: bool = True) -> dict[str, Any]:
    setup_id = str(row["setup_id"])
    component: dict[str, Any] = {
        "component_id": setup_id,
        "setup_id": setup_id,
        "family": str(row.get("family", "")),
        "symbol": str(row.get("symbol", "")),
        "side": str(row.get("side", "")),
        "experiment": str(row["experiment"]),
        "job": str(row["job"]),
        "policy": str(row["policy"]),
        "probability_column": str(row["probability_column"]),
        "rank_column": str(row.get("probability_column", "y_prob_raw")),
        "priority": int(priority),
        "required": bool(required),
    }
    if _is_present(row.get("threshold")):
        component["threshold"] = float(row["threshold"])
    if _is_present(row.get("top_percentile")):
        component["top_percentile"] = float(row["top_percentile"])
    return component


def side_coverage_from_components(components: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "eurusd_long_setup_count": 0,
        "eurusd_short_setup_count": 0,
        "xauusd_long_setup_count": 0,
        "xauusd_short_setup_count": 0,
    }
    seen: set[tuple[str, str, str]] = set()
    for c in components:
        setup_id = str(c.get("setup_id", c.get("component_id", "")))
        symbol = str(c.get("symbol", "")).upper()
        side = str(c.get("side", "")).lower()
        key = (setup_id, symbol, side)
        if key in seen:
            continue
        seen.add(key)
        if symbol == "EURUSD" and side == "long":
            counts["eurusd_long_setup_count"] += 1
        elif symbol == "EURUSD" and side == "short":
            counts["eurusd_short_setup_count"] += 1
        elif symbol == "XAUUSD" and side == "long":
            counts["xauusd_long_setup_count"] += 1
        elif symbol == "XAUUSD" and side == "short":
            counts["xauusd_short_setup_count"] += 1
    return counts


def _rows_for_setup_ids(policy_df: pd.DataFrame, setup_ids: list[str], *, allow_missing: bool = False) -> list[dict[str, Any]]:
    by_setup = {str(r["setup_id"]): r.to_dict() for _, r in policy_df.iterrows()}
    rows = []
    missing = []
    for setup_id in setup_ids:
        if setup_id not in by_setup:
            missing.append(setup_id)
        else:
            rows.append(by_setup[setup_id])
    if missing and not allow_missing:
        raise KeyError(f"Missing selected setup policies: {missing}")
    return rows


def make_portfolio_from_setup_ids(
    policy_df: pd.DataFrame,
    *,
    name: str,
    description: str,
    setup_ids: list[str],
    allow_missing: bool = False,
    priority_offset: int = 0,
) -> dict[str, Any]:
    rows = _rows_for_setup_ids(policy_df, setup_ids, allow_missing=allow_missing)
    components = [component_from_policy_row(r, priority=priority_offset + i + 1, required=not allow_missing) for i, r in enumerate(rows)]
    coverage = side_coverage_from_components(components)
    return {
        "name": name,
        "description": description,
        "setup_ids": [str(c.get("setup_id")) for c in components],
        "components": components,
        **coverage,
        "side_complete_3x3_configured": all(v >= 3 for v in coverage.values()),
    }


def default_inventory_portfolio_specs(config: Mapping[str, Any], policy_df: pd.DataFrame, *, allow_missing: bool = False) -> list[dict[str, Any]]:
    ipc = config.get("setup_inventory_portfolio", {})
    core_ids = [str(x) for x in ipc.get("core_setup_ids", CORE_12_SETUP_IDS)]
    backup_ids = [str(x) for x in ipc.get("backup_setup_ids", BACKUP_SETUP_IDS)]

    setup_meta = policy_df.set_index("setup_id", drop=False)

    def ids_for(symbol: str | None = None, side: str | None = None, setup_ids: list[str] | None = None) -> list[str]:
        base = setup_ids if setup_ids is not None else core_ids
        out = []
        for sid in base:
            if sid not in setup_meta.index:
                if allow_missing:
                    continue
                out.append(sid)
                continue
            row = setup_meta.loc[sid]
            if symbol and str(row.get("symbol", "")).upper() != symbol.upper():
                continue
            if side and str(row.get("side", "")).lower() != side.lower():
                continue
            out.append(sid)
        return out

    portfolios = [
        make_portfolio_from_setup_ids(
            policy_df,
            name="ip01_core_12_side_complete",
            description="Core 12 setup-inventory portfolio: 3 setups for each EURUSD/XAUUSD long/short side.",
            setup_ids=core_ids,
            allow_missing=allow_missing,
        ),
        make_portfolio_from_setup_ids(
            policy_df,
            name="ip02_core_12_plus_xau_sell_dxy_pressure",
            description="Core 12 plus XAU_SELL_DXY_PRESSURE backup/diversification candidate.",
            setup_ids=core_ids + backup_ids,
            allow_missing=allow_missing,
        ),
        make_portfolio_from_setup_ids(
            policy_df,
            name="ip03_eurusd_6_setups",
            description="EURUSD only: 3 buy setups plus 3 sell setups.",
            setup_ids=ids_for(symbol="EURUSD"),
            allow_missing=allow_missing,
        ),
        make_portfolio_from_setup_ids(
            policy_df,
            name="ip04_xauusd_6_setups",
            description="XAUUSD only: 3 buy setups plus 3 sell setups.",
            setup_ids=ids_for(symbol="XAUUSD"),
            allow_missing=allow_missing,
        ),
        make_portfolio_from_setup_ids(
            policy_df,
            name="ip05_buy_side_6_setups",
            description="Buy side only: EURUSD buy and XAUUSD buy setup inventory.",
            setup_ids=ids_for(side="long"),
            allow_missing=allow_missing,
        ),
        make_portfolio_from_setup_ids(
            policy_df,
            name="ip06_sell_side_6_setups",
            description="Sell side only: EURUSD sell and XAUUSD sell setup inventory.",
            setup_ids=ids_for(side="short"),
            allow_missing=allow_missing,
        ),
    ]

    # Conservative diagnostic: keep only rows already passing soft criteria with RoR <= 2%.
    if "risk_of_ruin_dd_25pct" in policy_df.columns:
        cons = policy_df.copy()
        cons["risk_of_ruin_dd_25pct"] = pd.to_numeric(cons["risk_of_ruin_dd_25pct"], errors="coerce")
        cons["trade_count"] = pd.to_numeric(cons.get("trade_count"), errors="coerce")
        cons["profit_factor_R"] = pd.to_numeric(cons.get("profit_factor_R"), errors="coerce")
        cons_ids = [
            str(x)
            for x in cons.loc[
                cons["setup_id"].isin(core_ids + backup_ids)
                & (cons["trade_count"] >= 20)
                & (cons["profit_factor_R"] >= 1.30)
                & (cons["risk_of_ruin_dd_25pct"] <= 0.02),
                "setup_id",
            ].tolist()
        ]
        if cons_ids:
            portfolios.append(
                make_portfolio_from_setup_ids(
                    policy_df,
                    name="ip07_low_ror_subset",
                    description="Diagnostic low-RoR subset from selected setup inventory policies.",
                    setup_ids=cons_ids,
                    allow_missing=allow_missing,
                )
            )
    return portfolios


def add_portfolio_metadata(summary: pd.DataFrame, portfolio: Mapping[str, Any]) -> pd.DataFrame:
    out = summary.copy()
    setup_ids = [str(x) for x in portfolio.get("setup_ids", [])]
    out["configured_setup_ids"] = ";".join(setup_ids)
    out["configured_setup_count"] = float(len(setup_ids))
    for k, v in side_coverage_from_components(list(portfolio.get("components", []))).items():
        out[k] = float(v)
    out["side_complete_3x3_configured"] = bool(portfolio.get("side_complete_3x3_configured", False))
    if "loaded_component_count" in out.columns and "configured_component_count" in out.columns:
        out["all_components_loaded"] = (
            pd.to_numeric(out["loaded_component_count"], errors="coerce")
            == pd.to_numeric(out["configured_component_count"], errors="coerce")
        )
    return out


def decision_columns(df: pd.DataFrame, *, min_trades: float, min_pf_r: float, max_ror: float, min_positive_folds: float) -> pd.DataFrame:
    out = df.copy()
    for col in ["trade_count", "profit_factor_R", "risk_of_ruin_dd_25pct", "positive_folds", "loaded_component_count", "configured_component_count"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["portfolio_pass"] = (
        (out.get("trade_count", 0) >= float(min_trades))
        & (out.get("profit_factor_R", 0) >= float(min_pf_r))
        & (out.get("risk_of_ruin_dd_25pct", 1) <= float(max_ror))
        & (out.get("positive_folds", 0) >= float(min_positive_folds))
    )
    if "configured_component_count" in out.columns and "loaded_component_count" in out.columns:
        out["portfolio_pass"] &= out["loaded_component_count"] == out["configured_component_count"]
    return out
