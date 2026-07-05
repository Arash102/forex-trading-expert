from __future__ import annotations

import pandas as pd

from debco.trading.inventory_portfolio import (
    component_from_policy_row,
    default_inventory_portfolio_specs,
    decision_columns,
    side_coverage_from_components,
)


def _row(setup_id: str, symbol: str, side: str) -> dict:
    return {
        "setup_id": setup_id,
        "family": "test",
        "symbol": symbol,
        "side": side,
        "experiment": f"exp_{setup_id.lower()}",
        "job": f"{symbol}_job_{side}",
        "policy": "fixed_threshold",
        "probability_column": "y_prob_raw",
        "threshold": 0.6,
        "top_percentile": pd.NA,
        "trade_count": 30,
        "profit_factor_R": 1.6,
        "net_R": 10,
        "risk_of_ruin_dd_25pct": 0.005,
        "candidate_pass_soft": True,
    }


def test_component_from_policy_row_omits_missing_top_percentile() -> None:
    c = component_from_policy_row(_row("A", "EURUSD", "long"), priority=7)
    assert c["component_id"] == "A"
    assert c["threshold"] == 0.6
    assert "top_percentile" not in c
    assert c["priority"] == 7


def test_side_coverage_counts_unique_symbol_sides() -> None:
    comps = [
        {"setup_id": "a", "symbol": "EURUSD", "side": "long"},
        {"setup_id": "b", "symbol": "EURUSD", "side": "short"},
        {"setup_id": "c", "symbol": "XAUUSD", "side": "long"},
        {"setup_id": "d", "symbol": "XAUUSD", "side": "short"},
    ]
    cov = side_coverage_from_components(comps)
    assert cov["eurusd_long_setup_count"] == 1
    assert cov["eurusd_short_setup_count"] == 1
    assert cov["xauusd_long_setup_count"] == 1
    assert cov["xauusd_short_setup_count"] == 1


def test_default_inventory_portfolio_specs_builds_core() -> None:
    core_ids = [
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
    symbols_sides = [
        ("EURUSD", "long"), ("EURUSD", "long"), ("EURUSD", "long"),
        ("EURUSD", "short"), ("EURUSD", "short"), ("EURUSD", "short"),
        ("XAUUSD", "long"), ("XAUUSD", "long"), ("XAUUSD", "long"),
        ("XAUUSD", "short"), ("XAUUSD", "short"), ("XAUUSD", "short"),
    ]
    df = pd.DataFrame([_row(sid, symbol, side) for sid, (symbol, side) in zip(core_ids, symbols_sides)])
    portfolios = default_inventory_portfolio_specs({"setup_inventory_portfolio": {"core_setup_ids": core_ids, "backup_setup_ids": []}}, df)
    core = [p for p in portfolios if p["name"] == "ip01_core_12_side_complete"][0]
    assert len(core["components"]) == 12
    assert core["side_complete_3x3_configured"] is True


def test_decision_columns_requires_loaded_components() -> None:
    df = pd.DataFrame([
        {
            "trade_count": 80,
            "profit_factor_R": 1.7,
            "risk_of_ruin_dd_25pct": 0.005,
            "positive_folds": 6,
            "configured_component_count": 12,
            "loaded_component_count": 11,
        }
    ])
    out = decision_columns(df, min_trades=60, min_pf_r=1.5, max_ror=0.01, min_positive_folds=6)
    assert bool(out.loc[0, "portfolio_pass"]) is False
