from __future__ import annotations

import pandas as pd

from debco.trading.inventory_ror_optimizer import (
    DecisionFilters,
    add_risk_multipliers,
    apply_decision_filters,
    risk_multiplier_for_trade,
    summarize_weighted_trades,
)


def sample_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_date": pd.date_range("2025-01-01", periods=8, freq="h"),
            "pnl_R": [1.8, -1.0, 1.8, -1.0, 1.8, -1.0, 1.8, 1.8],
            "component_id": ["A", "A", "B", "B", "A", "B", "A", "B"],
            "symbol": ["EURUSD", "XAUUSD", "EURUSD", "XAUUSD", "EURUSD", "XAUUSD", "EURUSD", "XAUUSD"],
            "side": ["long", "short", "long", "short", "long", "short", "long", "short"],
            "fold": ["wf_0", "wf_0", "wf_1", "wf_1", "wf_2", "wf_2", "wf_3", "wf_3"],
        }
    )


def test_risk_multiplier_for_trade_combines_symbol_side_component() -> None:
    row = {"symbol": "XAUUSD", "side": "short", "component_id": "A"}
    plan = {
        "symbol_weights": {"XAUUSD": 0.5},
        "side_weights": {"short": 0.8},
        "component_weights": {"A": 0.25},
    }
    assert risk_multiplier_for_trade(row, plan) == 0.1


def test_add_risk_multipliers_creates_weighted_pnl() -> None:
    df = sample_trades()
    out = add_risk_multipliers(df, {"symbol_weights": {"XAUUSD": 0.5}})
    assert "weighted_pnl_R" in out.columns
    assert out.loc[1, "weighted_pnl_R"] == -0.5
    assert out.loc[0, "weighted_pnl_R"] == 1.8


def test_summarize_weighted_trades_scales_drawdown_with_risk() -> None:
    df = sample_trades()
    low = summarize_weighted_trades(
        df,
        portfolio="p",
        risk_policy="r",
        risk_plan={"name": "uniform"},
        risk_per_trade=0.005,
        initial_capital=1000,
        ruin_drawdowns=[0.25],
        simulations=10,
        seed=1,
    )
    high = summarize_weighted_trades(
        df,
        portfolio="p",
        risk_policy="r",
        risk_plan={"name": "uniform"},
        risk_per_trade=0.02,
        initial_capital=1000,
        ruin_drawdowns=[0.25],
        simulations=10,
        seed=1,
    )
    assert high["profit_factor_R"] == low["profit_factor_R"]
    assert high["max_drawdown_pct"] > low["max_drawdown_pct"]
    assert high["net_return_pct_on_initial"] > low["net_return_pct_on_initial"]


def test_apply_decision_filters_requires_component_loading() -> None:
    df = pd.DataFrame(
        {
            "trade_count": [100],
            "profit_factor_R": [1.7],
            "risk_of_ruin_dd_25pct": [0.005],
            "positive_folds": [7],
            "loaded_component_count": [11],
            "configured_component_count": [12],
        }
    )
    out = apply_decision_filters(df, DecisionFilters())
    assert bool(out.loc[0, "portfolio_pass"]) is False
