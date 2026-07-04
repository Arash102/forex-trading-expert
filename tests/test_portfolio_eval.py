from __future__ import annotations

import pandas as pd

from debco.trading.portfolio_eval import apply_portfolio_risk_controls, infer_session_label, risk_policies_from_config


def test_portfolio_daily_risk_limits_trades() -> None:
    trades = pd.DataFrame(
        {
            "entry_date": pd.to_datetime(["2026-01-01 09:00", "2026-01-01 10:00", "2026-01-01 11:00"]),
            "exit_date": pd.to_datetime(["2026-01-01 09:30", "2026-01-01 10:30", "2026-01-01 11:30"]),
            "symbol": ["EURUSD", "XAUUSD", "EURUSD"],
            "side": ["long", "short", "long"],
            "component_id": ["a", "b", "c"],
            "component_priority": [1, 2, 3],
            "portfolio_rank_score": [0.5, 0.4, 0.3],
            "pnl_R": [1.0, -1.0, 1.0],
            "pnl_pips": [10.0, -8.0, 10.0],
        }
    )
    accepted, audit = apply_portfolio_risk_controls(
        trades,
        {"max_daily_risk": 0.04, "max_open_risk": 0.10, "max_trades_per_day": 10, "max_trades_per_symbol_per_day": 10},
        risk_per_trade=0.02,
    )
    assert len(accepted) == 2
    assert audit["accepted"].sum() == 2
    assert "reject_max_daily_risk" in set(audit["reject_reason"])


def test_portfolio_open_risk_limits_overlapping_trades() -> None:
    trades = pd.DataFrame(
        {
            "entry_date": pd.to_datetime(["2026-01-01 09:00", "2026-01-01 09:15", "2026-01-01 12:00"]),
            "exit_date": pd.to_datetime(["2026-01-01 11:00", "2026-01-01 10:00", "2026-01-01 12:30"]),
            "symbol": ["EURUSD", "XAUUSD", "EURUSD"],
            "side": ["long", "short", "long"],
            "component_id": ["a", "b", "c"],
            "component_priority": [1, 2, 3],
            "portfolio_rank_score": [0.5, 0.4, 0.3],
            "pnl_R": [1.0, -1.0, 1.0],
            "pnl_pips": [10.0, -8.0, 10.0],
        }
    )
    accepted, audit = apply_portfolio_risk_controls(
        trades,
        {"max_open_risk": 0.02, "max_daily_risk": 0.10, "max_trades_per_day": 10, "max_trades_per_symbol_per_day": 10},
        risk_per_trade=0.02,
    )
    assert len(accepted) == 2
    assert "reject_max_open_risk" in set(audit["reject_reason"])


def test_user_requested_caps_allow_four_per_symbol_and_limit_open_trades() -> None:
    trades = pd.DataFrame(
        {
            "entry_date": pd.to_datetime([f"2026-01-01 09:{i:02d}" for i in range(9)]),
            "exit_date": pd.to_datetime(["2026-01-01 12:00"] * 9),
            "symbol": ["EURUSD"] * 9,
            "side": ["long"] * 9,
            "component_id": ["a"] * 9,
            "component_priority": [1] * 9,
            "portfolio_rank_score": list(range(9, 0, -1)),
            "pnl_R": [1.0] * 9,
            "pnl_pips": [10.0] * 9,
        }
    )
    accepted, audit = apply_portfolio_risk_controls(
        trades,
        {
            "max_open_trades": 8,
            "max_open_risk": 0.16,
            "max_daily_risk": 0.16,
            "max_trades_per_day": 8,
            "max_trades_per_symbol_per_day": 4,
            "dedupe_same_symbol_entry": True,
        },
        risk_per_trade=0.02,
    )
    assert len(accepted) == 4
    assert "reject_max_trades_per_symbol_per_day" in set(audit["reject_reason"])


def test_non_overlap_same_symbol_rejects_overlapping_same_symbol() -> None:
    trades = pd.DataFrame(
        {
            "entry_date": pd.to_datetime(["2026-01-01 09:00", "2026-01-01 09:15", "2026-01-01 11:15"]),
            "exit_date": pd.to_datetime(["2026-01-01 11:00", "2026-01-01 10:00", "2026-01-01 11:45"]),
            "symbol": ["EURUSD", "EURUSD", "EURUSD"],
            "side": ["long", "long", "long"],
            "component_id": ["a", "b", "c"],
            "component_priority": [1, 1, 1],
            "portfolio_rank_score": [0.6, 0.7, 0.8],
            "pnl_R": [1.0, 1.0, 1.0],
            "pnl_pips": [10.0, 10.0, 10.0],
        }
    )
    accepted, audit = apply_portfolio_risk_controls(
        trades,
        {"no_overlap_same_symbol": True, "max_open_trades": 8, "max_trades_per_symbol_per_day": 4},
        risk_per_trade=0.02,
    )
    assert len(accepted) == 2
    assert "reject_same_symbol_overlap" in set(audit["reject_reason"])


def test_session_cap_limits_one_trade_per_symbol_session() -> None:
    trades = pd.DataFrame(
        {
            "entry_date": pd.to_datetime(["2026-01-01 12:15", "2026-01-01 12:45", "2026-01-01 14:15"]),
            "exit_date": pd.to_datetime(["2026-01-01 12:30", "2026-01-01 13:00", "2026-01-01 14:30"]),
            "symbol": ["XAUUSD", "XAUUSD", "XAUUSD"],
            "side": ["short", "short", "short"],
            "component_id": ["a", "b", "c"],
            "component_priority": [1, 1, 1],
            "portfolio_rank_score": [0.6, 0.7, 0.8],
            "pnl_R": [1.0, 1.0, 1.0],
            "pnl_pips": [1500.0, 1500.0, 1500.0],
        }
    )
    accepted, audit = apply_portfolio_risk_controls(
        trades,
        {"max_trades_per_symbol_per_session": 1, "max_trades_per_symbol_per_day": 4, "max_open_trades": 8},
        risk_per_trade=0.02,
    )
    assert len(accepted) == 2
    assert infer_session_label(pd.Timestamp("2026-01-01 12:15")) == "overlap_early"
    assert infer_session_label(pd.Timestamp("2026-01-01 14:15")) == "overlap_late"
    assert "reject_max_trades_per_symbol_per_session" in set(audit["reject_reason"])


def test_risk_policy_sweep_from_config() -> None:
    cfg = {
        "portfolio_eval": {
            "risk_controls": {"max_open_trades": 8},
            "risk_policy_sweep": {
                "enabled": True,
                "policies": [
                    {"name": "a", "controls": {"max_trades_per_symbol_per_day": 4}},
                    {"enabled": False, "name": "b"},
                ],
            },
        }
    }
    policies = risk_policies_from_config(cfg)
    assert [p["name"] for p in policies] == ["a"]
    assert policies[0]["controls"]["max_open_trades"] == 8
    assert policies[0]["controls"]["max_trades_per_symbol_per_day"] == 4


def test_portfolio_summary_reports_r_based_payoff_metrics() -> None:
    from debco.trading.risk_metrics import summarize_trade_pnl

    trades = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "pnl_pips": [15.0, -8.0, 30.0],
            "pnl_R": [1.875, -1.0, 1.875],
        }
    )
    summary = summarize_trade_pnl(trades, initial_capital=1000.0, risk_per_trade=0.02, n_ruin_sims=0)
    assert round(summary["profit_factor_R"], 6) == round((1.875 + 1.875) / 1.0, 6)
    assert round(summary["payoff_ratio_R"], 6) == round(1.875 / 1.0, 6)
    assert summary["gross_profit_R"] == 3.75
    assert summary["gross_loss_R"] == 1.0
