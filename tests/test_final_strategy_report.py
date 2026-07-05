from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from debco.reporting.final_strategy_report import select_strategy_row, write_report_bundle


def test_select_strategy_row_exact_match() -> None:
    df = pd.DataFrame(
        [
            {
                "portfolio": "ip01_core_12_side_complete",
                "risk_policy": "daily_loss_guard",
                "risk_plan": "xau_sell_50pct",
                "risk_per_trade_pct": 0.01,
                "trade_count": 523,
                "profit_factor_R": 1.92,
                "net_R_weighted": 204.0,
                "risk_of_ruin_dd_25pct": 0.0002,
                "positive_folds": 9,
            },
            {
                "portfolio": "ip01_core_12_side_complete",
                "risk_policy": "daily_loss_guard",
                "risk_plan": "xau_sell_50pct",
                "risk_per_trade_pct": 0.015,
                "trade_count": 523,
                "profit_factor_R": 1.92,
                "net_R_weighted": 204.0,
                "risk_of_ruin_dd_25pct": 0.008,
                "positive_folds": 9,
            },
        ]
    )
    row = select_strategy_row(
        df,
        {
            "selected_portfolio": "ip01_core_12_side_complete",
            "selected_risk_policy": "daily_loss_guard",
            "selected_risk_plan": "xau_sell_50pct",
            "selected_risk_per_trade": 0.01,
        },
    )
    assert float(row["risk_per_trade_pct"]) == 0.01
    assert float(row["profit_factor_R"]) == 1.92


def test_write_report_bundle_outputs(tmp_path: Path) -> None:
    ror_dir = tmp_path / "ror"
    setup_dir = tmp_path / "setup"
    out_dir = tmp_path / "final"
    ror_dir.mkdir()
    setup_dir.mkdir()

    pd.DataFrame(
        [
            {
                "portfolio": "ip01_core_12_side_complete",
                "risk_policy": "daily_loss_guard",
                "risk_plan": "xau_sell_50pct",
                "risk_per_trade_pct": 0.01,
                "initial_capital": 1000.0,
                "trade_count": 100,
                "win_rate": 0.5,
                "payoff_ratio_R": 1.9,
                "profit_factor_R": 1.8,
                "gross_profit_R": 180.0,
                "gross_loss_R": 100.0,
                "expectancy_R_weighted": 0.8,
                "net_R_weighted": 80.0,
                "net_return_pct_on_initial": 0.8,
                "net_dollars": 800.0,
                "max_drawdown_pct": 0.1,
                "max_drawdown_dollars": 100.0,
                "risk_of_ruin_dd_25pct": 0.001,
                "risk_of_ruin_dd_30pct": 0.0,
                "folds_with_trades": 9,
                "positive_folds": 8,
                "worst_fold_net_R_weighted": 1.0,
                "configured_component_count": 12,
                "loaded_component_count": 12,
                "side_complete_3x3_configured": True,
                "all_components_loaded": True,
            }
        ]
    ).to_csv(ror_dir / "ror_reduction_summary.csv", index=False)

    setup_rows = []
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
    for sid in core_ids:
        setup_rows.append(
            {
                "setup_id": sid,
                "family": "test",
                "symbol": "EURUSD" if sid.startswith("EUR") else "XAUUSD",
                "side": "long" if "BUY" in sid or sid in {"XAU_H1UP_BUY", "XAU_BUY_DXY_TREND"} else "short",
                "job": "job",
                "experiment": "exp",
                "policy": "top_percentile_by_fold",
                "probability_column": "y_prob_raw",
                "threshold": np.nan,
                "top_percentile": 5.0,
                "trade_count": 20,
                "win_rate": 0.5,
                "payoff_ratio_R": 1.5,
                "profit_factor_R": 1.4,
                "net_R": 5.0,
                "risk_of_ruin_dd_25pct": 0.01,
                "positive_folds": 6,
                "candidate_pass_soft": True,
            }
        )
    pd.DataFrame(setup_rows).to_csv(setup_dir / "setup_best_viable_policy_by_setup.csv", index=False)

    pd.DataFrame(
        [
            {
                "portfolio": "ip01_core_12_side_complete",
                "risk_policy": "daily_loss_guard",
                "risk_plan": "xau_sell_50pct",
                "component_id": "EUR_AH_ATR2_BUY",
                "symbol": "EURUSD",
                "side": "long",
                "trade_count": 20,
                "risk_multiplier_mean": 1.0,
                "profit_factor_R_weighted": 1.5,
                "net_R_weighted": 10.0,
                "max_drawdown_R_weighted": 2.0,
                "win_rate": 0.5,
            }
        ]
    ).to_csv(ror_dir / "ror_component_stress_summary.csv", index=False)

    config = {
        "setup_inventory_portfolio": {"core_setup_ids": core_ids},
        "portfolio_eval": {
            "risk_controls": {"max_open_trades": 8},
            "risk_policy_sweep": {
                "enabled": True,
                "policies": [
                    {
                        "name": "daily_loss_guard",
                        "controls": {"stop_after_daily_losses": 2},
                    }
                ],
            },
        },
        "inventory_ror_optimizer": {
            "risk_plans": [
                {
                    "name": "xau_sell_50pct",
                    "symbol_side_weights": {"XAUUSD|short": 0.5},
                }
            ]
        },
        "final_strategy_report": {
            "output_dir": str(out_dir),
            "selected_portfolio": "ip01_core_12_side_complete",
            "selected_risk_policy": "daily_loss_guard",
            "selected_risk_plan": "xau_sell_50pct",
            "selected_risk_per_trade": 0.01,
            "ror_summary_file": str(ror_dir / "ror_reduction_summary.csv"),
            "component_stress_file": str(ror_dir / "ror_component_stress_summary.csv"),
            "setup_best_policy_file": str(setup_dir / "setup_best_viable_policy_by_setup.csv"),
        },
    }
    outputs = write_report_bundle(config)
    assert outputs["report"].exists()
    assert outputs["live_spec"].exists()
    assert outputs["setup_matrix"].exists()
    spec_text = outputs["live_spec"].read_text(encoding="utf-8")
    assert "NaN" not in spec_text
    spec = json.loads(spec_text)
    assert spec["selected_setups"][0]["threshold"] is None
    text = outputs["report"].read_text(encoding="utf-8")
    assert "گزارش نهایی" in text
