from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.trading.inventory_portfolio import decision_columns
from debco.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare setup-inventory portfolio evaluation results.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    parser.add_argument("--min-trades", type=float, default=60.0)
    parser.add_argument("--min-pf-r", type=float, default=1.50)
    parser.add_argument("--max-ror", type=float, default=0.01)
    parser.add_argument("--min-positive-folds", type=float, default=6.0)
    args = parser.parse_args()

    config = read_json(args.config)
    out_root = Path(config.get("setup_inventory_portfolio", {}).get("output_dir", "data/inventory_portfolio_eval"))
    path = out_root / "all_inventory_portfolio_summary.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/09_inventory_portfolio_eval.py first.")

    df = pd.read_csv(path)
    df = decision_columns(
        df,
        min_trades=args.min_trades,
        min_pf_r=args.min_pf_r,
        max_ror=args.max_ror,
        min_positive_folds=args.min_positive_folds,
    )
    sort_cols = [c for c in ["portfolio_pass", "profit_factor_R", "risk_of_ruin_dd_25pct", "positive_folds", "net_R"] if c in df.columns]
    ascending = [False, False, True, False, False][: len(sort_cols)]
    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    focus = [
        c
        for c in [
            "portfolio_pass",
            "portfolio",
            "risk_policy",
            "trade_count",
            "win_rate",
            "payoff_ratio_R",
            "profit_factor_R",
            "expectancy_R",
            "net_R",
            "max_drawdown_pct",
            "drawdown_duration_trades_R",
            "risk_of_ruin_dd_25pct",
            "positive_folds",
            "negative_folds",
            "configured_component_count",
            "loaded_component_count",
            "all_components_loaded",
            "side_complete_3x3_configured",
            "eurusd_long_setup_count",
            "eurusd_short_setup_count",
            "xauusd_long_setup_count",
            "xauusd_short_setup_count",
        ]
        if c in df.columns
    ]
    print("\n--- INVENTORY PORTFOLIO DECISION MATRIX ---")
    print(df[focus].to_string(index=False))

    robust = df.loc[df["portfolio_pass"]].copy()
    if robust.empty:
        print("\nNo inventory portfolio passed the requested robust filters.")
    else:
        print("\n--- ROBUST PASS CANDIDATES ---")
        print(robust[focus].to_string(index=False))

    out_path = out_root / "inventory_portfolio_decision_matrix.csv"
    df.to_csv(out_path, index=False)
    robust.to_csv(out_root / "inventory_portfolio_robust_pass.csv", index=False)
    print(f"\nsaved decision matrix: {out_path}")


if __name__ == "__main__":
    main()
