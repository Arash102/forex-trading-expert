from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.trading.inventory_portfolio import (
    add_portfolio_metadata,
    default_inventory_portfolio_specs,
    load_best_policy_table,
)
from debco.trading.portfolio_eval import evaluate_portfolio, risk_policies_from_config
from debco.utils.io import ensure_dir, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate portfolio combinations built from setup-inventory best viable policies.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    parser.add_argument("--portfolio", default=None, help="Optional inventory portfolio name.")
    parser.add_argument("--risk-policy", default=None, help="Optional risk policy name.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing setup policy rows instead of failing.")
    args = parser.parse_args()

    config = read_json(args.config)
    ipc = config.get("setup_inventory_portfolio", {})
    if not bool(ipc.get("enabled", True)):
        raise SystemExit("setup_inventory_portfolio is disabled in config.")

    best_policy_file = Path(ipc.get("best_policy_file", "data/setup_inventory/setup_best_viable_policy_by_setup.csv"))
    policy_df = load_best_policy_table(best_policy_file)
    portfolios = default_inventory_portfolio_specs(config, policy_df, allow_missing=bool(args.allow_missing))
    if args.portfolio:
        portfolios = [p for p in portfolios if str(p.get("name")) == str(args.portfolio)]
    if not portfolios:
        raise SystemExit("No inventory portfolios matched. Check --portfolio or setup_inventory_portfolio config.")

    risk_policies = risk_policies_from_config(config)
    if args.risk_policy:
        risk_policies = [p for p in risk_policies if str(p.get("name")) == str(args.risk_policy)]
    if not risk_policies:
        raise SystemExit("No risk policies matched. Check --risk-policy or portfolio_eval.risk_policy_sweep.")

    out_root = ensure_dir(Path(ipc.get("output_dir", "data/inventory_portfolio_eval")))
    all_summaries: list[pd.DataFrame] = []

    # evaluate_portfolio reads output/risk settings from portfolio_eval. Keep those settings but write inventory outputs separately.
    eval_config = copy.deepcopy(config)
    eval_config.setdefault("portfolio_eval", {})
    eval_config["portfolio_eval"]["source_trading_eval_dir"] = str(config.get("portfolio_eval", {}).get("source_trading_eval_dir", "data/trading_eval"))

    for portfolio in portfolios:
        name = str(portfolio.get("name", "inventory_portfolio"))
        for rp in risk_policies:
            rp_name = str(rp.get("name", "risk_policy"))
            print(f"\n=== INVENTORY PORTFOLIO EVAL {name} | risk_policy={rp_name} ===")
            out_dir = ensure_dir(out_root / name / rp_name)
            result = evaluate_portfolio(eval_config, portfolio, rp)
            result["portfolio_summary"] = add_portfolio_metadata(result["portfolio_summary"], portfolio)
            for key, df in result.items():
                if not df.empty:
                    df.to_csv(out_dir / f"{key}.csv", index=False)
            summary = result["portfolio_summary"].copy()
            all_summaries.append(summary)
            focus = [
                c
                for c in [
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
                    "candidate_trade_count_before_controls",
                    "rejected_trade_count",
                ]
                if c in summary.columns
            ]
            print(summary[focus].to_string(index=False))

    all_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    all_df.to_csv(out_root / "all_inventory_portfolio_summary.csv", index=False)
    print(f"\nsaved inventory portfolio summary: {out_root / 'all_inventory_portfolio_summary.csv'} rows={len(all_df)}")


if __name__ == "__main__":
    main()
