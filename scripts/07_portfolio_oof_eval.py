from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.trading.portfolio_eval import evaluate_portfolio, risk_policies_from_config
from debco.utils.io import ensure_dir, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate portfolios of OOF trading policies with risk-policy sweep.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    parser.add_argument("--portfolio", default=None, help="Optional portfolio name to evaluate.")
    parser.add_argument("--risk-policy", default=None, help="Optional risk policy name to evaluate.")
    args = parser.parse_args()

    config = read_json(args.config)
    pe = config.get("portfolio_eval", {})
    output_root = ensure_dir(Path(pe.get("output_dir", "data/portfolio_eval")))
    portfolios = [p for p in pe.get("portfolios", []) if bool(p.get("enabled", True))]
    if args.portfolio:
        portfolios = [p for p in portfolios if str(p.get("name")) == args.portfolio]
    if not portfolios:
        raise SystemExit("No enabled portfolios found. Check portfolio_eval.portfolios in config.")

    risk_policies = risk_policies_from_config(config)
    if args.risk_policy:
        risk_policies = [p for p in risk_policies if str(p.get("name")) == args.risk_policy]
    if not risk_policies:
        raise SystemExit("No enabled risk policies found. Check portfolio_eval.risk_policy_sweep in config.")

    all_summaries = []
    for p in portfolios:
        name = str(p.get("name", "portfolio"))
        for rp in risk_policies:
            rp_name = str(rp.get("name", "risk_policy"))
            print(f"\n=== PORTFOLIO OOF EVAL {name} | risk_policy={rp_name} ===")
            out_dir = ensure_dir(output_root / name / rp_name)
            result = evaluate_portfolio(config, p, rp)
            for key, df in result.items():
                if not df.empty:
                    df.to_csv(out_dir / f"{key}.csv", index=False)
            summary = result["portfolio_summary"].copy()
            all_summaries.append(summary)
            focus = [c for c in [
                "portfolio", "risk_policy", "trade_count", "win_rate", "payoff_ratio_R", "profit_factor_R",
                "payoff_ratio", "profit_factor", "expectancy_R", "net_R", "net_dollars", "max_drawdown_pct", "drawdown_duration_trades_R",
                "risk_of_ruin_dd_25pct", "positive_folds", "negative_folds", "trades_per_month",
                "configured_component_count", "loaded_component_count", "candidate_trade_count_before_controls", "rejected_trade_count", "max_open_trades",
                "max_trades_per_symbol_per_day"
            ] if c in summary.columns]
            print(summary[focus].to_string(index=False))

    all_df = pd.concat(all_summaries, ignore_index=True)
    all_df.to_csv(output_root / "all_portfolio_summary.csv", index=False)
    print(f"\nsaved portfolio summary: {output_root / 'all_portfolio_summary.csv'} rows={len(all_df)}")


if __name__ == "__main__":
    main()
