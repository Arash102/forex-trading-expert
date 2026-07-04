from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare portfolio OOF evaluation summaries.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--max-risk-of-ruin", type=float, default=0.01)
    parser.add_argument("--min-positive-folds", type=int, default=6)
    args = parser.parse_args()

    cfg = read_json(args.config)
    root = Path(cfg.get("portfolio_eval", {}).get("output_dir", "data/portfolio_eval"))
    path = root / "all_portfolio_summary.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/07_portfolio_oof_eval.py first.")
    df = pd.read_csv(path)
    numeric_candidates = [
        "trade_count", "profit_factor", "profit_factor_R", "payoff_ratio_R", "win_rate",
        "risk_of_ruin_dd_25pct", "max_drawdown_pct", "positive_folds", "net_R"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    pf_col = "profit_factor_R" if "profit_factor_R" in df.columns else "profit_factor"
    cols = [c for c in [
        "portfolio", "risk_policy", "configured_component_count", "loaded_component_count", "trade_count", "win_rate",
        "payoff_ratio_R", "profit_factor_R", "payoff_ratio", "profit_factor",
        "expectancy_R", "net_R", "net_dollars", "return_pct_on_initial", "max_drawdown_pct",
        "drawdown_duration_trades_R", "drawdown_duration_days_R", "risk_of_ruin_dd_25pct", "risk_of_ruin_dd_30pct",
        "positive_folds", "negative_folds", "worst_fold_net_R", "worst_fold_profit_factor_R", "worst_fold_profit_factor",
        "trades_per_month", "candidate_trade_count_before_controls", "rejected_trade_count",
        "max_open_trades", "max_trades_per_symbol_per_day", "max_trades_per_day", "max_open_risk", "max_daily_risk"
    ] if c in df.columns]

    eligible = df[df["trade_count"] >= int(args.min_trades)].copy()
    print("\n--- BEST PORTFOLIOS BY PROFIT FACTOR ---")
    if eligible.empty:
        print("No portfolios matched min-trades.")
    else:
        print(eligible[cols].sort_values([pf_col, "risk_of_ruin_dd_25pct", "trade_count"], ascending=[False, True, False]).to_string(index=False))

    robust = eligible[
        (eligible[pf_col] >= 1.5)
        & (eligible["risk_of_ruin_dd_25pct"] <= float(args.max_risk_of_ruin))
        & (eligible["positive_folds"] >= int(args.min_positive_folds))
    ].copy()
    print("\n--- ROBUST PORTFOLIOS: PF>=1.5, RoR<=threshold, positive folds target ---")
    if robust.empty:
        print("No portfolios matched the robust filter.")
    else:
        print(robust[cols].sort_values([pf_col, "max_drawdown_pct"], ascending=[False, True]).to_string(index=False))

    high_conf = eligible[eligible["win_rate"] >= 0.60].copy()
    print("\n--- HIGH-CONFIDENCE PORTFOLIOS: win_rate >= 60% ---")
    if high_conf.empty:
        print("No portfolios reached win_rate >= 60% with the requested minimum trade count.")
    else:
        print(high_conf[cols].sort_values(["win_rate", pf_col], ascending=[False, False]).to_string(index=False))

    by_policy = eligible.groupby("risk_policy", dropna=False).agg(
        portfolios=("portfolio", "count"),
        best_profit_factor=(pf_col, "max"),
        best_win_rate=("win_rate", "max"),
        min_ror25=("risk_of_ruin_dd_25pct", "min"),
    ).reset_index()
    print("\n--- RISK POLICY SNAPSHOT ---")
    print(by_policy.sort_values(["best_profit_factor", "min_ror25"], ascending=[False, True]).to_string(index=False))


if __name__ == "__main__":
    main()
