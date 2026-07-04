from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OOF trading evaluation summaries.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--max-risk-of-ruin", type=float, default=0.01)
    args = parser.parse_args()

    cfg = read_json(args.config)
    te = cfg.get("trading_eval", {})
    root = Path(te.get("output_dir", "data/trading_eval"))
    path = root / "all_trading_summary.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/06_oof_trading_eval.py first.")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit("Trading summary is empty.")

    df["trade_count"] = pd.to_numeric(df.get("trade_count"), errors="coerce").fillna(0)
    df["profit_factor"] = pd.to_numeric(df.get("profit_factor"), errors="coerce")
    df["win_rate"] = pd.to_numeric(df.get("win_rate"), errors="coerce")
    df["risk_of_ruin_dd_25pct"] = pd.to_numeric(df.get("risk_of_ruin_dd_25pct"), errors="coerce")
    df["max_drawdown_pct"] = pd.to_numeric(df.get("max_drawdown_pct"), errors="coerce")

    eligible = df[df["trade_count"] >= int(args.min_trades)].copy()
    cols = [c for c in [
        "experiment", "job", "summary_table", "policy", "probability_column", "threshold", "top_percentile",
        "trade_count", "win_rate", "payoff_ratio", "profit_factor", "expectancy_pips", "net_pips",
        "max_drawdown_pct", "drawdown_duration_trades", "risk_of_ruin_dd_25pct", "risk_of_ruin_dd_30pct", "trades_per_month"
    ] if c in eligible.columns]

    print("\n--- BEST BY PROFIT FACTOR ---")
    print(eligible[cols].sort_values(["profit_factor", "trade_count"], ascending=[False, False]).head(30).to_string(index=False))

    robust = eligible[(eligible["profit_factor"] >= 1.5) & (eligible["risk_of_ruin_dd_25pct"] <= float(args.max_risk_of_ruin))].copy()
    print("\n--- ROBUST CANDIDATES: PF>=1.5 and risk_of_ruin_25dd <= threshold ---")
    if robust.empty:
        print("No rows matched the robust filter.")
    else:
        print(robust[cols].sort_values(["profit_factor", "max_drawdown_pct"], ascending=[False, True]).head(30).to_string(index=False))

    high_precision = eligible[eligible["win_rate"] >= 0.60].copy()
    print("\n--- HIGH-CONFIDENCE: win_rate >= 60% ---")
    if high_precision.empty:
        print("No rows reached win_rate >= 60% with the requested minimum trade count.")
    else:
        print(high_precision[cols].sort_values(["win_rate", "profit_factor"], ascending=[False, False]).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
