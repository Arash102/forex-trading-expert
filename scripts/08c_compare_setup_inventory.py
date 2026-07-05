from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.utils.io import read_json


def _sort_metric(df: pd.DataFrame) -> str:
    return "profit_factor_R" if "profit_factor_R" in df.columns else "profit_factor"


def _add_selection_columns(
    df: pd.DataFrame,
    *,
    min_trades: float,
    min_pf_r: float,
    max_ror: float,
) -> pd.DataFrame:
    out = df.copy()
    metric = _sort_metric(out)
    out["trade_count"] = pd.to_numeric(out.get("trade_count"), errors="coerce").fillna(0.0)
    out[metric] = pd.to_numeric(out.get(metric), errors="coerce")
    out["net_R"] = pd.to_numeric(out.get("net_R"), errors="coerce")
    out["risk_of_ruin_dd_25pct"] = pd.to_numeric(out.get("risk_of_ruin_dd_25pct"), errors="coerce")
    out["candidate_pass_soft"] = (
        (out["trade_count"] >= float(min_trades))
        & (out[metric] >= float(min_pf_r))
        & (out["risk_of_ruin_dd_25pct"].fillna(1.0) <= float(max_ror))
    )
    out["candidate_pass_pf_only"] = (
        (out["trade_count"] >= float(min_trades))
        & (out[metric] >= float(min_pf_r))
    )
    out["enough_trades"] = out["trade_count"] >= float(min_trades)
    out["finite_pf_for_sort"] = out[metric].replace([np.inf, -np.inf], np.nan).fillna(999.0)
    out["selection_tier"] = np.select(
        [
            out["candidate_pass_soft"],
            out["candidate_pass_pf_only"],
            out["enough_trades"],
        ],
        [3, 2, 1],
        default=0,
    )
    return out


def _best_policy_per_setup(df: pd.DataFrame) -> pd.DataFrame:
    metric = _sort_metric(df)
    return (
        df.sort_values(
            ["setup_id", "selection_tier", "net_R", "finite_pf_for_sort", "trade_count"],
            ascending=[True, False, False, False, False],
        )
        .groupby("setup_id", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare setup inventory results and show candidate coverage toward 3 setups per side.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    parser.add_argument("--min-trades", type=float, default=20.0)
    parser.add_argument("--min-pf-r", type=float, default=1.30)
    parser.add_argument("--max-ror", type=float, default=0.05)
    args = parser.parse_args()

    config = read_json(args.config)
    inv_out = Path(config.get("setup_inventory", {}).get("output_dir", "data/setup_inventory"))
    full_path = inv_out / "setup_trading_eval_summary.csv"
    if not full_path.exists():
        raise SystemExit(f"Missing {full_path}. Run scripts/08b_evaluate_setup_inventory_trading.py first.")

    all_df = pd.read_csv(full_path)
    all_df = _add_selection_columns(
        all_df,
        min_trades=args.min_trades,
        min_pf_r=args.min_pf_r,
        max_ror=args.max_ror,
    )

    policy_candidates = all_df.loc[all_df["candidate_pass_soft"]].copy()
    policy_candidates.to_csv(inv_out / "setup_policy_candidates.csv", index=False)

    best = _best_policy_per_setup(all_df)
    metric = _sort_metric(best)
    best.to_csv(inv_out / "setup_best_policy_by_setup.csv", index=False)
    best.to_csv(inv_out / "setup_best_viable_policy_by_setup.csv", index=False)

    print("\n--- SETUP INVENTORY SELECTED POLICIES ---")
    focus = [c for c in ["setup_id", "symbol", "side", "job", "policy", "probability_column", "threshold", "top_percentile", "trade_count", "win_rate", "profit_factor_R", "payoff_ratio_R", "net_R", "max_drawdown_pct", "risk_of_ruin_dd_25pct", "selection_tier", "candidate_pass_soft"] if c in best.columns]
    print(best[focus].sort_values(["symbol", "side", "selection_tier", metric], ascending=[True, True, False, False]).to_string(index=False))

    print("\n--- ALL SOFT-PASS POLICY CANDIDATES ---")
    if policy_candidates.empty:
        print("No policy rows passed the requested min-trades / PF_R / RoR filters.")
    else:
        pc_focus = [c for c in ["setup_id", "symbol", "side", "policy", "probability_column", "threshold", "top_percentile", "trade_count", "win_rate", "profit_factor_R", "net_R", "max_drawdown_pct", "risk_of_ruin_dd_25pct"] if c in policy_candidates.columns]
        print(
            policy_candidates[pc_focus]
            .sort_values(["symbol", "side", "setup_id", "net_R"], ascending=[True, True, True, False])
            .to_string(index=False)
        )

    coverage = best.groupby(["symbol", "side"], dropna=False).agg(
        setup_count=("setup_id", "count"),
        soft_pass_count=("candidate_pass_soft", "sum"),
        best_pf_R=("profit_factor_R", "max") if "profit_factor_R" in best.columns else (metric, "max"),
        mean_pf_R=("profit_factor_R", "mean") if "profit_factor_R" in best.columns else (metric, "mean"),
        total_trades=("trade_count", "sum"),
    ).reset_index()
    coverage["target_setup_count"] = 3
    coverage["coverage_gap"] = coverage["target_setup_count"] - coverage["soft_pass_count"]
    print("\n--- 3-SETUP-PER-SIDE COVERAGE ---")
    print(coverage.to_string(index=False))

    out_path = inv_out / "setup_inventory_decision_matrix.csv"
    best.to_csv(out_path, index=False)
    coverage.to_csv(inv_out / "setup_inventory_coverage_decision.csv", index=False)
    print(f"\nsaved decision matrix: {out_path}")
    print(f"saved policy candidates: {inv_out / 'setup_policy_candidates.csv'}")


if __name__ == "__main__":
    main()
