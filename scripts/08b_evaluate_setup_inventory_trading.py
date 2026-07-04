from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.ml.setup_inventory import config_for_setup, list_setup_specs, setup_matches_filters
from debco.trading.oof_eval import evaluate_job_trading
from debco.utils.io import ensure_dir, read_json


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
    out["max_drawdown_pct"] = pd.to_numeric(out.get("max_drawdown_pct"), errors="coerce")

    # Do not let 1-2 trade infinite-PF rows dominate setup selection.
    out["enough_trades"] = out["trade_count"] >= float(min_trades)
    out["candidate_pass_soft"] = (
        out["enough_trades"]
        & (out[metric] >= float(min_pf_r))
        & (out["risk_of_ruin_dd_25pct"].fillna(1.0) <= float(max_ror))
    )
    out["candidate_pass_pf_only"] = out["enough_trades"] & (out[metric] >= float(min_pf_r))
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
    sort_cols = [
        "setup_id",
        "selection_tier",
        "net_R",
        "finite_pf_for_sort",
        "trade_count",
    ]
    ascending = [True, False, False, False, False]
    existing_sort_cols = [c for c in sort_cols if c in df.columns]
    existing_ascending = [ascending[sort_cols.index(c)] for c in existing_sort_cols]
    return (
        df.sort_values(existing_sort_cols, ascending=existing_ascending)
        .groupby("setup_id", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained setup-inventory models as OOF trading strategies.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    parser.add_argument("--setup-id", default=None, help="Optional exact setup_id.")
    parser.add_argument("--symbol", default=None, help="Optional symbol filter.")
    parser.add_argument("--side", default=None, choices=["long", "short"], help="Optional side filter.")
    parser.add_argument("--family", default=None, help="Optional family filter.")
    parser.add_argument("--min-trades", type=float, default=20.0, help="Minimum trades for a viable setup-policy candidate.")
    parser.add_argument("--min-pf-r", type=float, default=1.30, help="Minimum R-based profit factor for a soft pass.")
    parser.add_argument("--max-ror", type=float, default=0.05, help="Maximum 25%% drawdown risk-of-ruin for a soft pass.")
    args = parser.parse_args()

    base_cfg = read_json(args.config)
    specs = [
        s for s in list_setup_specs(base_cfg)
        if setup_matches_filters(s, setup_id=args.setup_id, symbol=args.symbol, side=args.side, family=args.family)
    ]
    if not specs:
        raise SystemExit("No setup specs matched the requested filters.")

    results_dir = Path(base_cfg.get("output", {}).get("results_dir", "data/ml_results"))
    trading_root = ensure_dir(Path(base_cfg.get("trading_eval", {}).get("output_dir", "data/trading_eval")))
    inv_out = ensure_dir(Path(base_cfg.get("setup_inventory", {}).get("output_dir", "data/setup_inventory")))

    all_rows = []
    for spec in specs:
        cfg = config_for_setup(base_cfg, spec)
        exp_name = str(cfg.get("output", {}).get("experiment_name"))
        exp_root = results_dir / exp_name
        job_dir = exp_root / spec.job.name
        if not (job_dir / "oof_predictions.csv").exists():
            print(f"missing trained OOF predictions for {spec.setup_id}: {job_dir / 'oof_predictions.csv'}")
            continue
        print(f"\n=== SETUP TRADING EVAL {spec.setup_id} :: {spec.job.name} ===")
        result = evaluate_job_trading(cfg, job_dir=job_dir, job_name=spec.job.name)
        job_out = ensure_dir(trading_root / exp_name / spec.job.name)
        for name, df in result.items():
            if not df.empty:
                df.to_csv(job_out / f"{name}.csv", index=False)
        summaries = []
        for name in ["fixed_threshold_summary", "top_percentile_summary", "rolling_target_precision_summary"]:
            df = result.get(name, pd.DataFrame())
            if not df.empty:
                tmp = df.copy()
                tmp.insert(0, "summary_table", name)
                summaries.append(tmp)
        if summaries:
            combined = pd.concat(summaries, ignore_index=True)
            combined.insert(0, "job", spec.job.name)
            combined.insert(0, "experiment", exp_name)
            combined.insert(0, "side", spec.side)
            combined.insert(0, "symbol", spec.symbol)
            combined.insert(0, "family", spec.family)
            combined.insert(0, "setup_id", spec.setup_id)
            combined = _add_selection_columns(
                combined,
                min_trades=args.min_trades,
                min_pf_r=args.min_pf_r,
                max_ror=args.max_ror,
            )
            combined.to_csv(job_out / "trading_summary.csv", index=False)
            all_rows.append(combined)
            metric = _sort_metric(combined)
            focus = [c for c in ["summary_table", "policy", "probability_column", "threshold", "top_percentile", "trade_count", "win_rate", "payoff_ratio_R", "profit_factor_R", "expectancy_R", "net_R", "max_drawdown_pct", "risk_of_ruin_dd_25pct", "selection_tier", "candidate_pass_soft"] if c in combined.columns]
            print(combined[focus].sort_values(["selection_tier", "net_R", metric], ascending=[False, False, False]).head(8).to_string(index=False))

    if not all_rows:
        raise SystemExit("No setup trading evaluations were produced. Train setup models first.")
    all_summary = pd.concat(all_rows, ignore_index=True)
    all_summary = _add_selection_columns(
        all_summary,
        min_trades=args.min_trades,
        min_pf_r=args.min_pf_r,
        max_ror=args.max_ror,
    )
    all_summary.to_csv(inv_out / "setup_trading_eval_summary.csv", index=False)

    viable = all_summary.loc[all_summary["candidate_pass_soft"]].copy()
    viable.to_csv(inv_out / "setup_policy_candidates.csv", index=False)

    best = _best_policy_per_setup(all_summary)
    best.to_csv(inv_out / "setup_best_policy_by_setup.csv", index=False)
    best.to_csv(inv_out / "setup_best_viable_policy_by_setup.csv", index=False)

    metric = _sort_metric(best)
    coverage = best.groupby(["symbol", "side"], dropna=False).agg(
        setup_count=("setup_id", "count"),
        soft_pass_count=("candidate_pass_soft", "sum"),
        best_pf_R=("profit_factor_R", "max") if "profit_factor_R" in best.columns else (metric, "max"),
        mean_pf_R=("profit_factor_R", "mean") if "profit_factor_R" in best.columns else (metric, "mean"),
        total_trades=("trade_count", "sum"),
    ).reset_index()
    coverage["target_setup_count"] = 3
    coverage["coverage_gap"] = coverage["target_setup_count"] - coverage["soft_pass_count"]
    coverage.to_csv(inv_out / "setup_trading_coverage_matrix.csv", index=False)

    print(f"\nsaved setup trading summary: {inv_out / 'setup_trading_eval_summary.csv'}")
    print(f"saved viable setup-policy candidates: {inv_out / 'setup_policy_candidates.csv'}")
    print(f"saved best policies: {inv_out / 'setup_best_policy_by_setup.csv'}")
    print("\n--- BEST SETUP POLICY BY SYMBOL/SIDE ---")
    focus = [c for c in ["setup_id", "symbol", "side", "job", "summary_table", "policy", "probability_column", "threshold", "top_percentile", "trade_count", "win_rate", "profit_factor_R", "payoff_ratio_R", "net_R", "max_drawdown_pct", "risk_of_ruin_dd_25pct", "selection_tier", "candidate_pass_soft"] if c in best.columns]
    print(best[focus].sort_values(["symbol", "side", "selection_tier", metric], ascending=[True, True, False, False]).to_string(index=False))


if __name__ == "__main__":
    main()
