from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.ml.candidates import list_candidate_experiments
from debco.trading.oof_eval import evaluate_job_trading
from debco.utils.io import ensure_dir, read_json


def _candidate_experiment_names(config: dict) -> list[str]:
    names = []
    for exp in list_candidate_experiments(config):
        names.append(str(exp.config.get("output", {}).get("experiment_name", exp.name)))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OOF predictions as trading signals with PF/DD/ROR metrics.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    parser.add_argument("--experiment", default=None, help="Specific ML result experiment folder name under data/ml_results.")
    parser.add_argument("--job", default=None, help="Optional exact job folder name to evaluate.")
    args = parser.parse_args()

    config = read_json(args.config)
    out_cfg = config.get("output", {})
    te = config.get("trading_eval", {})
    results_dir = Path(out_cfg.get("results_dir", "data/ml_results"))
    trading_root = ensure_dir(Path(te.get("output_dir", "data/trading_eval")))

    experiments = [args.experiment] if args.experiment else list(te.get("source_experiments", []))
    if not experiments:
        experiments = _candidate_experiment_names(config)
    if not experiments:
        experiments = [str(out_cfg.get("experiment_name", "xgb_results"))]

    global_rows = []
    for exp_name in experiments:
        exp_root = results_dir / str(exp_name)
        if not exp_root.exists():
            print(f"missing experiment results: {exp_root}")
            continue
        job_dirs = [p for p in exp_root.iterdir() if p.is_dir() and (p / "oof_predictions.csv").exists()]
        if args.job:
            job_dirs = [p for p in job_dirs if p.name == args.job]
        if not job_dirs:
            print(f"no job prediction folders found in {exp_root}")
            continue

        exp_out = ensure_dir(trading_root / str(exp_name))
        for job_dir in job_dirs:
            print(f"\n=== OOF TRADING EVAL {exp_name} :: {job_dir.name} ===")
            job_out = ensure_dir(exp_out / job_dir.name)
            result = evaluate_job_trading(config, job_dir=job_dir, job_name=job_dir.name)
            for name, df in result.items():
                if name == "enriched_predictions":
                    # Keep enriched predictions available but avoid huge console output.
                    pass
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
                combined.insert(0, "job", job_dir.name)
                combined.insert(0, "experiment", exp_name)
                combined.to_csv(job_out / "trading_summary.csv", index=False)
                global_rows.append(combined)
                focus_cols = [c for c in ["summary_table", "policy", "probability_column", "threshold", "top_percentile", "trade_count", "win_rate", "payoff_ratio", "profit_factor", "net_pips", "max_drawdown_pct", "risk_of_ruin_dd_25pct", "trades_per_month"] if c in combined.columns]
                print(combined[focus_cols].sort_values(["profit_factor", "trade_count"], ascending=[False, False]).head(12).to_string(index=False))

    if global_rows:
        all_summary = pd.concat(global_rows, ignore_index=True)
        all_path = trading_root / "all_trading_summary.csv"
        all_summary.to_csv(all_path, index=False)
        print(f"\nsaved global trading summary: {all_path} rows={len(all_summary)}")
    else:
        raise SystemExit("No trading evaluations were produced.")


if __name__ == "__main__":
    main()
