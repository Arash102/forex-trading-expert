from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.ml.candidates import list_candidate_experiments
from debco.ml.reporting import collect_job_comparison
from debco.ml.xgb_optuna import TrainingJob, enabled_jobs, run_training_job
from debco.utils.io import ensure_dir, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost on enabled candidate-based/meta-label candidate sets.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    parser.add_argument("--candidate-set", default=None, help="Optional single candidate set name to train.")
    parser.add_argument("--job", default=None, help="Optional exact job name, e.g. EURUSD_fast_15_8_h16_short.")
    parser.add_argument("--symbol", default=None, help="Optional symbol filter, e.g. EURUSD or XAUUSD.")
    parser.add_argument("--profile", default=None, help="Optional profile filter, e.g. fast_15_8_h16.")
    parser.add_argument("--side", default=None, choices=["long", "short"], help="Optional side filter.")
    parser.add_argument("--max-jobs", type=int, default=None, help="Optional limit for quick smoke tests after filters.")
    args = parser.parse_args()

    base_cfg = read_json(args.config)
    base_jobs = enabled_jobs(base_cfg)
    if not base_jobs:
        raise SystemExit("No enabled ML jobs found in config.jobs.")

    def _job_matches(job: TrainingJob) -> bool:
        if args.job and job.name != args.job:
            return False
        if args.symbol and job.symbol != args.symbol:
            return False
        if args.profile and job.profile != args.profile:
            return False
        if args.side and job.side != args.side:
            return False
        return True

    base_jobs = [job for job in base_jobs if _job_matches(job)]
    if not base_jobs:
        raise SystemExit("No enabled ML jobs matched the requested filters.")
    if args.max_jobs is not None:
        base_jobs = base_jobs[: int(args.max_jobs)]

    print("selected jobs:", ", ".join(job.name for job in base_jobs))

    experiments = list_candidate_experiments(base_cfg)
    trained_any = False
    global_rows = []
    for exp in experiments:
        if args.candidate_set and exp.name != args.candidate_set:
            continue
        cfg = exp.config
        out_cfg = cfg.get("output", {})
        output_root = ensure_dir(Path(out_cfg.get("results_dir", "data/ml_results")) / str(out_cfg.get("experiment_name", f"candidate_{exp.name}")))
        print(f"\n############################")
        print(f"### CANDIDATE SET: {exp.name}")
        print(f"### OUTPUT: {output_root}")
        print(f"############################")
        summaries = []
        for job in base_jobs:
            print(f"\n=== TRAIN {exp.name} :: {job.name} ===")
            row = run_training_job(cfg, job, Path(output_root))
            row["candidate_set"] = exp.name
            summaries.append(row)
            global_rows.append({"experiment": str(out_cfg.get("experiment_name", exp.name)), **row})
            trained_any = True
        summary_df = pd.DataFrame(summaries)
        summary_path = Path(output_root) / "run_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nsaved run summary: {summary_path}")

        comparison = collect_job_comparison(Path(output_root))
        if not comparison.empty:
            comparison.insert(0, "candidate_set", exp.name)
            comparison_path = Path(output_root) / "job_comparison.csv"
            comparison.to_csv(comparison_path, index=False)
            print(f"saved job comparison: {comparison_path}")

    if not trained_any:
        raise SystemExit("No candidate sets matched the request.")

    out_cfg = base_cfg.get("output", {})
    global_root = ensure_dir(Path(out_cfg.get("results_dir", "data/ml_results")) / "candidate_global_v0_1_6")
    global_path = Path(global_root) / "candidate_training_summary.csv"
    pd.DataFrame(global_rows).to_csv(global_path, index=False)
    print(f"\nsaved global candidate training summary: {global_path}")


if __name__ == "__main__":
    main()
