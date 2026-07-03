from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.ml.xgb_optuna import enabled_jobs, run_training_job
from debco.utils.io import ensure_dir, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost models with Optuna and walk-forward/CPCV validation.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    args = parser.parse_args()

    cfg = read_json(args.config)
    jobs = enabled_jobs(cfg)
    if not jobs:
        raise SystemExit("No enabled ML jobs found in config.jobs.")
    out_cfg = cfg.get("output", {})
    output_root = ensure_dir(Path(out_cfg.get("results_dir", "data/ml_results")) / str(out_cfg.get("experiment_name", "xgb_experiment")))
    summaries = []
    for job in jobs:
        print(f"\n=== TRAIN {job.name} ===")
        summaries.append(run_training_job(cfg, job, Path(output_root)))
    summary_df = pd.DataFrame(summaries)
    summary_path = Path(output_root) / "run_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nsaved run summary: {summary_path}")


if __name__ == "__main__":
    main()
