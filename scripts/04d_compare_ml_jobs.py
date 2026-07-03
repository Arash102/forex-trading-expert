from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.ml.reporting import collect_job_comparison
from debco.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect all ML job results into one comparison CSV.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    args = parser.parse_args()
    cfg = read_json(args.config)
    out_cfg = cfg.get("output", {})
    root = Path(out_cfg.get("results_dir", "data/ml_results")) / str(out_cfg.get("experiment_name", "xgb_experiment"))
    comparison = collect_job_comparison(root)
    if comparison.empty:
        raise SystemExit(f"No job results found under: {root}")
    out_path = root / "job_comparison.csv"
    comparison.to_csv(out_path, index=False)
    print(f"saved job comparison: {out_path} rows={len(comparison)}")


if __name__ == "__main__":
    main()
