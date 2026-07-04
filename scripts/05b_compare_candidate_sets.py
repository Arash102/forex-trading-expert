from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.ml.candidates import list_candidate_experiments
from debco.ml.reporting import collect_job_comparison
from debco.utils.io import ensure_dir, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect candidate-set comparisons across v0.1.6 experiments.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    args = parser.parse_args()

    cfg = read_json(args.config)
    out_cfg = cfg.get("output", {})
    results_dir = Path(out_cfg.get("results_dir", "data/ml_results"))
    rows = []
    for exp in list_candidate_experiments(cfg):
        exp_name = str(exp.config.get("output", {}).get("experiment_name", exp.name))
        root = results_dir / exp_name
        comparison = collect_job_comparison(root)
        if comparison.empty:
            print(f"missing or empty: {root}")
            continue
        comparison.insert(0, "candidate_set", exp.name)
        comparison.insert(1, "experiment_name", exp_name)
        rows.append(comparison)

    if not rows:
        raise SystemExit("No candidate comparison rows found. Run 05_train_candidate_sets.py first.")
    out = pd.concat(rows, ignore_index=True)
    global_root = ensure_dir(results_dir / "candidate_global_v0_1_6")
    out_path = Path(global_root) / "candidate_set_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"saved candidate-set comparison: {out_path} rows={len(out)}")

    focus = out[out.get("summary_type", pd.Series(index=out.index, dtype=object)).astype(str).str.contains("best_threshold_y_prob_raw", na=False)].copy()
    if not focus.empty:
        cols = ["candidate_set", "job", "best_threshold", "best_mcc", "best_precision", "best_recall", "best_signal_rate"]
        print("\n--- BEST RAW THRESHOLDS BY CANDIDATE SET ---")
        print(focus[cols].sort_values(["best_mcc", "best_precision"], ascending=False).head(40).to_string(index=False))


if __name__ == "__main__":
    main()
