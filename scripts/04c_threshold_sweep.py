from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.ml.thresholding import threshold_sweep_from_predictions
from debco.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild threshold sweeps from OOF prediction files.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    args = parser.parse_args()
    cfg = read_json(args.config)
    out_cfg = cfg.get("output", {})
    root = Path(out_cfg.get("results_dir", "data/ml_results")) / str(out_cfg.get("experiment_name", "xgb_experiment"))
    sw_cfg = cfg.get("threshold_sweep", {})
    thresholds = [float(x) for x in sw_cfg.get("thresholds", [0.5, 0.6, 0.7])]
    prob_cols = [str(x) for x in sw_cfg.get("probability_columns", ["y_prob_calibrated", "y_prob_raw"])]
    if not root.exists():
        raise SystemExit(f"Result root not found: {root}")
    for pred_path in sorted(root.glob("*/oof_predictions.csv")):
        preds = pd.read_csv(pred_path)
        sweep = threshold_sweep_from_predictions(preds, thresholds=thresholds, probability_columns=prob_cols, y_col="y_true")
        out_path = pred_path.parent / "threshold_sweep.csv"
        sweep.to_csv(out_path, index=False)
        print(f"saved threshold sweep: {out_path} rows={len(sweep)}")


if __name__ == "__main__":
    main()
