from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.ml.xgb_optuna import enabled_jobs, load_job_frames, split_xy, build_validation_folds
from debco.utils.io import read_json


def check_job(cfg: dict, job) -> None:
    print(f"\n--- ML INPUT {job.name} ---")
    ml_ready, metadata = load_job_frames(cfg, job)
    target = str(cfg.get("sanity", {}).get("target_column", "label"))
    expected_features = int(cfg.get("sanity", {}).get("expected_feature_count", 100))
    forbidden = set(cfg.get("sanity", {}).get("forbidden_ml_columns", []))
    print("ml_rows:", len(ml_ready))
    print("ml_cols:", len(ml_ready.columns))
    print("metadata_rows:", len(metadata))
    print("target_exists:", target in ml_ready.columns)
    if target not in ml_ready.columns:
        return
    feature_cols = [c for c in ml_ready.columns if c != target]
    forbidden_found = [c for c in feature_cols if c in forbidden]
    print("feature_cols:", len(feature_cols))
    print("expected_feature_count:", expected_features)
    print("feature_count_ok:", len(feature_cols) == expected_features)
    print("forbidden_columns:", forbidden_found)
    print("strict_ml_ready_ok:", not forbidden_found)
    print("target_counts:", ml_ready[target].value_counts(dropna=False).sort_index().to_dict())
    print("positive_ratio:", round(float((ml_ready[target] == 1).mean()), 6))
    print("nan_total:", int(ml_ready.isna().sum().sum()))
    x, y = split_xy(ml_ready, target_col=target)
    folds = build_validation_folds(cfg, metadata.iloc[: len(x)].reset_index(drop=True))
    print("validation_method:", cfg.get("validation", {}).get("method", "walk_forward"))
    print("fold_count:", len(folds))
    for fold in folds[:5]:
        train_pos = float((y.iloc[fold.train_idx] == 1).mean()) if len(fold.train_idx) else 0.0
        test_pos = float((y.iloc[fold.test_idx] == 1).mean()) if len(fold.test_idx) else 0.0
        print("fold:", fold.fold_id, "train:", len(fold.train_idx), "test:", len(fold.test_idx), "train_pos:", round(train_pos, 4), "test_pos:", round(test_pos, 4))


def summarize_results(cfg: dict) -> None:
    out_cfg = cfg.get("output", {})
    root = Path(out_cfg.get("results_dir", "data/ml_results")) / str(out_cfg.get("experiment_name", "xgb_experiment"))
    if not root.exists():
        print(f"\n--- ML RESULTS ---\nNo result directory yet: {root}")
        return
    print(f"\n--- ML RESULTS {root} ---")
    run_summary = root / "run_summary.csv"
    comparison = root / "job_comparison.csv"
    if run_summary.exists():
        print(f"\n{run_summary}")
        print(pd.read_csv(run_summary).head(20).to_string(index=False))
    if comparison.exists():
        print(f"\n{comparison}")
        print(pd.read_csv(comparison).head(30).to_string(index=False))
    for summary_path in sorted(root.glob("*/metrics_summary.csv")):
        print(f"\n{summary_path}")
        df = pd.read_csv(summary_path)
        focus_metrics = ["average_precision", "roc_auc", "precision", "recall", "specificity", "balanced_accuracy", "mcc", "brier_score", "calibrated_brier_score", "raw_brier_score", "calibrated_ece", "raw_ece"]
        focus = df[df["metric"].isin(focus_metrics)]
        print(focus.to_string(index=False))
        sweep_path = summary_path.parent / "threshold_sweep.csv"
        if sweep_path.exists():
            sweep = pd.read_csv(sweep_path)
            all_rows = sweep[sweep["fold"].eq("ALL")]
            if not all_rows.empty:
                best = all_rows.sort_values(["mcc", "precision"], ascending=False).head(5)
                print("\nTop thresholds by MCC:")
                print(best[["probability_column", "threshold", "precision", "recall", "specificity", "mcc", "signal_rate"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check ML-ready files, validation folds, and optional ML results.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    args = parser.parse_args()
    cfg = read_json(args.config)
    jobs = enabled_jobs(cfg)
    if not jobs:
        raise SystemExit("No enabled ML jobs found in config.jobs.")
    for job in jobs:
        check_job(cfg, job)
    summarize_results(cfg)


if __name__ == "__main__":
    main()
