from __future__ import annotations

from pathlib import Path

import pandas as pd


def collect_job_comparison(results_root: Path) -> pd.DataFrame:
    rows = []
    root = Path(results_root)
    for job_dir in sorted([p for p in root.iterdir() if p.is_dir()] if root.exists() else []):
        summary_path = job_dir / "metrics_summary.csv"
        sweep_path = job_dir / "threshold_sweep.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            row = {"job": job_dir.name}
            for metric in ["average_precision", "roc_auc", "precision", "recall", "specificity", "balanced_accuracy", "mcc", "brier_score", "ece"]:
                sub = summary[summary["metric"] == metric]
                if not sub.empty:
                    row[f"{metric}_mean"] = float(sub["mean"].iloc[0])
            rows.append(row)
        if sweep_path.exists():
            sweep = pd.read_csv(sweep_path)
            if not sweep.empty:
                # Add one best-threshold row per probability column by MCC.
                for prob_col, sub in sweep[sweep["fold"].eq("ALL")].groupby("probability_column", sort=False):
                    if "mcc" not in sub.columns or sub.empty:
                        continue
                    best = sub.sort_values(["mcc", "precision"], ascending=False).iloc[0]
                    rows.append({
                        "job": job_dir.name,
                        "summary_type": f"best_threshold_{prob_col}",
                        "best_threshold": float(best["threshold"]),
                        "best_mcc": float(best["mcc"]),
                        "best_precision": float(best["precision"]),
                        "best_recall": float(best["recall"]),
                        "best_signal_rate": float(best["signal_rate"]),
                    })
    return pd.DataFrame(rows)
