from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.ml.candidates import candidate_mask_for_job, list_candidate_experiments
from debco.ml.xgb_optuna import (
    apply_missing_strategy,
    build_candidate_aware_validation_folds,
    candidate_stats_from_mask,
    enabled_jobs,
    load_job_frames,
    split_xy,
)
from debco.utils.io import ensure_dir, read_json


def candidate_summary(cfg: dict, candidate_set: str, job) -> dict:
    ml_ready, metadata = load_job_frames(cfg, job)
    target = str(cfg.get("sanity", {}).get("target_column", "label"))
    x, y = split_xy(ml_ready, target_col=target)
    metadata = metadata.iloc[: len(x)].reset_index(drop=True)
    missing_cfg = cfg.get("missing_values", {})
    x, y, metadata = apply_missing_strategy(x, y, metadata, dropna=bool(missing_cfg.get("dropna", False)))
    mask = candidate_mask_for_job(x, symbol=job.symbol, side=job.side, config=cfg)
    mask = pd.Series(mask, index=x.index).fillna(False).astype(bool).reset_index(drop=True)
    x2 = x.loc[mask].reset_index(drop=True)
    y2 = y.loc[mask].reset_index(drop=True)
    stats = candidate_stats_from_mask(x, y, mask, config=cfg)
    folds, fold_stats = build_candidate_aware_validation_folds(cfg, metadata, mask, y)
    row = {
        "candidate_set": candidate_set,
        "job": job.name,
        "symbol": job.symbol,
        "profile": job.profile,
        "side": job.side,
        "rows_before": len(x),
        "rows_after": len(x2),
        "keep_ratio": float(len(x2) / len(x)) if len(x) else 0.0,
        "positive_rate_before": float((y == 1).mean()) if len(y) else 0.0,
        "positive_rate_after": float((y2 == 1).mean()) if len(y2) else 0.0,
        "positive_lift": (float((y2 == 1).mean()) / float((y == 1).mean())) if len(y2) and float((y == 1).mean()) > 0 else 0.0,
        "positive_count_after": int((y2 == 1).sum()) if len(y2) else 0,
        "negative_count_after": int((y2 == 0).sum()) if len(y2) else 0,
        "fold_count_after": len(folds),
    }
    row.update(stats)
    row.update(fold_stats)
    if len(x2) and "session_block_id" in x2.columns:
        counts = x2["session_block_id"].value_counts(dropna=False).sort_index().to_dict()
        row["session_block_counts_after"] = str(counts)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check candidate filters before expensive XGBoost training.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    parser.add_argument("--candidate-set", default=None, help="Optional single candidate set name to check.")
    parser.add_argument("--save", action="store_true", help="Save candidate summary CSV under data/ml_results.")
    args = parser.parse_args()

    base_cfg = read_json(args.config)
    jobs = enabled_jobs(base_cfg)
    if not jobs:
        raise SystemExit("No enabled ML jobs found in config.jobs.")

    rows = []
    experiments = list_candidate_experiments(base_cfg)
    for exp in experiments:
        if args.candidate_set and exp.name != args.candidate_set:
            continue
        print(f"\n=== CANDIDATE SET {exp.name} ===")
        for job in jobs:
            row = candidate_summary(exp.config, exp.name, job)
            rows.append(row)
            print(
                f"{job.name}: rows {row['rows_before']} -> {row['rows_after']} "
                f"keep={row['keep_ratio']:.3f} pos {row['positive_rate_before']:.3f} -> {row['positive_rate_after']:.3f} "
                f"lift={row['positive_lift']:.3f} folds={row['fold_count_after']} "
                f"base_folds={int(row.get('base_fold_count', 0))} skipped={int(row.get('candidate_folds_skipped', 0))}"
            )

    if not rows:
        raise SystemExit("No candidate sets matched the request.")
    df = pd.DataFrame(rows)
    print("\n--- CANDIDATE SUMMARY ---")
    cols = [
        "candidate_set", "job", "rows_after", "keep_ratio",
        "positive_rate_before", "positive_rate_after", "positive_lift",
        "positive_count_after", "fold_count_after",
    ]
    print(df[cols].to_string(index=False))

    if args.save:
        out_cfg = base_cfg.get("output", {})
        root = ensure_dir(Path(out_cfg.get("results_dir", "data/ml_results")) / "candidate_sanity_v0_1_6")
        out_path = Path(root) / "candidate_summary.csv"
        df.to_csv(out_path, index=False)
        print(f"\nsaved candidate summary: {out_path}")


if __name__ == "__main__":
    main()
