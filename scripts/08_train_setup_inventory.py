from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.ml.reporting import collect_job_comparison
from debco.ml.setup_inventory import config_for_setup, list_setup_specs, setup_matches_filters
from debco.ml.xgb_optuna import run_training_job
from debco.utils.io import ensure_dir, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train setup-specific meta-labeling models from setup_inventory.setups.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    parser.add_argument("--setup-id", default=None, help="Optional exact setup_id.")
    parser.add_argument("--symbol", default=None, help="Optional symbol filter.")
    parser.add_argument("--side", default=None, choices=["long", "short"], help="Optional side filter.")
    parser.add_argument("--family", default=None, help="Optional family filter.")
    parser.add_argument("--max-setups", type=int, default=None, help="Optional limit for smoke tests.")
    args = parser.parse_args()

    base_cfg = read_json(args.config)
    specs = [
        s for s in list_setup_specs(base_cfg)
        if setup_matches_filters(s, setup_id=args.setup_id, symbol=args.symbol, side=args.side, family=args.family)
    ]
    if args.max_setups is not None:
        specs = specs[: int(args.max_setups)]
    if not specs:
        raise SystemExit("No setup specs matched the requested filters.")

    results_dir = Path(base_cfg.get("output", {}).get("results_dir", "data/ml_results"))
    global_rows = []
    for spec in specs:
        cfg = config_for_setup(base_cfg, spec)
        exp_name = str(cfg.get("output", {}).get("experiment_name"))
        output_root = ensure_dir(results_dir / exp_name)
        job = spec.job
        print("\n############################")
        print(f"### SETUP: {spec.setup_id}")
        print(f"### JOB:   {job.name}")
        print(f"### OUT:   {output_root}")
        print("############################")
        row = run_training_job(cfg, job, output_root)
        row.insert(0, "setup_id", spec.setup_id) if hasattr(row, "insert") else None
        row["setup_id"] = spec.setup_id
        row["family"] = spec.family
        row["candidate_set"] = spec.candidate_set_name
        summary_path = output_root / "run_summary.csv"
        pd.DataFrame([row]).to_csv(summary_path, index=False)
        comparison = collect_job_comparison(output_root)
        if not comparison.empty:
            comparison.insert(0, "setup_id", spec.setup_id)
            comparison.to_csv(output_root / "job_comparison.csv", index=False)
        global_rows.append({"experiment": exp_name, "setup_id": spec.setup_id, "job": job.name, **row})

    out_dir = ensure_dir(Path(base_cfg.get("setup_inventory", {}).get("output_dir", "data/setup_inventory")))
    summary_path = out_dir / "setup_training_summary.csv"
    new_df = pd.DataFrame(global_rows)

    # When this script is called once per setup from a terminal loop, do not
    # overwrite previous setup rows. Replace matching experiment/setup/job rows
    # and append new ones so the summary remains a true inventory.
    if summary_path.exists():
        old_df = pd.read_csv(summary_path)
        if not old_df.empty and not new_df.empty:
            key_cols = [c for c in ["experiment", "setup_id", "job"] if c in old_df.columns and c in new_df.columns]
            if key_cols:
                old_keys = old_df[key_cols].astype(str).agg("||".join, axis=1)
                new_keys = set(new_df[key_cols].astype(str).agg("||".join, axis=1))
                old_df = old_df.loc[~old_keys.isin(new_keys)].copy()
            new_df = pd.concat([old_df, new_df], ignore_index=True, sort=False)

    if not new_df.empty:
        sort_cols = [c for c in ["symbol", "side", "setup_id", "experiment", "job"] if c in new_df.columns]
        if sort_cols:
            new_df = new_df.sort_values(sort_cols).reset_index(drop=True)
    new_df.to_csv(summary_path, index=False)
    print(f"\nsaved setup training summary: {summary_path}")


if __name__ == "__main__":
    main()
