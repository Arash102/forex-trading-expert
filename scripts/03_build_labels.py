from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.labels.triple_barrier import (
    build_label_metadata,
    build_ml_ready_dataset,
    build_triple_barrier_labels,
    iter_label_jobs,
    join_features_with_labels,
    make_params_from_config,
)
from debco.utils.io import ensure_dir, load_csv, read_json, save_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build profile-aware binary Triple Barrier labels and strict ML-ready datasets.")
    parser.add_argument("--config", required=True, help="Path to labels config JSON.")
    args = parser.parse_args()

    cfg = read_json(args.config)
    input_cfg = cfg.get("input", {})
    output_cfg = cfg.get("output", {})
    features_dir = Path(input_cfg.get("features_dir", "data/features/model"))
    feature_template = str(input_cfg.get("file_template", "{symbol}_features_model.csv"))

    labels_dir = ensure_dir(output_cfg.get("labels_dir", "data/labels"))
    label_template = str(output_cfg.get("file_template", "{symbol}_{profile}_labels_{side}.csv"))

    # Audit/debug joined dataset: can include OHLC and label metadata.
    write_joined = bool(output_cfg.get("write_joined_dataset", False))
    joined_template = str(output_cfg.get("joined_file_template", "{symbol}_{profile}_dataset_{side}.csv"))

    # Strict training table: configured model feature columns + one binary label only.
    write_ml_ready = bool(output_cfg.get("write_ml_ready_dataset", True))
    ml_ready_dir = ensure_dir(output_cfg.get("ml_ready_dir", "data/ml_ready"))
    ml_ready_template = str(output_cfg.get("ml_ready_file_template", "{symbol}_{profile}_ml_ready_{side}.csv"))
    ml_target_column = str(output_cfg.get("ml_target_column", "label"))
    dropna_ml_ready = bool(output_cfg.get("dropna_ml_ready", False))

    # Non-model metadata for validation/debug. This is never an XGBoost input.
    write_metadata = bool(output_cfg.get("write_label_metadata", True))
    metadata_dir = ensure_dir(output_cfg.get("metadata_dir", "data/label_metadata"))
    metadata_template = str(output_cfg.get("metadata_file_template", "{symbol}_{profile}_metadata_{side}.csv"))

    feature_cache = {}
    for job in iter_label_jobs(cfg):
        if job.symbol not in feature_cache:
            features_path = features_dir / feature_template.format(symbol=job.symbol)
            feature_cache[job.symbol] = load_csv(features_path, parse_dates=("date",))
        features = feature_cache[job.symbol]

        params = make_params_from_config(cfg, job.symbol, job.side, job.profile)
        labels = build_triple_barrier_labels(features, params)

        label_path = labels_dir / label_template.format(symbol=job.symbol, profile=job.profile, side=job.side)
        save_csv(labels, label_path)
        counts = labels["label"].value_counts(dropna=False).sort_index().to_dict() if len(labels) else {}
        events = labels["event_type"].value_counts(dropna=False).to_dict() if len(labels) else {}
        print(
            f"saved labels: {label_path} rows={len(labels)} "
            f"positive_ratio={(float(labels['label'].mean()) if len(labels) else 0.0):.6f} "
            f"counts={counts} events={events}"
        )

        if write_joined:
            dataset = join_features_with_labels(features, labels)
            dataset_path = labels_dir / joined_template.format(symbol=job.symbol, profile=job.profile, side=job.side)
            save_csv(dataset, dataset_path)
            print(f"saved joined audit dataset: {dataset_path} rows={len(dataset)} cols={len(dataset.columns)}")

        if write_ml_ready:
            ml_ready = build_ml_ready_dataset(
                features,
                labels,
                output_target_column=ml_target_column,
                dropna=dropna_ml_ready,
            )
            ml_ready_path = ml_ready_dir / ml_ready_template.format(symbol=job.symbol, profile=job.profile, side=job.side)
            save_csv(ml_ready, ml_ready_path)
            feature_count = len(ml_ready.columns) - 1
            print(f"saved ml-ready dataset: {ml_ready_path} rows={len(ml_ready)} feature_cols={feature_count} target={ml_target_column}")

        if write_metadata:
            meta = build_label_metadata(labels)
            meta_path = metadata_dir / metadata_template.format(symbol=job.symbol, profile=job.profile, side=job.side)
            save_csv(meta, meta_path)
            print(f"saved label metadata: {meta_path} rows={len(meta)} cols={len(meta.columns)}")


if __name__ == "__main__":
    main()
