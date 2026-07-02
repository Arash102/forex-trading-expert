from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.labels.triple_barrier import iter_label_jobs
from debco.utils.io import read_json


MODEL_AUDIT_FORBIDDEN_COLUMNS = {
    "date", "symbol", "open", "high", "low", "close", "tick_volume", "spread", "real_volume",
    "dxy_close", "dxy_inverse_close", "index_close",
    "outcome_label", "event_type", "entry_date", "exit_date", "entry_price", "exit_price",
    "tp_pips", "sl_pips", "max_horizon_bars", "realized_pips", "mfe_pips", "mae_pips",
    "same_bar_hit", "bars_to_event", "profile", "side", "target_name",
}


def summarize_label_file(path: Path, required: list[str], min_positive_ratio: float, max_same_bar_ratio: float) -> None:
    print(f"\n--- LABEL AUDIT {path} ---")
    if not path.exists():
        print("MISSING")
        return
    df = pd.read_csv(path)
    print("rows:", len(df))
    print("cols:", len(df.columns))
    missing = [c for c in required if c not in df.columns]
    print("missing_required_columns:", missing)
    if len(df) == 0 or missing:
        return

    label_counts = df["label"].value_counts(dropna=False).sort_index().to_dict()
    outcome_counts = df["outcome_label"].value_counts(dropna=False).sort_index().to_dict() if "outcome_label" in df.columns else {}
    event_counts = df["event_type"].value_counts(dropna=False).to_dict()
    positive_ratio = float((df["label"] == 1).mean())
    sl_ratio = float((df.get("outcome_label", pd.Series([0] * len(df))) == -1).mean())
    vertical_ratio = float((df["event_type"] == "vertical").mean()) if "event_type" in df.columns else 0.0
    same_bar_ratio = float(df.get("same_bar_hit", pd.Series([0] * len(df))).fillna(0).astype(float).mean())

    print("label_counts_binary:", label_counts)
    print("outcome_counts_signed:", outcome_counts)
    print("event_counts:", event_counts)
    print("positive_ratio:", round(positive_ratio, 6))
    print("sl_ratio:", round(sl_ratio, 6))
    print("vertical_ratio:", round(vertical_ratio, 6))
    print("same_bar_ratio:", round(same_bar_ratio, 6))
    print("min_positive_ratio:", min_positive_ratio)
    print("positive_ok:", positive_ratio >= min_positive_ratio)
    print("max_same_bar_ratio:", max_same_bar_ratio)
    print("same_bar_ok:", same_bar_ratio <= max_same_bar_ratio)
    if "date" in df.columns:
        print("from:", df["date"].iloc[0])
        print("to:", df["date"].iloc[-1])
    if "realized_pips" in df.columns:
        print("realized_pips_mean:", round(float(pd.to_numeric(df["realized_pips"], errors="coerce").mean()), 6))
        print("realized_pips_median:", round(float(pd.to_numeric(df["realized_pips"], errors="coerce").median()), 6))
    if "bars_to_event" in df.columns:
        print("bars_to_event_mean:", round(float(pd.to_numeric(df["bars_to_event"], errors="coerce").mean()), 6))


def summarize_ml_ready_file(path: Path, target_col: str, max_features: int) -> None:
    print(f"\n--- ML READY {path} ---")
    if not path.exists():
        print("MISSING")
        return
    df = pd.read_csv(path)
    print("rows:", len(df))
    print("cols:", len(df.columns))
    print("target_col:", target_col)
    print("target_exists:", target_col in df.columns)
    if target_col not in df.columns:
        return
    feature_cols = [c for c in df.columns if c != target_col]
    forbidden = [c for c in df.columns if c in MODEL_AUDIT_FORBIDDEN_COLUMNS]
    print("feature_cols:", len(feature_cols))
    print("max_features:", max_features)
    print("feature_cap_ok:", len(feature_cols) <= max_features)
    print("forbidden_non_feature_columns:", forbidden)
    print("strict_ml_ready_ok:", len(forbidden) == 0)
    print("target_counts:", df[target_col].value_counts(dropna=False).sort_index().to_dict())
    print("positive_ratio:", round(float((df[target_col] == 1).mean()), 6))
    print("nan_total:", int(df.isna().sum().sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check profile-aware binary Triple Barrier label and ML-ready files.")
    parser.add_argument("--config", required=True, help="Path to labels config JSON.")
    args = parser.parse_args()
    cfg = read_json(args.config)

    output_cfg = cfg.get("output", {})
    sanity_cfg = cfg.get("sanity", {})
    labels_dir = Path(output_cfg.get("labels_dir", "data/labels"))
    label_template = str(output_cfg.get("file_template", "{symbol}_{profile}_labels_{side}.csv"))
    required = list(sanity_cfg.get("required_label_columns", ["date", "label", "outcome_label", "event_type"]))
    min_positive_ratio = float(sanity_cfg.get("min_positive_ratio", 0.01))
    max_same_bar_ratio = float(sanity_cfg.get("max_same_bar_ratio", 0.05))

    ml_ready_dir = Path(output_cfg.get("ml_ready_dir", "data/ml_ready"))
    ml_ready_template = str(output_cfg.get("ml_ready_file_template", "{symbol}_{profile}_ml_ready_{side}.csv"))
    target_col = str(output_cfg.get("ml_target_column", "label"))
    max_features = int(cfg.get("model_features", {}).get("max_features_with_lags", 100))

    for job in iter_label_jobs(cfg):
        label_path = labels_dir / label_template.format(symbol=job.symbol, profile=job.profile, side=job.side)
        summarize_label_file(label_path, required, min_positive_ratio, max_same_bar_ratio)

        ml_path = ml_ready_dir / ml_ready_template.format(symbol=job.symbol, profile=job.profile, side=job.side)
        summarize_ml_ready_file(ml_path, target_col, max_features)


if __name__ == "__main__":
    main()
