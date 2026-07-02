from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ID_COLUMNS = {
    "date", "symbol", "open", "high", "low", "close", "tick_volume", "spread", "real_volume",
    "dxy_close", "dxy_inverse_close", "index_close",
}


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(path: Path, essential: list[str], *, model_cap: int | None = None) -> dict:
    df = pd.read_csv(path)
    missing = [c for c in essential if c not in df.columns]
    feature_cols = [c for c in df.columns if c not in ID_COLUMNS]
    warmup_cols = [c for c in essential if c in df.columns]
    report = {
        "file": str(path),
        "rows": len(df),
        "cols": len(df.columns),
        "feature_cols": len(feature_cols),
        "from": df["date"].iloc[0] if "date" in df.columns and len(df) else None,
        "to": df["date"].iloc[-1] if "date" in df.columns and len(df) else None,
        "nan_total": int(df.isna().sum().sum()),
        "missing_essential_columns": missing,
        "rows_after_dropna_essential": int(df.dropna(subset=warmup_cols).shape[0]) if warmup_cols else 0,
    }
    if model_cap is not None:
        report["model_cap"] = model_cap
        report["model_cap_ok"] = len(feature_cols) <= model_cap
    return report


def print_report(report: dict) -> None:
    for k, v in report.items():
        print(f"{k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-config", required=True)
    args = parser.parse_args()

    cfg = read_json(args.features_config)
    full_dir = Path(cfg.get("output", {}).get("full_dir", "data/features/full"))
    model_dir = Path(cfg.get("output", {}).get("model_dir", "data/features/model"))
    sanity = cfg.get("sanity", {})
    essential_full = sanity.get("essential_full_columns", [])
    essential_model = sanity.get("essential_model_columns", [])
    max_model = int(cfg.get("model_features", {}).get("max_features_with_lags", 100))

    any_error = False
    for symbol, symbol_cfg in cfg.get("symbols", {}).items():
        if not bool(symbol_cfg.get("enabled", True)):
            continue
        for kind, directory, suffix, essential, cap in [
            ("FULL", full_dir, "features_full", essential_full, None),
            ("MODEL", model_dir, "features_model", essential_model, max_model),
        ]:
            path = directory / f"{symbol}_{suffix}.csv"
            print(f"\n--- {kind} {symbol}: {path} ---")
            if not path.exists():
                print("MISSING FILE")
                any_error = True
                continue
            report = summarize(path, essential, model_cap=cap)
            print_report(report)
            if report["missing_essential_columns"]:
                any_error = True
            if cap is not None and not report.get("model_cap_ok", False):
                any_error = True

    if any_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
