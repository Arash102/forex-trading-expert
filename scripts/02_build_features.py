from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.features.feature_builder import build_feature_outputs


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--features-config", required=True)
    args = parser.parse_args()

    data_cfg = read_json(args.data_config)
    feat_cfg = read_json(args.features_config)

    input_dir = Path(feat_cfg.get("input_dir", data_cfg.get("output_dir", "data/raw")))
    full_dir = ensure_dir(feat_cfg.get("output", {}).get("full_dir", "data/features/full"))
    model_dir = ensure_dir(feat_cfg.get("output", {}).get("model_dir", "data/features/model"))

    symbols_cfg = feat_cfg.get("symbols", {})
    if not symbols_cfg:
        raise ValueError("features config has no symbols block.")

    for symbol, symbol_cfg in symbols_cfg.items():
        if not bool(symbol_cfg.get("enabled", True)):
            print(f"skip disabled symbol: {symbol}")
            continue
        path = input_dir / f"{symbol}_raw_with_dxy.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        df = pd.read_csv(path)
        full, model = build_feature_outputs(df, symbol=symbol, config=feat_cfg)

        full_path = full_dir / f"{symbol}_features_full.csv"
        model_path = model_dir / f"{symbol}_features_model.csv"
        full.to_csv(full_path, index=False)
        model.to_csv(model_path, index=False)

        model_feature_count = len([c for c in model.columns if c not in {"date", "symbol", "open", "high", "low", "close", "tick_volume", "spread", "real_volume", "dxy_close", "dxy_inverse_close", "index_close"}])
        print(f"saved full features : {full_path} rows={len(full)} cols={len(full.columns)}")
        print(f"saved model features: {model_path} rows={len(model)} cols={len(model.columns)} model_features={model_feature_count}")
        print(f"  date_from={full['date'].iloc[0]} date_to={full['date'].iloc[-1]}")


if __name__ == "__main__":
    main()
