from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.features.feature_builder import build_features
from debco.utils.io import read_json, load_csv, save_csv, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--features-config", required=True)
    args = parser.parse_args()

    data_cfg = read_json(args.data_config)
    feat_cfg = read_json(args.features_config)
    input_dir = Path(feat_cfg.get("input_dir", data_cfg.get("output_dir", "data/raw")))
    output_dir = ensure_dir(feat_cfg.get("output_dir", "data/features"))
    pip_map = feat_cfg["pip_size"]

    for symbol in feat_cfg["symbols"]:
        path = input_dir / f"{symbol}_raw_with_dxy.csv"
        df = load_csv(path)
        features = build_features(df, symbol=symbol, pip_size=float(pip_map[symbol]), config=feat_cfg)
        out = save_csv(features, output_dir / f"{symbol}_features.csv")
        print(f"saved features: {out} rows={len(features)} cols={len(features.columns)}")


if __name__ == "__main__":
    main()
