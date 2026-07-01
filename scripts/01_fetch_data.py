from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.data.mt5_loader import MT5DataLoader
from debco.data.dxy import build_dxy_from_components, merge_symbol_with_dxy
from debco.utils.io import read_json, save_csv, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = read_json(args.config)
    out_dir = ensure_dir(cfg.get("output_dir", "data/raw"))
    timeframe = cfg["timeframe"]
    bars = int(cfg["bars"])

    loader = MT5DataLoader()
    loader.connect()
    try:
        component_frames = {}
        for name, item in cfg["dxy_components"].items():
            df = loader.fetch_bars(item["broker_symbol"], timeframe, bars)
            component_frames[name] = df
            save_csv(df, out_dir / f"{name}_raw.csv")

        dxy = build_dxy_from_components(
            component_frames,
            constant=float(cfg.get("dxy_constant", 50.14348112)),
            weights={name: float(item["weight"]) for name, item in cfg["dxy_components"].items()},
        )
        save_csv(dxy, out_dir / "DXY_built.csv")

        for canonical, item in cfg["symbols"].items():
            if not item.get("enabled", True):
                continue
            broker_symbol = item["broker_symbol"]
            if canonical in component_frames and broker_symbol == cfg["dxy_components"].get(canonical, {}).get("broker_symbol"):
                raw = component_frames[canonical]
            else:
                raw = loader.fetch_bars(broker_symbol, timeframe, bars)
            save_csv(raw, out_dir / f"{canonical}_raw.csv")
            merged = merge_symbol_with_dxy(raw, dxy)
            save_csv(merged, out_dir / f"{canonical}_raw_with_dxy.csv")
            print(f"saved {canonical}: raw={len(raw)} merged={len(merged)}")
    finally:
        loader.shutdown()


if __name__ == "__main__":
    main()
