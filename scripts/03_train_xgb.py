from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.utils.io import read_json, load_csv
from debco.labels.triple_barrier import triple_barrier_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = read_json(args.config)
    df = load_csv(cfg["dataset_path"])
    target = cfg["target"]
    # pip size is intentionally explicit; add symbol-aware config in next iteration.
    pip_size = 0.0001 if "EURUSD" in cfg["dataset_path"].upper() else 0.01
    labeled = triple_barrier_labels(
        df,
        side=target["side"],
        take_profit_pips=float(target["take_profit_pips"]),
        stop_loss_pips=float(target["stop_loss_pips"]),
        horizon_bars=int(target["horizon_bars"]),
        pip_size=pip_size,
        same_bar_policy=target.get("same_bar_policy", "sl_first"),
    )
    print(labeled["tb_label"].value_counts(dropna=False).sort_index())
    print("ML training loop is intentionally next step after feature ground truth confirmation.")


if __name__ == "__main__":
    main()
