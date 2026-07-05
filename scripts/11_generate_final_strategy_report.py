from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.reporting.final_strategy_report import write_report_bundle
from debco.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final strategy report and live execution specification from inventory ROR results.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    args = parser.parse_args()

    config = read_json(args.config)
    outputs = write_report_bundle(config)
    print("\n--- FINAL STRATEGY REPORT OUTPUTS ---")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
