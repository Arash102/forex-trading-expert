from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.ml.setup_inventory import list_setup_specs, setup_candidate_summary, setup_matches_filters
from debco.utils.io import ensure_dir, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check setup-specific candidate inventory before training.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    parser.add_argument("--setup-id", default=None, help="Optional exact setup_id, e.g. EUR_AH_ATR2_BUY.")
    parser.add_argument("--symbol", default=None, help="Optional symbol filter: EURUSD or XAUUSD.")
    parser.add_argument("--side", default=None, choices=["long", "short"], help="Optional side filter.")
    parser.add_argument("--family", default=None, help="Optional family filter.")
    parser.add_argument("--save", action="store_true", help="Save setup_raw_audit.csv under data/setup_inventory.")
    args = parser.parse_args()

    config = read_json(args.config)
    specs = [
        s for s in list_setup_specs(config)
        if setup_matches_filters(s, setup_id=args.setup_id, symbol=args.symbol, side=args.side, family=args.family)
    ]
    if not specs:
        raise SystemExit("No setup specs matched the requested filters.")

    rows = []
    print("\n=== SETUP INVENTORY SANITY ===")
    for spec in specs:
        row = setup_candidate_summary(config, spec)
        rows.append(row)
        print(
            f"{spec.setup_id:32s} {row['job']:35s} rows {row['rows_before']} -> {row['rows_after']} "
            f"keep={row['keep_ratio']:.3f} pos {row['positive_rate_before']:.3f}->{row['positive_rate_after']:.3f} "
            f"lift={row['positive_lift']:.3f} folds={row['fold_count_after']} "
            f"skipped={int(row.get('candidate_folds_skipped', 0))}"
        )

    df = pd.DataFrame(rows)
    focus = [
        "setup_id", "symbol", "side", "profile", "rows_after", "keep_ratio",
        "positive_rate_before", "positive_rate_after", "positive_lift",
        "positive_count_after", "fold_count_after",
    ]
    print("\n--- SETUP RAW AUDIT ---")
    print(df[[c for c in focus if c in df.columns]].to_string(index=False))

    matrix = df.groupby(["symbol", "side"], dropna=False).agg(
        setup_count=("setup_id", "count"),
        total_candidates=("rows_after", "sum"),
        mean_positive_lift=("positive_lift", "mean"),
        setups_with_folds=("fold_count_after", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
    ).reset_index()
    print("\n--- COVERAGE MATRIX ---")
    print(matrix.to_string(index=False))

    if args.save:
        out_dir = ensure_dir(Path(config.get("setup_inventory", {}).get("output_dir", "data/setup_inventory")))
        df.to_csv(out_dir / "setup_raw_audit.csv", index=False)
        matrix.to_csv(out_dir / "setup_coverage_matrix.csv", index=False)
        print(f"\nsaved setup inventory sanity outputs under: {out_dir}")


if __name__ == "__main__":
    main()
