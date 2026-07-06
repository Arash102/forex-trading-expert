from __future__ import annotations

import argparse
from pathlib import Path

from debco.live.config import load_json, selected_setup_ids_from_spec
from debco.live.model_registry import LiveModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exported live model artifacts for all selected setups.")
    parser.add_argument("--live-spec", default="data/final_strategy_report/live_execution_spec.json")
    parser.add_argument("--models-dir", default="data/live_models")
    args = parser.parse_args()
    spec = load_json(args.live_spec)
    setup_ids = selected_setup_ids_from_spec(spec)
    registry = LiveModelRegistry(args.models_dir)
    issues: list[str] = []
    for sid in setup_ids:
        if not registry.has_artifact(sid):
            issues.append(f"missing artifact for {sid}")
            continue
        try:
            art = registry.load_artifact(sid)
            if not art.is_valid_for_inference:
                issues.append(f"invalid model/cutoff for {sid}")
            if not art.feature_columns:
                issues.append(f"empty feature column list for {sid}")
        except Exception as exc:
            issues.append(f"{sid}: {type(exc).__name__}: {exc}")
    if issues:
        print("FAILED live model validation:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print(f"OK: {len(setup_ids)} live model artifacts are valid in {Path(args.models_dir)}")


if __name__ == "__main__":
    main()
