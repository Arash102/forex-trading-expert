from __future__ import annotations

import argparse
from pathlib import Path

from debco.live.config import load_json, load_live_router_config, resolve_paths, validate_router_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate live router config + final live execution spec.")
    parser.add_argument("--live-config", default="configs/live_router.example.json")
    args = parser.parse_args()
    live_cfg = load_live_router_config(args.live_config)
    paths = resolve_paths(Path(args.live_config), live_cfg)
    spec = load_json(paths.live_execution_spec_path)
    issues = validate_router_bundle(live_cfg, spec)
    if issues:
        print("FAILED live router validation:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("OK: live router config and live execution spec are valid.")
    print(f"live_execution_spec: {paths.live_execution_spec_path}")
    print(f"state_db_path: {paths.state_db_path}")
    print(f"chart_event_dir: {paths.chart_event_dir}")


if __name__ == "__main__":
    main()
