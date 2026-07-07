from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate static one-month demo readiness checks.")
    parser.add_argument("--live-config", default="configs/live_router.local.json")
    args = parser.parse_args()
    cfg = _load(args.live_config)
    issues: list[str] = []
    execution = cfg.get("execution", {}) or {}
    inference = cfg.get("inference", {}) or {}
    reports = cfg.get("reports", {}) or {}
    pm = cfg.get("position_manager", {}) or {}

    if not bool(execution.get("demo_only", True)):
        issues.append("execution.demo_only must be true")
    if not bool(execution.get("require_demo_orders_cli_flag", True)):
        issues.append("execution.require_demo_orders_cli_flag must be true")
    if not bool(execution.get("horizon_exit_enabled", True)):
        issues.append("execution.horizon_exit_enabled should be true")
    if not bool(inference.get("enabled", False)):
        issues.append("inference.enabled should be true for model-driven demo")
    if not bool(pm.get("enabled", True)):
        issues.append("position_manager.enabled should be true")
    if not bool(reports.get("enabled", True)):
        issues.append("reports.enabled should be true")

    spec_path = Path(str(cfg.get("live_execution_spec_path", "data/final_strategy_report/live_execution_spec.json")))
    if not spec_path.exists():
        issues.append(f"missing live execution spec: {spec_path}")
    models_dir = Path(str(inference.get("live_models_dir", "data/live_models")))
    if not models_dir.exists():
        issues.append(f"missing live models dir: {models_dir}")
    else:
        artifact_count = len(list(models_dir.glob("*/artifact.json")))
        if artifact_count < 12:
            issues.append(f"expected at least 12 live model artifacts, found {artifact_count}")

    print("DEBCO one-month demo readiness static check")
    if issues:
        print("NOT READY")
        for issue in issues:
            print("-", issue)
        raise SystemExit(1)
    print("OK: static readiness checks passed.")
    print("Next: run a short supervised demo session before leaving the router unattended.")


if __name__ == "__main__":
    main()
