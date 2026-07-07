from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a safe local live-router config for one-month demo preparation.")
    parser.add_argument("--from-config", default="configs/live_router.example.json")
    parser.add_argument("--to-config", default="configs/live_router.local.json")
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    args = parser.parse_args()

    src = Path(args.from_config)
    dst = Path(args.to_config)
    cfg = json.loads(src.read_text(encoding="utf-8"))
    cfg["router_id"] = str(cfg.get("router_id", "debco_forward_demo")) + "_local"
    cfg.setdefault("execution", {})["risk_per_trade"] = float(args.risk_per_trade)
    cfg["execution"]["dry_run"] = True
    cfg["execution"]["enable_orders"] = False
    cfg["execution"]["demo_only"] = True
    cfg["execution"]["require_demo_orders_cli_flag"] = True
    cfg.setdefault("reports", {})["enabled"] = True
    cfg.setdefault("position_manager", {})["enabled"] = True
    cfg.setdefault("logging", {})["print_heartbeat"] = True
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"OK: wrote safe local config to {dst}")
    print("Default is still dry_run=true. Use --enable-demo-orders at runtime only after demo checks pass.")


if __name__ == "__main__":
    main()
