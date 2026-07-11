from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from debco.live.heartbeat import heartbeat_status
from debco.live.ops_alerts import AlertManager


def _read_ops_config(path: str | Path | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def check_once(*, heartbeat_path: str, stale_after_seconds: float, alert: AlertManager) -> int:
    status = heartbeat_status(heartbeat_path, stale_after_seconds=stale_after_seconds)
    print(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False))
    if not status.get("ok"):
        alert.send(
            level="CRITICAL",
            event="heartbeat_problem",
            message=f"Live router heartbeat problem: {status.get('reason')}",
            details=status,
            throttle_key=f"watchdog:{status.get('reason')}",
        )
        return 2
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch heartbeat.json and alert if router is stale or missing.")
    ap.add_argument("--ops-config", default="configs/live_ops.local.json")
    ap.add_argument("--heartbeat-path", default=None)
    ap.add_argument("--stale-after-seconds", type=float, default=None)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    args = ap.parse_args()

    cfg = _read_ops_config(args.ops_config)
    heartbeat_path = args.heartbeat_path or str(cfg.get("heartbeat_path", "data/live_runtime/heartbeat.json"))
    stale_after = float(args.stale_after_seconds or cfg.get("watchdog_stale_after_seconds", 600))
    alert = AlertManager(cfg.get("alerts", cfg))

    while True:
        rc = check_once(heartbeat_path=heartbeat_path, stale_after_seconds=stale_after, alert=alert)
        if not args.loop:
            raise SystemExit(rc)
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    main()
