from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from debco.live.heartbeat import write_heartbeat
from debco.live.ops_alerts import AlertManager


def _utc() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _run_gate(cmd: list[str], *, name: str, alert: AlertManager) -> None:
    print(f"\n=== GATE: {name} ===")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout)
    if proc.returncode != 0:
        alert.send(
            level="CRITICAL",
            event="preflight_gate_failed",
            message=f"Preflight gate failed: {name}",
            details={"returncode": proc.returncode, "cmd": cmd, "output_tail": proc.stdout[-4000:]},
            force=True,
        )
        raise SystemExit(proc.returncode)


def _looks_bad(line: str) -> bool:
    lower = line.lower()
    bad_tokens = [
        "runtime_error",
        "traceback",
        "exception",
        "not ready",
        "not locked",
        "mt5.initialize failed",
        "model_feature_invalid",
        "daily_report_error",
        "position_manager_error",
        "order_send_exception",
    ]
    return any(tok in lower for tok in bad_tokens)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one-month demo router with preflight gates, heartbeat and alerts.")
    ap.add_argument("--live-config", default="configs/live_router.local.json")
    ap.add_argument("--launch-lock", default="configs/demo_month_launch_lock.local.json")
    ap.add_argument("--ops-config", default="configs/live_ops.local.json")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--auto-restart", action="store_true")
    ap.add_argument("--max-restarts", type=int, default=0)
    args = ap.parse_args()

    ops = _read_json(args.ops_config)
    heartbeat_path = str(ops.get("heartbeat_path", "data/live_runtime/heartbeat.json"))
    heartbeat_interval = float(ops.get("heartbeat_interval_seconds", 60))
    alert = AlertManager(ops.get("alerts", ops))

    py = sys.executable
    if not args.skip_preflight:
        _run_gate([py, "scripts/17_diagnose_live_features.py", "--live-config", args.live_config], name="feature_diagnostics", alert=alert)
        _run_gate([py, "scripts/12_forward_demo_router.py", "--live-config", args.live_config, "--startup-healthcheck-only", "--enable-inference", "--enable-demo-orders"], name="startup_healthcheck", alert=alert)
        _run_gate([py, "scripts/16_validate_demo_month_launch_lock.py", "--live-config", args.live_config, "--launch-lock", args.launch_lock], name="launch_lock", alert=alert)

    router_cmd = [py, "scripts/12_forward_demo_router.py", "--live-config", args.live_config, "--enable-inference", "--enable-demo-orders"]
    restarts = 0
    alert.send(level="INFO", event="supervisor_starting", message="DEBCO demo-month supervisor is starting", details={"cmd": router_cmd}, force=True)

    while True:
        print("\n=== START ROUTER ===")
        print(" ".join(router_cmd))
        proc = subprocess.Popen(
            router_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        write_heartbeat(heartbeat_path, status="router_started", pid=proc.pid, details={"cmd": router_cmd, "restarts": restarts})
        last_heartbeat = time.time()

        try:
            assert proc.stdout is not None
            while True:
                line = proc.stdout.readline()
                now = time.time()
                if line:
                    print(line, end="")
                    if _looks_bad(line):
                        alert.send(
                            level="WARNING",
                            event="router_log_warning",
                            message=line.strip()[:1000],
                            details={"line": line.strip()},
                            throttle_key="router_log_warning:" + line.strip()[:120],
                        )
                if now - last_heartbeat >= heartbeat_interval:
                    write_heartbeat(
                        heartbeat_path,
                        status="router_running",
                        pid=proc.pid,
                        details={"returncode": proc.poll(), "last_log_line": line.strip() if line else None, "restarts": restarts},
                    )
                    last_heartbeat = now
                rc = proc.poll()
                if rc is not None:
                    break
                if not line:
                    time.sleep(1.0)
        finally:
            rc = proc.poll()
            write_heartbeat(heartbeat_path, status="router_exited", pid=proc.pid, details={"returncode": rc, "restarts": restarts})

        if rc == 0:
            alert.send(level="INFO", event="router_exited", message="Router exited cleanly", details={"returncode": rc}, force=True)
            raise SystemExit(0)

        alert.send(level="CRITICAL", event="router_crashed", message=f"Router exited with returncode={rc}", details={"returncode": rc, "restarts": restarts}, force=True)
        if not args.auto_restart or restarts >= int(args.max_restarts):
            raise SystemExit(int(rc or 3))
        restarts += 1
        time.sleep(float(ops.get("restart_sleep_seconds", 30)))


if __name__ == "__main__":
    main()
