from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from debco.live.heartbeat import atomic_write_json, heartbeat_status, read_heartbeat, write_heartbeat


def test_write_and_read_heartbeat(tmp_path: Path):
    path = tmp_path / "heartbeat.json"
    payload = write_heartbeat(path, status="router_running", router_id="x", pid=123, details={"a": 1})
    loaded = read_heartbeat(path)
    assert loaded is not None
    assert loaded["status"] == "router_running"
    assert loaded["router_id"] == "x"
    assert loaded["pid"] == 123
    assert payload["timestamp_utc"].endswith("Z")
    assert heartbeat_status(path, stale_after_seconds=3600)["ok"] is True


def test_heartbeat_stale(tmp_path: Path):
    path = tmp_path / "heartbeat.json"
    old = datetime.now(tz=timezone.utc) - timedelta(hours=2)
    atomic_write_json(path, {"timestamp_utc": old.isoformat().replace("+00:00", "Z"), "status": "router_running"})
    status = heartbeat_status(path, stale_after_seconds=10)
    assert status["ok"] is False
    assert status["reason"] == "heartbeat_stale"
