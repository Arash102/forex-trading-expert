from __future__ import annotations

import json
from pathlib import Path

from debco.live.ops_alerts import AlertManager


def test_alert_manager_writes_jsonl_and_throttles(tmp_path: Path):
    path = tmp_path / "alerts.jsonl"
    mgr = AlertManager({"jsonl_path": str(path), "throttle_seconds": 999999})
    first = mgr.send(level="WARNING", event="x", message="hello", throttle_key="same")
    second = mgr.send(level="WARNING", event="x", message="hello", throttle_key="same")
    assert first["throttled"] is False
    assert second["throttled"] is True
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["level"] == "WARNING"
