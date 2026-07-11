from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.stem}.{uuid4().hex}.tmp")
    tmp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(p)
    return p


def write_heartbeat(
    path: str | Path,
    *,
    status: str,
    router_id: str | None = None,
    pid: int | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "timestamp_utc": utc_now_iso(),
        "status": str(status),
        "router_id": router_id,
        "pid": int(pid if pid is not None else os.getpid()),
        "host": socket.gethostname(),
        "details": dict(details or {}),
    }
    atomic_write_json(path, payload)
    return payload


def read_heartbeat(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def heartbeat_age_seconds(path: str | Path, *, now: datetime | None = None) -> float | None:
    hb = read_heartbeat(path)
    if not hb:
        return None
    ts = parse_utc(hb.get("timestamp_utc"))
    if ts is None:
        return None
    current = now or datetime.now(tz=timezone.utc)
    return max(0.0, (current - ts).total_seconds())


def heartbeat_status(path: str | Path, *, stale_after_seconds: float = 600.0) -> dict[str, Any]:
    hb = read_heartbeat(path)
    if hb is None:
        return {"ok": False, "reason": "heartbeat_missing", "age_seconds": None, "heartbeat": None}
    age = heartbeat_age_seconds(path)
    if age is None:
        return {"ok": False, "reason": "heartbeat_unreadable_timestamp", "age_seconds": None, "heartbeat": hb}
    if age > float(stale_after_seconds):
        return {"ok": False, "reason": "heartbeat_stale", "age_seconds": age, "heartbeat": hb}
    return {"ok": True, "reason": "heartbeat_ok", "age_seconds": age, "heartbeat": hb}
