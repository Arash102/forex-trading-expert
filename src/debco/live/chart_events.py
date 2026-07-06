from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


def now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_text(value: Any) -> str:
    return str(value).replace(";", "_").replace("\n", " ").strip()


def write_chart_event_files(event_dir: str | Path, event: Mapping[str, Any]) -> dict[str, Path]:
    """Write audit JSON and a simple .cmd file for the MQL chart helper.

    The .cmd format is semicolon separated because the MQL helper deliberately avoids
    strategy logic and JSON parsing. It only draws markers and screenshots.
    """
    out_dir = Path(event_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    event_id = safe_text(event.get("event_id") or uuid4())
    base = f"{now_token()}_{safe_text(event.get('symbol'))}_{safe_text(event.get('setup_id'))}_{event_id}"
    json_path = out_dir / f"{base}.json"
    cmd_path = out_dir / f"{base}.cmd"
    payload = dict(event)
    payload["event_id"] = event_id
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    fields = [
        event_id,
        safe_text(payload.get("event_type", "entry")),
        safe_text(payload.get("symbol", "")),
        safe_text(payload.get("setup_id", "")),
        safe_text(payload.get("side", "")),
        safe_text(payload.get("magic", "")),
        safe_text(payload.get("event_time_utc", "")),
        safe_text(payload.get("price", "")),
        safe_text(payload.get("marker_color", "lime")),
        safe_text(payload.get("label", payload.get("setup_id", ""))),
        safe_text(payload.get("screenshot_name", "")),
    ]
    cmd_path.write_text(";".join(fields), encoding="utf-8")
    return {"json": json_path, "cmd": cmd_path}


def build_entry_event(
    *,
    symbol: str,
    setup_id: str,
    side: str,
    magic: int,
    event_time_utc: str,
    price: float | None = None,
    screenshot_enabled: bool = True,
    screenshot_width: int = 1280,
    screenshot_height: int = 720,
) -> dict[str, Any]:
    label = f"{setup_id}"
    screenshot_name = ""
    if screenshot_enabled:
        screenshot_name = f"debco_{symbol}_{setup_id}_{event_time_utc.replace(':','').replace('-','').replace('Z','')}_entry.png"
    return {
        "event_id": str(uuid4()),
        "event_type": "entry",
        "symbol": symbol,
        "setup_id": setup_id,
        "side": side,
        "magic": int(magic),
        "event_time_utc": event_time_utc,
        "price": price,
        "marker_color": "lime",
        "marker_shape": "circle",
        "label": label,
        "screenshot": bool(screenshot_enabled),
        "screenshot_name": screenshot_name,
        "screenshot_width": int(screenshot_width),
        "screenshot_height": int(screenshot_height),
        "status": "pending",
    }
