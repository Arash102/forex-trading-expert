from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .state_store import LiveStateStore


def today_utc() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def _snapshot_path(path: Path) -> Path:
    ts = datetime.now(tz=timezone.utc).strftime("%H%M%S")
    return path.with_name(f"{path.stem}_{ts}_{uuid4().hex[:8]}_snapshot{path.suffix}")


def _replace_or_snapshot(tmp_path: Path, final_path: Path) -> Path:
    try:
        tmp_path.replace(final_path)
        return final_path
    except PermissionError:
        snapshot = _snapshot_path(final_path)
        tmp_path.replace(snapshot)
        return snapshot


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    tmp_path = path.with_name(f".{path.stem}.{uuid4().hex}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return _replace_or_snapshot(tmp_path, path)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.{uuid4().hex}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return _replace_or_snapshot(tmp_path, path)


def _query_day(state: LiveStateStore, table: str, day_utc: str) -> list[dict[str, Any]]:
    with state.connect() as con:
        return [
            dict(r)
            for r in con.execute(
                f"SELECT * FROM {table} WHERE substr(created_at_utc,1,10)=? ORDER BY created_at_utc",
                (day_utc,),
            ).fetchall()
        ]


def write_daily_report(state: LiveStateStore, output_dir: str | Path, *, day_utc: str | None = None) -> dict[str, Any]:
    day = day_utc or today_utc()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    signals = _query_day(state, "signals", day)
    orders = _query_day(state, "orders", day)
    positions = _query_day(state, "positions", day)
    guards = _query_day(state, "guard_events", day)
    chart_events = _query_day(state, "chart_events", day)

    planned_paths = {
        "signals_csv": out / f"{day}_signals.csv",
        "orders_csv": out / f"{day}_orders.csv",
        "positions_csv": out / f"{day}_positions.csv",
        "guards_csv": out / f"{day}_guards.csv",
        "chart_events_csv": out / f"{day}_chart_events.csv",
        "summary_json": out / f"{day}_summary.json",
    }

    actual_paths: dict[str, Path] = {}
    actual_paths["signals_csv"] = _write_csv(planned_paths["signals_csv"], signals)
    actual_paths["orders_csv"] = _write_csv(planned_paths["orders_csv"], orders)
    actual_paths["positions_csv"] = _write_csv(planned_paths["positions_csv"], positions)
    actual_paths["guards_csv"] = _write_csv(planned_paths["guards_csv"], guards)
    actual_paths["chart_events_csv"] = _write_csv(planned_paths["chart_events_csv"], chart_events)

    status_counts: dict[str, int] = {}
    for row in orders:
        status = str(row.get("status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "day_utc": day,
        "signal_count": len(signals),
        "order_count": len(orders),
        "position_count": len(positions),
        "guard_event_count": len(guards),
        "chart_event_count": len(chart_events),
        "order_status_counts": status_counts,
        "files": {k: str(v) for k, v in actual_paths.items()},
        "note": "If a CSV was open in Excel, a *_snapshot.csv file may be written instead of overwriting the locked file.",
    }

    actual_paths["summary_json"] = _write_json(planned_paths["summary_json"], summary)
    summary["files"] = {k: str(v) for k, v in actual_paths.items()}
    _write_json(actual_paths["summary_json"], summary)

    return summary
