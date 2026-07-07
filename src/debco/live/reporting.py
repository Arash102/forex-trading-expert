from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state_store import LiveStateStore


def today_utc() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _query_day(state: LiveStateStore, table: str, day_utc: str) -> list[dict[str, Any]]:
    with state.connect() as con:
        return [dict(r) for r in con.execute(f"SELECT * FROM {table} WHERE substr(created_at_utc,1,10)=? ORDER BY created_at_utc", (day_utc,)).fetchall()]


def write_daily_report(state: LiveStateStore, output_dir: str | Path, *, day_utc: str | None = None) -> dict[str, Any]:
    day = day_utc or today_utc()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    signals = _query_day(state, "signals", day)
    orders = _query_day(state, "orders", day)
    positions = _query_day(state, "positions", day)
    guards = _query_day(state, "guard_events", day)
    chart_events = _query_day(state, "chart_events", day)

    paths = {
        "signals_csv": out / f"{day}_signals.csv",
        "orders_csv": out / f"{day}_orders.csv",
        "positions_csv": out / f"{day}_positions.csv",
        "guards_csv": out / f"{day}_guards.csv",
        "chart_events_csv": out / f"{day}_chart_events.csv",
        "summary_json": out / f"{day}_summary.json",
    }
    _write_csv(paths["signals_csv"], signals)
    _write_csv(paths["orders_csv"], orders)
    _write_csv(paths["positions_csv"], positions)
    _write_csv(paths["guards_csv"], guards)
    _write_csv(paths["chart_events_csv"], chart_events)

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
        "files": {k: str(v) for k, v in paths.items()},
    }
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary
