from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping
from uuid import uuid4


def now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class LiveStateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def init_db(self) -> None:
        with self.connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS processed_bars (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    closed_bar_time_utc TEXT NOT NULL,
                    current_bar_time_utc TEXT NOT NULL,
                    processed_at_utc TEXT NOT NULL,
                    status TEXT NOT NULL,
                    raw_json TEXT,
                    PRIMARY KEY (symbol, timeframe, closed_bar_time_utc)
                );

                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    setup_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    magic INTEGER NOT NULL,
                    signal_bar_time_utc TEXT NOT NULL,
                    decision_bar_time_utc TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    probability REAL,
                    threshold REAL,
                    dry_run INTEGER NOT NULL,
                    raw_json TEXT,
                    UNIQUE(symbol, timeframe, setup_id, side, signal_bar_time_utc)
                );

                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    signal_id TEXT,
                    created_at_utc TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    setup_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    magic INTEGER NOT NULL,
                    volume REAL,
                    status TEXT NOT NULL,
                    raw_json TEXT
                );

                CREATE TABLE IF NOT EXISTS chart_events (
                    event_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    setup_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    magic INTEGER NOT NULL,
                    event_time_utc TEXT NOT NULL,
                    price REAL,
                    file_path TEXT,
                    status TEXT NOT NULL,
                    raw_json TEXT
                );
                """
            )

    def has_processed_bar(self, symbol: str, timeframe: str, closed_bar_time_utc: str) -> bool:
        with self.connect() as con:
            row = con.execute(
                "SELECT 1 FROM processed_bars WHERE symbol=? AND timeframe=? AND closed_bar_time_utc=?",
                (symbol, timeframe, closed_bar_time_utc),
            ).fetchone()
            return row is not None

    def mark_processed_bar(
        self,
        *,
        symbol: str,
        timeframe: str,
        closed_bar_time_utc: str,
        current_bar_time_utc: str,
        status: str,
        raw: Mapping[str, Any] | None = None,
    ) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO processed_bars
                (symbol, timeframe, closed_bar_time_utc, current_bar_time_utc, processed_at_utc, status, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    timeframe,
                    closed_bar_time_utc,
                    current_bar_time_utc,
                    now_utc_iso(),
                    status,
                    json.dumps(raw or {}, ensure_ascii=False, allow_nan=False),
                ),
            )

    def insert_signal(self, payload: Mapping[str, Any]) -> str:
        signal_id = str(payload.get("signal_id") or uuid4())
        with self.connect() as con:
            con.execute(
                """
                INSERT OR IGNORE INTO signals
                (signal_id, created_at_utc, symbol, timeframe, setup_id, side, magic, signal_bar_time_utc,
                 decision_bar_time_utc, action, reason, probability, threshold, dry_run, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal_id,
                    now_utc_iso(),
                    str(payload["symbol"]),
                    str(payload["timeframe"]),
                    str(payload["setup_id"]),
                    str(payload["side"]),
                    int(payload["magic"]),
                    str(payload["signal_bar_time_utc"]),
                    str(payload["decision_bar_time_utc"]),
                    str(payload.get("action", "no_signal")),
                    str(payload.get("reason", "")),
                    payload.get("probability"),
                    payload.get("threshold"),
                    1 if bool(payload.get("dry_run", True)) else 0,
                    json.dumps(dict(payload), ensure_ascii=False, allow_nan=False),
                ),
            )
        return signal_id

    def insert_order(self, payload: Mapping[str, Any]) -> str:
        order_id = str(payload.get("order_id") or uuid4())
        with self.connect() as con:
            con.execute(
                """
                INSERT OR IGNORE INTO orders
                (order_id, signal_id, created_at_utc, symbol, setup_id, side, magic, volume, status, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    payload.get("signal_id"),
                    now_utc_iso(),
                    str(payload["symbol"]),
                    str(payload["setup_id"]),
                    str(payload["side"]),
                    int(payload["magic"]),
                    payload.get("volume"),
                    str(payload.get("status", "dry_run")),
                    json.dumps(dict(payload), ensure_ascii=False, allow_nan=False),
                ),
            )
        return order_id

    def insert_chart_event(self, payload: Mapping[str, Any], file_path: str | Path | None = None) -> str:
        event_id = str(payload.get("event_id") or uuid4())
        with self.connect() as con:
            con.execute(
                """
                INSERT OR IGNORE INTO chart_events
                (event_id, created_at_utc, event_type, symbol, setup_id, side, magic, event_time_utc, price, file_path, status, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    now_utc_iso(),
                    str(payload.get("event_type", "entry")),
                    str(payload["symbol"]),
                    str(payload["setup_id"]),
                    str(payload["side"]),
                    int(payload["magic"]),
                    str(payload["event_time_utc"]),
                    payload.get("price"),
                    str(file_path) if file_path is not None else None,
                    str(payload.get("status", "pending")),
                    json.dumps(dict(payload), ensure_ascii=False, allow_nan=False),
                ),
            )
        return event_id
