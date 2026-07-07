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


def _json(payload: Mapping[str, Any] | None) -> str:
    return json.dumps(dict(payload or {}), ensure_ascii=False, allow_nan=False)


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

                CREATE TABLE IF NOT EXISTS positions (
                    position_id TEXT PRIMARY KEY,
                    mt5_position_ticket TEXT UNIQUE,
                    order_id TEXT,
                    signal_id TEXT,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    setup_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    magic INTEGER NOT NULL,
                    volume REAL,
                    entry_price REAL,
                    sl_price REAL,
                    tp_price REAL,
                    open_time_utc TEXT,
                    signal_bar_time_utc TEXT,
                    decision_bar_time_utc TEXT,
                    horizon_bars INTEGER,
                    status TEXT NOT NULL,
                    close_time_utc TEXT,
                    close_price REAL,
                    close_reason TEXT,
                    profit REAL,
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

                CREATE TABLE IF NOT EXISTS guard_events (
                    guard_event_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    symbol TEXT,
                    setup_id TEXT,
                    side TEXT,
                    guard_name TEXT NOT NULL,
                    allowed INTEGER NOT NULL,
                    reason TEXT NOT NULL,
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
                    _json(raw),
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
                    _json(payload),
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
                    _json(payload),
                ),
            )
        return order_id

    def upsert_position(self, payload: Mapping[str, Any]) -> str:
        position_id = str(payload.get("position_id") or uuid4())
        ticket = payload.get("mt5_position_ticket") or payload.get("ticket")
        if ticket is not None:
            ticket = str(ticket)
        now = now_utc_iso()
        with self.connect() as con:
            existing = None
            if ticket:
                existing = con.execute("SELECT position_id FROM positions WHERE mt5_position_ticket=?", (ticket,)).fetchone()
            if existing is not None:
                position_id = str(existing["position_id"])
                con.execute(
                    """
                    UPDATE positions SET
                        updated_at_utc=?, order_id=COALESCE(?, order_id), signal_id=COALESCE(?, signal_id),
                        symbol=?, setup_id=?, side=?, magic=?, volume=COALESCE(?, volume),
                        entry_price=COALESCE(?, entry_price), sl_price=COALESCE(?, sl_price), tp_price=COALESCE(?, tp_price),
                        open_time_utc=COALESCE(?, open_time_utc), signal_bar_time_utc=COALESCE(?, signal_bar_time_utc),
                        decision_bar_time_utc=COALESCE(?, decision_bar_time_utc), horizon_bars=COALESCE(?, horizon_bars),
                        status=?, profit=COALESCE(?, profit), raw_json=?
                    WHERE position_id=?
                    """,
                    (
                        now,
                        payload.get("order_id"),
                        payload.get("signal_id"),
                        str(payload["symbol"]),
                        str(payload["setup_id"]),
                        str(payload["side"]),
                        int(payload["magic"]),
                        payload.get("volume"),
                        payload.get("entry_price"),
                        payload.get("sl_price"),
                        payload.get("tp_price"),
                        payload.get("open_time_utc"),
                        payload.get("signal_bar_time_utc"),
                        payload.get("decision_bar_time_utc"),
                        payload.get("horizon_bars"),
                        str(payload.get("status", "open")),
                        payload.get("profit"),
                        _json(payload),
                        position_id,
                    ),
                )
            else:
                con.execute(
                    """
                    INSERT INTO positions
                    (position_id, mt5_position_ticket, order_id, signal_id, created_at_utc, updated_at_utc,
                     symbol, setup_id, side, magic, volume, entry_price, sl_price, tp_price, open_time_utc,
                     signal_bar_time_utc, decision_bar_time_utc, horizon_bars, status, close_time_utc,
                     close_price, close_reason, profit, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        position_id,
                        ticket,
                        payload.get("order_id"),
                        payload.get("signal_id"),
                        now,
                        now,
                        str(payload["symbol"]),
                        str(payload["setup_id"]),
                        str(payload["side"]),
                        int(payload["magic"]),
                        payload.get("volume"),
                        payload.get("entry_price"),
                        payload.get("sl_price"),
                        payload.get("tp_price"),
                        payload.get("open_time_utc"),
                        payload.get("signal_bar_time_utc"),
                        payload.get("decision_bar_time_utc"),
                        payload.get("horizon_bars"),
                        str(payload.get("status", "open")),
                        payload.get("close_time_utc"),
                        payload.get("close_price"),
                        payload.get("close_reason"),
                        payload.get("profit"),
                        _json(payload),
                    ),
                )
        return position_id

    def list_open_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM positions WHERE status IN ('open','horizon_exit_failed')"
        args: tuple[Any, ...] = ()
        if symbol:
            sql += " AND symbol=?"
            args = (symbol,)
        with self.connect() as con:
            return [dict(r) for r in con.execute(sql, args).fetchall()]

    def mark_position_closed(
        self,
        *,
        mt5_position_ticket: str,
        close_reason: str,
        status: str = "closed",
        close_time_utc: str | None = None,
        close_price: float | None = None,
        profit: float | None = None,
        raw: Mapping[str, Any] | None = None,
    ) -> None:
        with self.connect() as con:
            con.execute(
                """
                UPDATE positions SET
                    updated_at_utc=?, status=?, close_time_utc=?, close_price=COALESCE(?, close_price),
                    close_reason=?, profit=COALESCE(?, profit), raw_json=COALESCE(?, raw_json)
                WHERE mt5_position_ticket=?
                """,
                (
                    now_utc_iso(),
                    status,
                    close_time_utc or now_utc_iso(),
                    close_price,
                    close_reason,
                    profit,
                    _json(raw) if raw is not None else None,
                    str(mt5_position_ticket),
                ),
            )

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
                    _json(payload),
                ),
            )
        return event_id

    def insert_guard_event(self, payload: Mapping[str, Any]) -> str:
        event_id = str(payload.get("guard_event_id") or uuid4())
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO guard_events
                (guard_event_id, created_at_utc, symbol, setup_id, side, guard_name, allowed, reason, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    now_utc_iso(),
                    payload.get("symbol"),
                    payload.get("setup_id"),
                    payload.get("side"),
                    str(payload.get("guard_name", "pre_trade")),
                    1 if bool(payload.get("allowed", False)) else 0,
                    str(payload.get("reason", "")),
                    _json(payload),
                ),
            )
        return event_id

    def count_open_positions(self, symbol: str | None = None, setup_id: str | None = None, side: str | None = None) -> int:
        clauses = ["status IN ('open','horizon_exit_failed')"]
        args: list[Any] = []
        if symbol:
            clauses.append("symbol=?")
            args.append(symbol)
        if setup_id:
            clauses.append("setup_id=?")
            args.append(setup_id)
        if side:
            clauses.append("side=?")
            args.append(side)
        with self.connect() as con:
            row = con.execute(f"SELECT COUNT(*) AS n FROM positions WHERE {' AND '.join(clauses)}", tuple(args)).fetchone()
            return int(row["n"])

    def count_orders_for_day(self, day_utc: str, symbol: str | None = None) -> int:
        clauses = ["substr(created_at_utc,1,10)=?"]
        args: list[Any] = [day_utc]
        if symbol:
            clauses.append("symbol=?")
            args.append(symbol)
        with self.connect() as con:
            row = con.execute(f"SELECT COUNT(*) AS n FROM orders WHERE {' AND '.join(clauses)}", tuple(args)).fetchone()
            return int(row["n"])

    def count_losing_closed_positions_for_day(self, day_utc: str) -> int:
        with self.connect() as con:
            row = con.execute(
                """
                SELECT COUNT(*) AS n FROM positions
                WHERE close_time_utc IS NOT NULL AND substr(close_time_utc,1,10)=?
                  AND profit IS NOT NULL AND profit < 0
                """,
                (day_utc,),
            ).fetchone()
            return int(row["n"])
