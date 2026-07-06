from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from debco.live.chart_events import write_chart_event_files
from debco.live.config import DEFAULT_SETUP_MAGIC_NUMBERS, validate_magic_numbers, validate_router_bundle
from debco.live.scheduler import choose_sleep_seconds, detect_new_bar
from debco.live.signal_engine import LiveSignalEngine
from debco.live.state_store import LiveStateStore


def minimal_spec():
    rows = []
    for sid, magic in DEFAULT_SETUP_MAGIC_NUMBERS.items():
        symbol = "EURUSD" if sid.startswith("EUR") else "XAUUSD"
        side = "short" if "SELL" in sid or "SHORT" in sid else "long"
        rows.append({"setup_id": sid, "symbol": symbol, "side": side, "policy": "top_percentile", "threshold": None, "top_percentile": 2.0})
    return {"selected_setups": rows, "risk_per_trade": 0.01}


def test_magic_numbers_cover_core_12_and_are_unique():
    setup_ids = list(DEFAULT_SETUP_MAGIC_NUMBERS)
    assert len(setup_ids) == 12
    assert validate_magic_numbers(setup_ids, DEFAULT_SETUP_MAGIC_NUMBERS) == []
    bad = dict(DEFAULT_SETUP_MAGIC_NUMBERS)
    bad[setup_ids[1]] = bad[setup_ids[0]]
    assert any("duplicate" in x for x in validate_magic_numbers(setup_ids, bad))


def test_router_bundle_validation_accepts_minimal_spec():
    live_cfg = {"setup_magic_numbers": DEFAULT_SETUP_MAGIC_NUMBERS, "symbols": ["EURUSD", "XAUUSD"]}
    assert validate_router_bundle(live_cfg, minimal_spec()) == []


def test_detect_new_bar_uses_last_two_chronological_times():
    base = datetime(2026, 7, 6, 13, 45, tzinfo=timezone.utc)
    rates = [
        {"time": int((base - timedelta(minutes=15)).timestamp())},
        {"time": int(base.timestamp())},
        {"time": int((base + timedelta(minutes=15)).timestamp())},
    ]
    event = detect_new_bar(symbol="EURUSD", timeframe="M15", rates=rates, last_seen_current_bar_time=base)
    assert event is not None
    assert event.current_bar_time == base + timedelta(minutes=15)
    assert event.closed_bar_time == base


def test_detect_new_bar_accepts_mt5_numpy_structured_rows():
    np = pytest.importorskip("numpy")
    base = datetime(2026, 7, 6, 13, 45, tzinfo=timezone.utc)
    dtype = [
        ("time", "<i8"),
        ("open", "<f8"),
        ("high", "<f8"),
        ("low", "<f8"),
        ("close", "<f8"),
        ("tick_volume", "<u8"),
        ("spread", "<i4"),
        ("real_volume", "<u8"),
    ]
    rates = np.array(
        [
            (int((base - timedelta(minutes=15)).timestamp()), 1.0, 1.0, 1.0, 1.0, 100, 1, 0),
            (int(base.timestamp()), 1.0, 1.0, 1.0, 1.0, 100, 1, 0),
            (int((base + timedelta(minutes=15)).timestamp()), 1.0, 1.0, 1.0, 1.0, 100, 1, 0),
        ],
        dtype=dtype,
    )
    event = detect_new_bar(symbol="EURUSD", timeframe="M15", rates=rates, last_seen_current_bar_time=base)
    assert event is not None
    assert event.current_bar_time == base + timedelta(minutes=15)
    assert event.closed_bar_time == base


def test_fast_poll_window_selects_fast_sleep():
    next_bar = datetime(2026, 7, 6, 14, 0, tzinfo=timezone.utc)
    now = next_bar - timedelta(seconds=3)
    sleep = choose_sleep_seconds(
        now=now,
        next_bar_time=next_bar,
        normal_poll_seconds=5,
        fast_poll_seconds=0.5,
        pre_bar_fast_window_seconds=8,
        post_bar_fast_window_seconds=10,
    )
    assert sleep == 0.5


def test_state_store_idempotent_processed_bar(tmp_path: Path):
    db = tmp_path / "state.sqlite"
    state = LiveStateStore(db)
    assert not state.has_processed_bar("EURUSD", "M15", "2026-07-06T13:45:00Z")
    state.mark_processed_bar(
        symbol="EURUSD",
        timeframe="M15",
        closed_bar_time_utc="2026-07-06T13:45:00Z",
        current_bar_time_utc="2026-07-06T14:00:00Z",
        status="processed",
    )
    assert state.has_processed_bar("EURUSD", "M15", "2026-07-06T13:45:00Z")
    with sqlite3.connect(db) as con:
        count = con.execute("SELECT COUNT(*) FROM processed_bars").fetchone()[0]
    assert count == 1


def test_chart_event_files_are_written_for_mql_helper(tmp_path: Path):
    files = write_chart_event_files(
        tmp_path,
        {
            "event_id": "evt1",
            "event_type": "entry",
            "symbol": "EURUSD",
            "setup_id": "EUR_AH_ATR2_BUY",
            "side": "long",
            "magic": 130103,
            "event_time_utc": "2026-07-06T14:00:00Z",
            "price": 1.2345,
            "marker_color": "lime",
            "label": "EUR_AH_ATR2_BUY",
            "screenshot_name": "test.png",
        },
    )
    assert files["json"].exists()
    assert files["cmd"].exists()
    assert "EUR_AH_ATR2_BUY" in files["cmd"].read_text(encoding="utf-8")


def test_signal_engine_injected_test_signal():
    engine = LiveSignalEngine(minimal_spec(), DEFAULT_SETUP_MAGIC_NUMBERS, dry_run=True)
    decisions = engine.evaluate_closed_bar(
        symbol="EURUSD",
        timeframe="M15",
        signal_bar_time_utc="2026-07-06T13:45:00Z",
        decision_bar_time_utc="2026-07-06T14:00:00Z",
        inject_test_signal="EUR_AH_ATR2_BUY",
    )
    injected = [d for d in decisions if d.setup_id == "EUR_AH_ATR2_BUY"][0]
    assert injected.action == "enter"
    assert injected.magic == 130103
