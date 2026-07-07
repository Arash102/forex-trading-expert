from __future__ import annotations

import json
from pathlib import Path

from debco.live.guards import LiveGuardEngine
from debco.live.reporting import write_daily_report
from debco.live.signal_engine import SignalDecision
from debco.live.state_store import LiveStateStore


def decision() -> SignalDecision:
    return SignalDecision(
        symbol="EURUSD",
        timeframe="M15",
        setup_id="EUR_AH_ATR2_BUY",
        side="long",
        magic=130103,
        signal_bar_time_utc="2026-01-01T00:00:00Z",
        decision_bar_time_utc="2026-01-01T00:15:00Z",
        action="enter",
        reason="unit_test",
        dry_run=False,
        tp_pips=15,
        sl_pips=8,
        horizon_bars=16,
    )


def test_guard_blocks_when_max_open_trades_reached(tmp_path: Path):
    state = LiveStateStore(tmp_path / "state.sqlite")
    state.upsert_position(
        {
            "mt5_position_ticket": "1",
            "symbol": "EURUSD",
            "setup_id": "EUR_AH_ATR2_BUY",
            "side": "long",
            "magic": 130103,
            "status": "open",
        }
    )
    guard = LiveGuardEngine(state, {"max_open_trades": 1})
    result = guard.evaluate_pre_trade(decision())
    assert not result.allowed
    assert result.reason == "max_open_trades_reached"


def test_daily_report_writes_summary_files(tmp_path: Path):
    state = LiveStateStore(tmp_path / "state.sqlite")
    d = decision()
    signal_id = state.insert_signal(d.to_payload())
    state.insert_order({"signal_id": signal_id, "symbol": "EURUSD", "setup_id": "EUR_AH_ATR2_BUY", "side": "long", "magic": 130103, "status": "dry_run_order_intent_created"})
    summary = write_daily_report(state, tmp_path / "reports")
    assert summary["signal_count"] == 1
    assert summary["order_count"] == 1
    summary_path = Path(summary["files"]["summary_json"])
    assert summary_path.exists()
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["order_status_counts"]["dry_run_order_intent_created"] == 1
