from __future__ import annotations

from typing import Any, Mapping
from uuid import uuid4

from .chart_events import build_entry_event, write_chart_event_files
from .signal_engine import SignalDecision
from .state_store import LiveStateStore


class DryRunOrderExecutor:
    def __init__(self, *, state: LiveStateStore, chart_event_dir: str, chart_config: Mapping[str, Any] | None = None):
        self.state = state
        self.chart_event_dir = chart_event_dir
        self.chart_config = dict(chart_config or {})

    def handle_decision(self, decision: SignalDecision, *, price: float | None = None) -> dict[str, Any] | None:
        signal_id = self.state.insert_signal(decision.to_payload())
        if decision.action != "enter":
            return None
        order_payload = {
            "order_id": str(uuid4()),
            "signal_id": signal_id,
            "symbol": decision.symbol,
            "setup_id": decision.setup_id,
            "side": decision.side,
            "magic": decision.magic,
            "volume": None,
            "status": "dry_run_order_intent_created",
            "reason": decision.reason,
        }
        order_id = self.state.insert_order(order_payload)
        event = build_entry_event(
            symbol=decision.symbol,
            setup_id=decision.setup_id,
            side=decision.side,
            magic=decision.magic,
            event_time_utc=decision.decision_bar_time_utc,
            price=price,
            screenshot_enabled=bool(self.chart_config.get("screenshot_on_entry", True)),
            screenshot_width=int(self.chart_config.get("screenshot_width", 1280)),
            screenshot_height=int(self.chart_config.get("screenshot_height", 720)),
        )
        files = write_chart_event_files(self.chart_event_dir, event)
        self.state.insert_chart_event(event, file_path=files.get("cmd"))
        return {"signal_id": signal_id, "order_id": order_id, "chart_files": {k: str(v) for k, v in files.items()}}
