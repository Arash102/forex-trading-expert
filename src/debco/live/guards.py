from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .signal_engine import SignalDecision
from .state_store import LiveStateStore


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    reason: str
    details: dict[str, Any]

    def to_payload(self, decision: SignalDecision | None = None) -> dict[str, Any]:
        payload = {
            "guard_name": "pre_trade",
            "allowed": self.allowed,
            "reason": self.reason,
            "details": self.details,
        }
        if decision is not None:
            payload.update({
                "symbol": decision.symbol,
                "setup_id": decision.setup_id,
                "side": decision.side,
            })
        return payload


def utc_day(value: str | None = None) -> str:
    if value:
        return str(value)[:10]
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


class LiveGuardEngine:
    """Operational pre-trade guards for one-month demo runs.

    These checks run after model inference and before order creation. They do not
    change model probabilities or setup logic; they only prevent operationally
    unsafe additional orders.
    """

    def __init__(self, state: LiveStateStore, safety_config: Mapping[str, Any] | None = None):
        self.state = state
        self.safety = dict(safety_config or {})

    def evaluate_pre_trade(self, decision: SignalDecision) -> GuardResult:
        if decision.action != "enter":
            return GuardResult(True, "not_entry_decision", {})

        day = utc_day(decision.decision_bar_time_utc)
        max_open_total = int(self.safety.get("max_open_trades", 0) or 0)
        max_open_symbol = int(self.safety.get("max_open_trades_per_symbol", 0) or 0)
        max_open_setup = int(self.safety.get("max_open_trades_per_setup", 1) or 1)
        max_trades_day = int(self.safety.get("max_trades_per_day", 0) or 0)
        max_trades_symbol_day = int(self.safety.get("max_trades_per_symbol_per_day", 0) or 0)
        stop_after_daily_losses = int(self.safety.get("stop_after_daily_losses", 0) or 0)
        block_opposite = bool(self.safety.get("block_opposite_symbol_positions", True))

        open_total = self.state.count_open_positions()
        if max_open_total > 0 and open_total >= max_open_total:
            return GuardResult(False, "max_open_trades_reached", {"open_total": open_total, "limit": max_open_total})

        open_symbol = self.state.count_open_positions(symbol=decision.symbol)
        if max_open_symbol > 0 and open_symbol >= max_open_symbol:
            return GuardResult(False, "max_open_trades_per_symbol_reached", {"open_symbol": open_symbol, "limit": max_open_symbol})

        open_setup = self.state.count_open_positions(symbol=decision.symbol, setup_id=decision.setup_id)
        if max_open_setup > 0 and open_setup >= max_open_setup:
            return GuardResult(False, "max_open_trades_per_setup_reached", {"open_setup": open_setup, "limit": max_open_setup})

        if block_opposite:
            opposite = "short" if str(decision.side).lower() == "long" else "long"
            open_opposite = self.state.count_open_positions(symbol=decision.symbol, side=opposite)
            if open_opposite > 0:
                return GuardResult(False, "opposite_symbol_position_open", {"opposite_side": opposite, "open_opposite": open_opposite})

        trades_today = self.state.count_orders_for_day(day)
        if max_trades_day > 0 and trades_today >= max_trades_day:
            return GuardResult(False, "max_trades_per_day_reached", {"trades_today": trades_today, "limit": max_trades_day})

        trades_symbol_today = self.state.count_orders_for_day(day, symbol=decision.symbol)
        if max_trades_symbol_day > 0 and trades_symbol_today >= max_trades_symbol_day:
            return GuardResult(False, "max_trades_per_symbol_per_day_reached", {"trades_symbol_today": trades_symbol_today, "limit": max_trades_symbol_day})

        losses_today = self.state.count_losing_closed_positions_for_day(day)
        if stop_after_daily_losses > 0 and losses_today >= stop_after_daily_losses:
            return GuardResult(False, "daily_loss_guard_triggered", {"losses_today": losses_today, "limit": stop_after_daily_losses})

        return GuardResult(True, "allowed", {"open_total": open_total, "open_symbol": open_symbol, "trades_today": trades_today})
