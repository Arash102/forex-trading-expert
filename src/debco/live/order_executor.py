from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping
from uuid import uuid4

from .chart_events import build_entry_event, write_chart_event_files
from .risk import pip_size_for_symbol, risk_weight_for_decision, normalize_volume
from .signal_engine import SignalDecision
from .state_store import LiveStateStore


def _obj_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    if hasattr(obj, name):
        return getattr(obj, name)
    if hasattr(obj, "_asdict"):
        try:
            return obj._asdict().get(name, default)
        except Exception:
            return default
    return default


def _obj_as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "_asdict"):
        try:
            return dict(obj._asdict())
        except Exception:
            return {"repr": repr(obj)}
    out: dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            out[name] = value
    return out or {"repr": repr(obj)}


def _round_price(price: float, digits: int | None) -> float:
    if digits is None:
        return float(price)
    return round(float(price), int(digits))


@dataclass(frozen=True)
class TradePlan:
    symbol: str
    side: str
    entry_price: float
    sl_price: float
    tp_price: float
    sl_pips: float
    tp_pips: float
    horizon_bars: int | None
    risk_per_trade: float
    risk_weight: float
    effective_risk_per_trade: float
    account_equity: float
    risk_amount: float
    volume: float
    risk_per_lot: float
    pip_size: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "sl_pips": self.sl_pips,
            "tp_pips": self.tp_pips,
            "horizon_bars": self.horizon_bars,
            "risk_per_trade": self.risk_per_trade,
            "risk_weight": self.risk_weight,
            "effective_risk_per_trade": self.effective_risk_per_trade,
            "account_equity": self.account_equity,
            "risk_amount": self.risk_amount,
            "volume": self.volume,
            "risk_per_lot": self.risk_per_lot,
            "pip_size": self.pip_size,
        }


class DemoOrderExecutor:
    """Create dry-run intents or send explicit demo MT5 market orders.

    The default path remains safe: dry_run=true and enable_orders=false.  Actual
    MT5 order_send is only allowed when all three are true:
    - execution.enable_orders=true
    - execution.dry_run=false
    - runtime flag --enable-demo-orders confirmed this run

    If execution.demo_only=true, a non-demo or unverifiable account is blocked.
    """

    def __init__(
        self,
        *,
        state: LiveStateStore,
        chart_event_dir: str,
        live_spec: Mapping[str, Any],
        execution_config: Mapping[str, Any] | None = None,
        chart_config: Mapping[str, Any] | None = None,
        mt5_client: Any | None = None,
    ):
        self.state = state
        self.chart_event_dir = chart_event_dir
        self.live_spec = live_spec
        self.execution_config = dict(execution_config or {})
        self.chart_config = dict(chart_config or {})
        self.mt5_client = mt5_client

    def attach_mt5_client(self, mt5_client: Any) -> None:
        self.mt5_client = mt5_client

    @property
    def dry_run(self) -> bool:
        return bool(self.execution_config.get("dry_run", True))

    @property
    def orders_enabled(self) -> bool:
        return bool(self.execution_config.get("enable_orders", False))

    @property
    def demo_orders_confirmed(self) -> bool:
        return bool(self.execution_config.get("runtime_demo_orders_confirmed", False))

    def _mt5(self) -> Any:
        if self.mt5_client is None:
            return None
        return getattr(self.mt5_client, "mt5", None)

    def _is_demo_account(self) -> tuple[bool, dict[str, Any]]:
        if self.mt5_client is None or not hasattr(self.mt5_client, "account_info"):
            return False, {"error": "mt5_client_has_no_account_info"}
        info = self.mt5_client.account_info()
        payload = _obj_as_dict(info)
        trade_mode = _obj_get(info, "trade_mode")
        mt5 = self._mt5()
        demo_const = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0) if mt5 is not None else 0
        is_demo = trade_mode == demo_const
        payload["demo_const"] = demo_const
        payload["is_demo_account"] = bool(is_demo)
        return bool(is_demo), payload

    def _can_send_demo_order(self) -> tuple[bool, str, dict[str, Any]]:
        if self.dry_run:
            return False, "dry_run_enabled", {}
        if not self.orders_enabled:
            return False, "order_execution_disabled", {}
        if not self.demo_orders_confirmed:
            return False, "demo_order_cli_flag_not_confirmed", {}
        if self.mt5_client is None:
            return False, "mt5_client_missing", {}
        if bool(self.execution_config.get("demo_only", True)):
            ok, account_payload = self._is_demo_account()
            if not ok:
                return False, "demo_only_account_check_failed", account_payload
            return True, "demo_account_confirmed", account_payload
        return True, "demo_only_disabled_by_config", {}

    def _build_trade_plan(self, decision: SignalDecision) -> TradePlan:
        if decision.tp_pips is None or decision.sl_pips is None:
            raise ValueError(f"No executable TP/SL profile for setup_id={decision.setup_id}; job={decision.job!r}")
        if self.mt5_client is None:
            raise ValueError("MT5 client is required to build an executable trade plan")
        symbol_info = self.mt5_client.symbol_info(decision.symbol)
        tick = self.mt5_client.symbol_info_tick(decision.symbol)
        side = str(decision.side).lower()
        entry = float(_obj_get(tick, "ask") if side == "long" else _obj_get(tick, "bid"))
        if not math.isfinite(entry) or entry <= 0:
            raise ValueError(f"Invalid tick entry price for {decision.symbol}: {entry!r}")

        pip_size = pip_size_for_symbol(decision.symbol, config=self.execution_config)
        sl_distance = float(decision.sl_pips) * pip_size
        tp_distance = float(decision.tp_pips) * pip_size
        digits = _obj_get(symbol_info, "digits")
        if side == "long":
            sl_price = entry - sl_distance
            tp_price = entry + tp_distance
        else:
            sl_price = entry + sl_distance
            tp_price = entry - tp_distance
        sl_price = _round_price(sl_price, digits)
        tp_price = _round_price(tp_price, digits)
        entry_price = _round_price(entry, digits)

        account = self.mt5_client.account_info()
        equity = float(_obj_get(account, "equity", _obj_get(account, "balance", 0.0)) or 0.0)
        if equity <= 0 or not math.isfinite(equity):
            raise ValueError(f"Invalid account equity: {equity!r}")
        base_risk = float(self.execution_config.get("risk_per_trade", self.live_spec.get("risk_per_trade", 0.01)))
        risk_weight = risk_weight_for_decision(
            self.live_spec,
            symbol=decision.symbol,
            side=decision.side,
            setup_id=decision.setup_id,
        )
        effective_risk = base_risk * risk_weight
        risk_amount = equity * effective_risk

        tick_size = float(_obj_get(symbol_info, "trade_tick_size", _obj_get(symbol_info, "point", pip_size)) or pip_size)
        tick_value = float(
            _obj_get(symbol_info, "trade_tick_value_loss", None)
            or _obj_get(symbol_info, "trade_tick_value", None)
            or 0.0
        )
        if tick_size <= 0 or tick_value <= 0:
            raise ValueError(
                f"Cannot compute lot size for {decision.symbol}: trade_tick_size={tick_size!r}, trade_tick_value={tick_value!r}"
            )
        risk_per_lot = abs(entry_price - sl_price) / tick_size * tick_value
        if risk_per_lot <= 0 or not math.isfinite(risk_per_lot):
            raise ValueError(f"Invalid risk_per_lot for {decision.symbol}: {risk_per_lot!r}")
        raw_volume = risk_amount / risk_per_lot
        volume = normalize_volume(
            raw_volume,
            volume_min=float(_obj_get(symbol_info, "volume_min", 0.01) or 0.01),
            volume_max=float(_obj_get(symbol_info, "volume_max", 100.0) or 100.0),
            volume_step=float(_obj_get(symbol_info, "volume_step", 0.01) or 0.01),
        )
        if volume <= 0:
            raise ValueError(f"Computed zero/invalid volume for {decision.symbol}: raw={raw_volume!r}")

        return TradePlan(
            symbol=decision.symbol,
            side=decision.side,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_pips=float(decision.sl_pips),
            tp_pips=float(decision.tp_pips),
            horizon_bars=decision.horizon_bars,
            risk_per_trade=base_risk,
            risk_weight=risk_weight,
            effective_risk_per_trade=effective_risk,
            account_equity=equity,
            risk_amount=risk_amount,
            volume=volume,
            risk_per_lot=risk_per_lot,
            pip_size=pip_size,
        )

    def _build_order_request(self, decision: SignalDecision, plan: TradePlan) -> dict[str, Any]:
        mt5 = self._mt5()
        if mt5 is None:
            raise ValueError("MT5 module is not attached")
        side = str(decision.side).lower()
        filling = str(self.execution_config.get("type_filling", "ORDER_FILLING_IOC"))
        req = {
            "action": getattr(mt5, "TRADE_ACTION_DEAL"),
            "symbol": decision.symbol,
            "volume": plan.volume,
            "type": getattr(mt5, "ORDER_TYPE_BUY") if side == "long" else getattr(mt5, "ORDER_TYPE_SELL"),
            "price": plan.entry_price,
            "sl": plan.sl_price,
            "tp": plan.tp_price,
            "deviation": int(self.execution_config.get("deviation_points", 20)),
            "magic": int(decision.magic),
            "comment": f"DEBCO {decision.setup_id[:20]}",
            "type_time": getattr(mt5, "ORDER_TIME_GTC"),
        }
        if hasattr(mt5, filling):
            req["type_filling"] = getattr(mt5, filling)
        return req

    def _send_mt5_order(self, decision: SignalDecision, plan: TradePlan) -> tuple[str, dict[str, Any], dict[str, Any]]:
        request = self._build_order_request(decision, plan)
        result = self.mt5_client.order_send(request)  # type: ignore[union-attr]
        result_payload = _obj_as_dict(result)
        mt5 = self._mt5()
        ok_codes = {
            getattr(mt5, "TRADE_RETCODE_DONE", 10009),
            getattr(mt5, "TRADE_RETCODE_PLACED", 10008),
        }
        retcode = _obj_get(result, "retcode")
        status = "demo_order_sent" if retcode in ok_codes else "mt5_order_rejected"
        return status, request, result_payload

    def handle_decision(self, decision: SignalDecision) -> dict[str, Any] | None:
        signal_id = self.state.insert_signal(decision.to_payload())
        if decision.action != "enter":
            return None

        can_send, can_reason, account_payload = self._can_send_demo_order()
        order_uuid = str(uuid4())
        order_payload: dict[str, Any] = {
            "order_id": order_uuid,
            "signal_id": signal_id,
            "symbol": decision.symbol,
            "setup_id": decision.setup_id,
            "side": decision.side,
            "magic": decision.magic,
            "volume": None,
            "status": "dry_run_order_intent_created",
            "reason": decision.reason,
            "execution_reason": can_reason,
            "account": account_payload,
        }
        price_for_chart: float | None = None

        try:
            if self.mt5_client is not None:
                plan = self._build_trade_plan(decision)
                order_payload.update(plan.to_payload())
                order_payload["volume"] = plan.volume
                price_for_chart = plan.entry_price
            else:
                plan = None  # type: ignore[assignment]
        except Exception as exc:
            if can_send:
                order_payload["status"] = "blocked_trade_plan_error"
            order_payload["trade_plan_error"] = repr(exc)
            plan = None  # type: ignore[assignment]

        if can_send and plan is not None:
            try:
                status, request, result_payload = self._send_mt5_order(decision, plan)
                order_payload["status"] = status
                order_payload["mt5_request"] = request
                order_payload["mt5_result"] = result_payload
                order_payload["ticket"] = result_payload.get("order") or result_payload.get("deal")
            except Exception as exc:
                order_payload["status"] = "mt5_order_send_exception"
                order_payload["mt5_error"] = repr(exc)
        elif can_send and plan is None:
            order_payload.setdefault("status", "blocked_trade_plan_error")
        elif not self.dry_run:
            order_payload["status"] = f"blocked_{can_reason}"

        order_id = self.state.insert_order(order_payload)
        if order_payload.get("status") == "demo_order_sent" and order_payload.get("ticket") is not None:
            self.state.upsert_position(
                {
                    "mt5_position_ticket": str(order_payload.get("ticket")),
                    "order_id": order_id,
                    "signal_id": signal_id,
                    "symbol": decision.symbol,
                    "setup_id": decision.setup_id,
                    "side": decision.side,
                    "magic": decision.magic,
                    "volume": order_payload.get("volume"),
                    "entry_price": order_payload.get("entry_price"),
                    "sl_price": order_payload.get("sl_price"),
                    "tp_price": order_payload.get("tp_price"),
                    "signal_bar_time_utc": decision.signal_bar_time_utc,
                    "decision_bar_time_utc": decision.decision_bar_time_utc,
                    "horizon_bars": decision.horizon_bars,
                    "status": "open",
                    "raw": order_payload,
                }
            )
        event = build_entry_event(
            symbol=decision.symbol,
            setup_id=decision.setup_id,
            side=decision.side,
            magic=decision.magic,
            event_time_utc=decision.decision_bar_time_utc,
            price=price_for_chart,
            screenshot_enabled=bool(self.chart_config.get("screenshot_on_entry", True)),
            screenshot_width=int(self.chart_config.get("screenshot_width", 1280)),
            screenshot_height=int(self.chart_config.get("screenshot_height", 720)),
        )
        files = write_chart_event_files(self.chart_event_dir, event)
        self.state.insert_chart_event(event, file_path=files.get("cmd"))
        return {
            "signal_id": signal_id,
            "order_id": order_id,
            "status": order_payload.get("status"),
            "ticket": order_payload.get("ticket"),
            "volume": order_payload.get("volume"),
            "chart_files": {k: str(v) for k, v in files.items()},
        }


# Backwards-compatible name used by v0.1.13a/b router/tests.
DryRunOrderExecutor = DemoOrderExecutor
