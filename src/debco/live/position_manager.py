from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .chart_events import build_exit_event, write_chart_event_files
from .state_store import LiveStateStore, now_utc_iso


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


def _parse_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(microsecond=0)
        except Exception:
            return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).replace(microsecond=0)
    except Exception:
        return None


def _to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def timeframe_minutes(timeframe: str) -> int:
    tf = str(timeframe).upper()
    if tf.startswith("M"):
        return int(tf[1:])
    if tf.startswith("H"):
        return int(tf[1:]) * 60
    raise ValueError(f"Unsupported timeframe for horizon exit: {timeframe}")


@dataclass(frozen=True)
class PositionManagerResult:
    synced_positions: int = 0
    externally_closed_positions: int = 0
    horizon_exit_attempts: int = 0
    horizon_exit_sent: int = 0
    horizon_exit_failed: int = 0

    def to_payload(self) -> dict[str, int]:
        return {
            "synced_positions": self.synced_positions,
            "externally_closed_positions": self.externally_closed_positions,
            "horizon_exit_attempts": self.horizon_exit_attempts,
            "horizon_exit_sent": self.horizon_exit_sent,
            "horizon_exit_failed": self.horizon_exit_failed,
        }


class LivePositionManager:
    def __init__(
        self,
        *,
        state: LiveStateStore,
        mt5_client: Any,
        setup_magic_numbers: Mapping[str, Any],
        execution_config: Mapping[str, Any] | None = None,
        chart_config: Mapping[str, Any] | None = None,
        chart_event_dir: str = "data/live_state/chart_events",
    ):
        self.state = state
        self.mt5_client = mt5_client
        self.magic_to_setup = {int(v): str(k) for k, v in dict(setup_magic_numbers or {}).items()}
        self.execution_config = dict(execution_config or {})
        self.chart_config = dict(chart_config or {})
        self.chart_event_dir = chart_event_dir

    def _mt5(self) -> Any:
        return getattr(self.mt5_client, "mt5", None)

    def _side_from_position_type(self, pos_type: Any) -> str:
        mt5 = self._mt5()
        buy_const = getattr(mt5, "POSITION_TYPE_BUY", 0) if mt5 is not None else 0
        return "long" if pos_type == buy_const else "short"

    def _position_ticket(self, pos: Any) -> str | None:
        ticket = _obj_get(pos, "ticket", None)
        if ticket is None:
            ticket = _obj_get(pos, "identifier", None)
        return str(ticket) if ticket is not None else None

    def sync_open_positions_from_mt5(self) -> dict[str, int]:
        raw_positions = self.mt5_client.positions_get()
        if raw_positions is None:
            raw_positions = []
        seen_tickets: set[str] = set()
        synced = 0
        for pos in raw_positions:
            magic = _obj_get(pos, "magic", None)
            try:
                magic_int = int(magic)
            except Exception:
                continue
            if magic_int not in self.magic_to_setup:
                continue
            ticket = self._position_ticket(pos)
            if not ticket:
                continue
            seen_tickets.add(ticket)
            setup_id = self.magic_to_setup[magic_int]
            open_time = _parse_utc(_obj_get(pos, "time", None) or _obj_get(pos, "time_msc", None))
            self.state.upsert_position(
                {
                    "mt5_position_ticket": ticket,
                    "symbol": str(_obj_get(pos, "symbol", "")),
                    "setup_id": setup_id,
                    "side": self._side_from_position_type(_obj_get(pos, "type", None)),
                    "magic": magic_int,
                    "volume": _obj_get(pos, "volume", None),
                    "entry_price": _obj_get(pos, "price_open", None),
                    "sl_price": _obj_get(pos, "sl", None),
                    "tp_price": _obj_get(pos, "tp", None),
                    "open_time_utc": _to_iso(open_time),
                    "status": "open",
                    "profit": _obj_get(pos, "profit", None),
                    "raw": _obj_as_dict(pos),
                }
            )
            synced += 1

        externally_closed = 0
        for row in self.state.list_open_positions():
            ticket = row.get("mt5_position_ticket")
            if ticket and str(ticket) not in seen_tickets:
                self.state.mark_position_closed(
                    mt5_position_ticket=str(ticket),
                    close_reason="not_found_in_mt5_positions",
                    status="externally_closed_or_tp_sl",
                    raw={"sync_seen_tickets": sorted(seen_tickets)},
                )
                externally_closed += 1
        return {"synced_positions": synced, "externally_closed_positions": externally_closed}

    def _can_close(self) -> bool:
        return (
            bool(self.execution_config.get("horizon_exit_enabled", True))
            and bool(self.execution_config.get("enable_orders", False))
            and not bool(self.execution_config.get("dry_run", True))
            and bool(self.execution_config.get("runtime_demo_orders_confirmed", False))
        )

    def _close_position_request(self, row: Mapping[str, Any]) -> tuple[dict[str, Any], float | None]:
        mt5 = self._mt5()
        if mt5 is None:
            raise RuntimeError("MT5 module is not attached")
        symbol = str(row["symbol"])
        side = str(row["side"]).lower()
        tick = self.mt5_client.symbol_info_tick(symbol)
        close_price = float(_obj_get(tick, "bid") if side == "long" else _obj_get(tick, "ask"))
        order_type = getattr(mt5, "ORDER_TYPE_SELL") if side == "long" else getattr(mt5, "ORDER_TYPE_BUY")
        filling = str(self.execution_config.get("type_filling", "ORDER_FILLING_IOC"))
        req = {
            "action": getattr(mt5, "TRADE_ACTION_DEAL"),
            "symbol": symbol,
            "volume": float(row.get("volume") or 0.0),
            "type": order_type,
            "position": int(row["mt5_position_ticket"]),
            "price": close_price,
            "deviation": int(self.execution_config.get("deviation_points", 20)),
            "magic": int(row["magic"]),
            "comment": f"DEBCO horizon {str(row['setup_id'])[:18]}",
            "type_time": getattr(mt5, "ORDER_TIME_GTC"),
        }
        if hasattr(mt5, filling):
            req["type_filling"] = getattr(mt5, filling)
        return req, close_price

    def _send_close(self, row: Mapping[str, Any]) -> tuple[bool, dict[str, Any], dict[str, Any], float | None]:
        req, close_price = self._close_position_request(row)
        result = self.mt5_client.order_send(req)
        result_payload = _obj_as_dict(result)
        mt5 = self._mt5()
        ok_codes = {getattr(mt5, "TRADE_RETCODE_DONE", 10009), getattr(mt5, "TRADE_RETCODE_PLACED", 10008)}
        ok = _obj_get(result, "retcode") in ok_codes
        return ok, req, result_payload, close_price

    def manage_horizon_exits(self, *, timeframe: str, current_bar_times: Mapping[str, str]) -> dict[str, int]:
        if not bool(self.execution_config.get("horizon_exit_enabled", True)):
            return {"horizon_exit_attempts": 0, "horizon_exit_sent": 0, "horizon_exit_failed": 0}
        minutes = timeframe_minutes(timeframe)
        attempts = sent = failed = 0
        for row in self.state.list_open_positions():
            horizon = row.get("horizon_bars")
            if horizon is None:
                continue
            try:
                horizon_int = int(horizon)
            except Exception:
                continue
            if horizon_int <= 0:
                continue
            current = _parse_utc(current_bar_times.get(str(row["symbol"])))
            start = _parse_utc(row.get("decision_bar_time_utc") or row.get("signal_bar_time_utc") or row.get("open_time_utc"))
            if current is None or start is None:
                continue
            elapsed = int((current - start).total_seconds() // (minutes * 60))
            if elapsed < horizon_int:
                continue
            attempts += 1
            ticket = str(row["mt5_position_ticket"])
            if not self._can_close():
                self.state.mark_position_closed(
                    mt5_position_ticket=ticket,
                    close_reason="horizon_exit_due_but_execution_disabled",
                    status="horizon_exit_due_dry_run",
                    raw={"elapsed_bars": elapsed, "horizon_bars": horizon_int},
                )
                failed += 1
                continue
            try:
                ok, request, result, close_price = self._send_close(row)
                if ok:
                    self.state.mark_position_closed(
                        mt5_position_ticket=ticket,
                        close_reason="horizon_exit",
                        status="horizon_exit_sent",
                        close_price=close_price,
                        raw={"elapsed_bars": elapsed, "horizon_bars": horizon_int, "mt5_request": request, "mt5_result": result},
                    )
                    event = build_exit_event(
                        symbol=str(row["symbol"]),
                        setup_id=str(row["setup_id"]),
                        side=str(row["side"]),
                        magic=int(row["magic"]),
                        event_time_utc=current_bar_times.get(str(row["symbol"])) or now_utc_iso(),
                        price=close_price,
                        reason="horizon_exit",
                        screenshot_enabled=bool(self.chart_config.get("screenshot_on_exit", True)),
                        screenshot_width=int(self.chart_config.get("screenshot_width", 1280)),
                        screenshot_height=int(self.chart_config.get("screenshot_height", 720)),
                    )
                    files = write_chart_event_files(self.chart_event_dir, event)
                    self.state.insert_chart_event(event, file_path=files.get("cmd"))
                    sent += 1
                else:
                    self.state.mark_position_closed(
                        mt5_position_ticket=ticket,
                        close_reason="horizon_exit_rejected",
                        status="horizon_exit_failed",
                        raw={"elapsed_bars": elapsed, "horizon_bars": horizon_int, "mt5_request": request, "mt5_result": result},
                    )
                    failed += 1
            except Exception as exc:
                self.state.mark_position_closed(
                    mt5_position_ticket=ticket,
                    close_reason="horizon_exit_exception",
                    status="horizon_exit_failed",
                    raw={"elapsed_bars": elapsed, "horizon_bars": horizon_int, "error": repr(exc)},
                )
                failed += 1
        return {"horizon_exit_attempts": attempts, "horizon_exit_sent": sent, "horizon_exit_failed": failed}

    def manage(self, *, timeframe: str, current_bar_times: Mapping[str, str]) -> PositionManagerResult:
        sync = self.sync_open_positions_from_mt5()
        horizon = self.manage_horizon_exits(timeframe=timeframe, current_bar_times=current_bar_times)
        return PositionManagerResult(
            synced_positions=int(sync.get("synced_positions", 0)),
            externally_closed_positions=int(sync.get("externally_closed_positions", 0)),
            horizon_exit_attempts=int(horizon.get("horizon_exit_attempts", 0)),
            horizon_exit_sent=int(horizon.get("horizon_exit_sent", 0)),
            horizon_exit_failed=int(horizon.get("horizon_exit_failed", 0)),
        )
