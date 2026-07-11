from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from .position_manager import _obj_as_dict, _obj_get
from .state_store import LiveStateStore


def _to_iso_from_any(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        except Exception:
            return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def _history_deals_get(mt5_client: Any, date_from: datetime, date_to: datetime, *, position_ticket: str) -> list[Any]:
    # Prefer client wrapper if one exists; otherwise use raw MetaTrader5 module.
    fn = getattr(mt5_client, "history_deals_get", None)
    if callable(fn):
        try:
            deals = fn(date_from, date_to, position=int(position_ticket))
            return list(deals or [])
        except TypeError:
            deals = fn(date_from, date_to)
            return [d for d in list(deals or []) if str(_obj_get(d, "position_id", _obj_get(d, "position", ""))) == str(position_ticket)]

    mt5 = getattr(mt5_client, "mt5", None)
    fn = getattr(mt5, "history_deals_get", None)
    if callable(fn):
        try:
            deals = fn(date_from, date_to, position=int(position_ticket))
            return list(deals or [])
        except TypeError:
            deals = fn(date_from, date_to)
            return [d for d in list(deals or []) if str(_obj_get(d, "position_id", _obj_get(d, "position", ""))) == str(position_ticket)]
    return []


def _pick_closing_deal(deals: list[Any], mt5: Any = None) -> Any | None:
    if not deals:
        return None
    entry_in = getattr(mt5, "DEAL_ENTRY_IN", 0) if mt5 is not None else 0
    # Closing deals are usually DEAL_ENTRY_OUT or DEAL_ENTRY_INOUT. If constants are
    # unavailable, choose the latest deal with a non-zero profit; otherwise latest deal.
    candidates = [d for d in deals if _obj_get(d, "entry", None) is not None and _obj_get(d, "entry", None) != entry_in]
    if not candidates:
        candidates = [d for d in deals if float(_obj_get(d, "profit", 0.0) or 0.0) != 0.0]
    if not candidates:
        candidates = list(deals)
    return sorted(candidates, key=lambda d: float(_obj_get(d, "time_msc", None) or _obj_get(d, "time", 0) or 0))[-1]


def closing_payload_from_deals(mt5_client: Any, position_ticket: str, *, lookback_days: int = 14) -> dict[str, Any] | None:
    now = datetime.now(tz=timezone.utc)
    deals = _history_deals_get(mt5_client, now - timedelta(days=int(lookback_days)), now + timedelta(days=1), position_ticket=str(position_ticket))
    if not deals:
        return None
    mt5 = getattr(mt5_client, "mt5", None)
    closing = _pick_closing_deal(deals, mt5=mt5)
    if closing is None:
        return None
    raw_deals = [_obj_as_dict(d) for d in deals]
    close_time = _to_iso_from_any(_obj_get(closing, "time_msc", None) or _obj_get(closing, "time", None))
    return {
        "close_time_utc": close_time,
        "close_price": _obj_get(closing, "price", None),
        "profit": _obj_get(closing, "profit", None),
        "history_deal_ticket": _obj_get(closing, "ticket", None),
        "history_deals": raw_deals,
    }


@dataclass(frozen=True)
class HistorySyncResult:
    checked_positions: int = 0
    enriched_positions: int = 0
    missing_history: int = 0

    def to_payload(self) -> dict[str, int]:
        return {
            "checked_positions": self.checked_positions,
            "enriched_positions": self.enriched_positions,
            "missing_history": self.missing_history,
        }


def enrich_closed_positions_from_history(state: LiveStateStore, mt5_client: Any, *, lookback_days: int = 14) -> HistorySyncResult:
    with state.connect() as con:
        rows = [
            dict(r)
            for r in con.execute(
                """
                SELECT * FROM positions
                WHERE mt5_position_ticket IS NOT NULL
                  AND status NOT IN ('open', 'horizon_exit_failed')
                  AND (profit IS NULL OR close_price IS NULL OR close_time_utc IS NULL)
                ORDER BY updated_at_utc DESC
                """
            ).fetchall()
        ]
    checked = enriched = missing = 0
    for row in rows:
        checked += 1
        ticket = str(row.get("mt5_position_ticket") or "")
        payload = closing_payload_from_deals(mt5_client, ticket, lookback_days=lookback_days)
        if not payload:
            missing += 1
            continue
        state.mark_position_closed(
            mt5_position_ticket=ticket,
            close_reason=str(row.get("close_reason") or "history_close_sync"),
            status=str(row.get("status") or "closed"),
            close_time_utc=payload.get("close_time_utc"),
            close_price=payload.get("close_price"),
            profit=payload.get("profit"),
            raw={"history_sync": payload},
        )
        enriched += 1
    return HistorySyncResult(checked_positions=checked, enriched_positions=enriched, missing_history=missing)
