from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_SETUP_MAGIC_NUMBERS: dict[str, int] = {
    "EUR_BUY_MOMENTUM_OVERLAP": 130101,
    "EUR_L4_NOT2_BUY": 130102,
    "EUR_AH_ATR2_BUY": 130103,
    "EUR_LONDON_WEAK_SHORT": 130201,
    "EUR_SELL_H1DOWN_CONT": 130202,
    "EUR_SELL_LONDON_BREAKDOWN": 130203,
    "XAU_H1UP_BUY": 130301,
    "XAU_BUY_DXY_TREND": 130302,
    "XAU_BUY_ASIA_HIGH_RECLAIM_DXY": 130303,
    "XAU_SHORT_REVERSAL": 130401,
    "XAU_SELL_LONDON_REJECTION": 130402,
    "XAU_SELL_H1DOWN_CONT": 130403,
}


@dataclass(frozen=True)
class LiveRouterPaths:
    live_config_path: Path
    live_execution_spec_path: Path
    state_db_path: Path
    chart_event_dir: Path


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing JSON file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def dump_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_live_router_config(path: str | Path) -> dict[str, Any]:
    cfg = load_json(path)
    if "setup_magic_numbers" not in cfg:
        cfg["setup_magic_numbers"] = dict(DEFAULT_SETUP_MAGIC_NUMBERS)
    return cfg


def resolve_paths(live_config_path: str | Path, cfg: Mapping[str, Any]) -> LiveRouterPaths:
    return LiveRouterPaths(
        live_config_path=Path(live_config_path),
        live_execution_spec_path=Path(str(cfg.get("live_execution_spec_path", "data/final_strategy_report/live_execution_spec.json"))),
        state_db_path=Path(str(cfg.get("state_db_path", "data/live_state/forward_demo.sqlite"))),
        chart_event_dir=Path(str(cfg.get("chart_event_dir", "data/live_state/chart_events"))),
    )


def selected_setup_ids_from_spec(spec: Mapping[str, Any]) -> list[str]:
    setups = spec.get("selected_setups", []) or []
    out: list[str] = []
    for row in setups:
        sid = str(row.get("setup_id", "")).strip()
        if sid:
            out.append(sid)
    return out


def _is_finite_number(value: Any) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


def validate_magic_numbers(setup_ids: list[str], magic_numbers: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    seen: dict[int, str] = {}
    for sid in setup_ids:
        if sid not in magic_numbers:
            issues.append(f"missing magic number for setup_id={sid}")
            continue
        raw = magic_numbers[sid]
        if not isinstance(raw, int):
            if not (isinstance(raw, str) and raw.isdigit()):
                issues.append(f"magic number for setup_id={sid} must be an integer, got {raw!r}")
                continue
            raw = int(raw)
        magic = int(raw)
        if magic <= 0:
            issues.append(f"magic number for setup_id={sid} must be positive, got {magic}")
        if magic in seen:
            issues.append(f"duplicate magic number {magic} for setup_id={sid} and setup_id={seen[magic]}")
        seen[magic] = sid
    return issues


def validate_live_execution_spec(spec: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if not spec.get("selected_setups"):
        issues.append("live_execution_spec.json has no selected_setups")
    for row in spec.get("selected_setups", []) or []:
        sid = str(row.get("setup_id", "<missing>"))
        if str(row.get("symbol", "")).upper() not in {"EURUSD", "XAUUSD"}:
            issues.append(f"unexpected/missing symbol for {sid}: {row.get('symbol')!r}")
        if str(row.get("side", "")).lower() not in {"long", "short"}:
            issues.append(f"unexpected/missing side for {sid}: {row.get('side')!r}")
        for k in ["threshold", "top_percentile"]:
            v = row.get(k)
            if isinstance(v, float) and not math.isfinite(v):
                issues.append(f"{sid} contains non-standard numeric {k}={v!r}; use null instead")
    risk = spec.get("risk_per_trade")
    if risk is not None and not _is_finite_number(risk):
        issues.append(f"risk_per_trade must be finite, got {risk!r}")
    return issues


def validate_router_bundle(live_cfg: Mapping[str, Any], spec: Mapping[str, Any]) -> list[str]:
    setup_ids = selected_setup_ids_from_spec(spec)
    issues = []
    issues.extend(validate_live_execution_spec(spec))
    issues.extend(validate_magic_numbers(setup_ids, live_cfg.get("setup_magic_numbers", {})))
    configured_symbols = {str(x).upper() for x in live_cfg.get("symbols", [])}
    spec_symbols = {str(r.get("symbol", "")).upper() for r in spec.get("selected_setups", []) or []}
    missing_symbols = spec_symbols.difference(configured_symbols)
    if missing_symbols:
        issues.append(f"live config symbols missing symbols used by spec: {sorted(missing_symbols)}")
    return issues
