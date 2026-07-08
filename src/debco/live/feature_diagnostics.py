from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .dxy import DXY_COMPONENT_WEIGHTS, rates_to_ohlc_frame
from .router import ForwardDemoRouter
from .scheduler import detect_new_bar


@dataclass(frozen=True)
class FeatureProblem:
    setup_id: str
    symbol: str
    side: str
    candidate_pass: bool
    reason: str
    missing_features: list[str]
    bad_features: list[str]
    likely_source: str
    closed_bar_time_utc: str
    decision_bar_time_utc: str


def _json_default(value: Any) -> str:
    try:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except Exception:
        pass
    return str(value)


def _to_utc_iso(value: Any) -> str:
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return str(value)


def _finite_number(value: Any) -> bool:
    try:
        x = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return not pd.isna(x) and math.isfinite(float(x))
    except Exception:
        return False


def classify_feature_source(feature_name: str) -> str:
    name = str(feature_name)
    if name in {"dxy_close", "dxy_inverse_close", "index_close"} or name.startswith("dxy_"):
        return "DXY"
    if name in {"market_x", "market_y", "market_quadrant", "market_regime"} or name.startswith("market_"):
        return "MARKET/DXY_RELATIVE_FEATURE"
    if name.startswith("h1_"):
        return "H1_RESAMPLE"
    if "session" in name or name.startswith("asia_") or name.startswith("london_"):
        return "SESSION_FEATURE"
    if "zigzag" in name:
        return "ZIGZAG_FEATURE"
    if name.startswith("prev_day") or name.startswith("day_") or "day_" in name:
        return "DAY_FEATURE"
    return "MODEL_FEATURE"


def likely_source_for_bad_features(
    *,
    bad_features: list[str],
    missing_features: list[str],
    dxy_last_time: str | None,
    closed_bar_time: str | None,
    has_dxy_exact_bar: bool | None,
) -> str:
    all_bad = list(missing_features) + list(bad_features)
    groups = {classify_feature_source(c) for c in all_bad}
    if not all_bad:
        return "OK"
    if "DXY" in groups or "MARKET/DXY_RELATIVE_FEATURE" in groups:
        if has_dxy_exact_bar is False:
            if dxy_last_time and closed_bar_time and dxy_last_time < closed_bar_time:
                return "DXY_TIME_LAG: DXY آخرین کندل بسته‌شده نماد را ندارد. کمی بعد دوباره healthcheck بگیر یا componentهای DXY را بررسی کن."
            return "DXY_TIME_ALIGNMENT: زمان DXY با کندل بسته‌شده نماد exact match نشده است."
        return "DXY_OR_MARKET_SOURCE_NAN: DXY وجود دارد ولی یکی از فیچرهای DXY/market مقدار غیرعددی ساخته است."
    return "FEATURE_BUILDER_NAN: ستون‌ها از گروه‌های غیر DXY هستند و باید feature builder همان گروه بررسی شود."


def _artifact_feature_status(feature_row: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str], dict[str, Any]]:
    missing: list[str] = []
    bad: list[str] = []
    values: dict[str, Any] = {}
    if feature_row is None or feature_row.empty:
        return list(feature_columns), [], {}
    latest = feature_row.iloc[-1]
    for col in feature_columns:
        if col not in feature_row.columns:
            missing.append(col)
            continue
        value = latest[col]
        values[col] = None if pd.isna(value) else value
        if not _finite_number(value):
            bad.append(col)
    return missing, bad, values


def _frame_summary(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"rows": 0, "first_time_utc": None, "last_time_utc": None}
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce") if "date" in frame.columns else pd.Series([], dtype="datetime64[ns, UTC]")
    dates = dates.dropna()
    return {
        "rows": int(len(frame)),
        "first_time_utc": _to_utc_iso(dates.iloc[0]) if len(dates) else None,
        "last_time_utc": _to_utc_iso(dates.iloc[-1]) if len(dates) else None,
    }


def diagnose_live_feature_state(
    live_config_path: str | Path,
    *,
    force_inference_enabled: bool = True,
    force_demo_orders_enabled: bool = True,
    max_bars: int = 5000,
) -> dict[str, Any]:
    router = ForwardDemoRouter(
        live_config_path,
        force_inference_enabled=force_inference_enabled,
        force_demo_orders_enabled=force_demo_orders_enabled,
    )
    client = router.connect_mt5()
    try:
        timeframe = str(router.cfg.get("timeframe", "M15")).upper()
        bars = int((router.cfg.get("mt5", {}) or {}).get("history_bars", 5000))
        count = min(int(max_bars), bars, 5000)
        symbols = [str(s).upper() for s in router.cfg.get("symbols", [])]

        report: dict[str, Any] = {
            "live_config_path": str(live_config_path),
            "timeframe": timeframe,
            "history_bars_requested": count,
            "symbols": symbols,
            "account": None,
            "dxy": {},
            "symbols_detail": {},
            "problems": [],
            "summary": {"problem_count": 0, "candidate_pass_problem_count": 0},
        }

        try:
            info = client.account_info()
            report["account"] = {
                "login": getattr(info, "login", None),
                "server": getattr(info, "server", None),
                "trade_mode": getattr(info, "trade_mode", None),
                "equity": getattr(info, "equity", None),
            }
        except Exception as exc:
            report["account"] = {"error": f"{type(exc).__name__}:{exc}"}

        component_detail: dict[str, Any] = {}
        dxy_cfg = router.cfg.get("dxy", {}) or {}
        symbol_map = dxy_cfg.get("component_symbol_map", {}) or {}
        for dxy_symbol in DXY_COMPONENT_WEIGHTS:
            broker_symbol = str(symbol_map.get(dxy_symbol, dxy_symbol))
            try:
                rates = client.latest_rates(broker_symbol, timeframe, count=count)
                frame = rates_to_ohlc_frame(rates, symbol=dxy_symbol)
                component_detail[dxy_symbol] = {
                    "broker_symbol": broker_symbol,
                    **_frame_summary(frame),
                    "last_close": None if frame.empty else float(pd.to_numeric(frame["close"], errors="coerce").iloc[-1]),
                }
            except Exception as exc:
                component_detail[dxy_symbol] = {
                    "broker_symbol": broker_symbol,
                    "error": f"{type(exc).__name__}:{exc}",
                }

        try:
            dxy_frame = router._collect_dxy_frame(client, timeframe, count)
        except Exception as exc:
            dxy_frame = None
            report["dxy"]["collection_error"] = f"{type(exc).__name__}:{exc}"

        dxy_summary = _frame_summary(dxy_frame)
        dxy_dates: set[str] = set()
        if dxy_frame is not None and not dxy_frame.empty and "date" in dxy_frame.columns:
            dxy_dates = {_to_utc_iso(x) for x in pd.to_datetime(dxy_frame["date"], utc=True, errors="coerce").dropna()}
        report["dxy"].update({"summary": dxy_summary, "components": component_detail})

        for symbol in symbols:
            symbol_payload: dict[str, Any] = {}
            report["symbols_detail"][symbol] = symbol_payload
            try:
                rates = client.latest_rates(symbol, timeframe, count=count)
                frame = rates_to_ohlc_frame(rates, symbol=symbol)
                symbol_payload["rates"] = _frame_summary(frame)
                event = detect_new_bar(
                    symbol=symbol,
                    timeframe=timeframe,
                    rates=rates,
                    last_seen_current_bar_time=None,
                )
                if event is None:
                    symbol_payload["error"] = "could_not_detect_new_bar"
                    continue
                closed_time = _to_utc_iso(event.closed_bar_time_utc)
                current_time = _to_utc_iso(event.current_bar_time_utc)
                symbol_payload["closed_bar_time_utc"] = closed_time
                symbol_payload["current_bar_time_utc"] = current_time
                symbol_payload["dxy_has_exact_closed_bar"] = closed_time in dxy_dates

                snapshot = router._build_feature_snapshot_safe(
                    symbol=symbol,
                    rates=rates,
                    event=event,
                    dxy_frame=dxy_frame,
                )
                if snapshot is None:
                    symbol_payload["snapshot"] = "failed"
                    report["problems"].append(
                        {
                            "symbol": symbol,
                            "setup_id": "*",
                            "candidate_pass": None,
                            "missing_features": [],
                            "bad_features": [],
                            "likely_source": "LIVE_FEATURE_SNAPSHOT_FAILED",
                            "closed_bar_time_utc": closed_time,
                            "decision_bar_time_utc": current_time,
                        }
                    )
                    continue

                symbol_payload["snapshot"] = {
                    "signal_bar_time_utc": snapshot.signal_bar_time_utc,
                    "full_frame_tail_rows": snapshot.full_frame_tail_rows,
                    "missing_feature_columns_from_required_union": snapshot.missing_feature_columns,
                }

                setup_rows: list[dict[str, Any]] = []
                for setup in router.engine.setups_for_symbol(symbol):
                    setup_id = str(setup.setup_id)
                    row_payload: dict[str, Any] = {
                        "setup_id": setup_id,
                        "side": str(setup.side),
                        "candidate_pass": None,
                        "candidate_reason": None,
                        "artifact": "missing",
                        "missing_features": [],
                        "bad_features": [],
                        "bad_feature_groups": {},
                        "likely_source": "not_checked",
                    }
                    setup_rows.append(row_payload)

                    if router.inference_engine is None:
                        row_payload["artifact"] = "inference_disabled"
                        continue
                    if not router.inference_engine.models.has_artifact(setup_id):
                        continue
                    try:
                        artifact = router.inference_engine.models.load_artifact(setup_id)
                        row_payload["artifact"] = "loaded"
                        row_payload["artifact_feature_count"] = len(artifact.feature_columns)
                    except Exception as exc:
                        row_payload["artifact"] = f"artifact_error:{type(exc).__name__}:{exc}"
                        continue

                    candidate_row = snapshot.candidate_feature_row if snapshot.candidate_feature_row is not None else snapshot.model_feature_row
                    try:
                        candidate_ok, candidate_reason = router.inference_engine._candidate_pass(setup, candidate_row)
                    except Exception as exc:
                        candidate_ok, candidate_reason = False, f"candidate_check_exception:{type(exc).__name__}:{exc}"
                    row_payload["candidate_pass"] = bool(candidate_ok)
                    row_payload["candidate_reason"] = str(candidate_reason)

                    missing, bad, _values = _artifact_feature_status(snapshot.model_feature_row, artifact.feature_columns)
                    groups: dict[str, list[str]] = {}
                    for col in missing + bad:
                        groups.setdefault(classify_feature_source(col), []).append(col)
                    likely = likely_source_for_bad_features(
                        bad_features=bad,
                        missing_features=missing,
                        dxy_last_time=dxy_summary.get("last_time_utc"),
                        closed_bar_time=closed_time,
                        has_dxy_exact_bar=(closed_time in dxy_dates),
                    )
                    row_payload["missing_features"] = missing
                    row_payload["bad_features"] = bad
                    row_payload["bad_feature_groups"] = groups
                    row_payload["likely_source"] = likely

                    if missing or bad:
                        problem = {
                            "symbol": symbol,
                            "setup_id": setup_id,
                            "side": str(setup.side),
                            "candidate_pass": bool(candidate_ok),
                            "candidate_reason": str(candidate_reason),
                            "missing_features": missing,
                            "bad_features": bad,
                            "bad_feature_groups": groups,
                            "likely_source": likely,
                            "closed_bar_time_utc": closed_time,
                            "decision_bar_time_utc": current_time,
                        }
                        report["problems"].append(problem)

                symbol_payload["setups"] = setup_rows
            except Exception as exc:
                symbol_payload["error"] = f"{type(exc).__name__}:{exc}"

        report["summary"]["problem_count"] = len(report["problems"])
        report["summary"]["candidate_pass_problem_count"] = sum(
            1 for p in report["problems"] if p.get("candidate_pass") is True
        )
        return report
    finally:
        client.shutdown()


def format_feature_diagnostics(report: Mapping[str, Any], *, only_candidate_pass: bool = True) -> str:
    lines: list[str] = []
    lines.append("DEBCO LIVE FEATURE DIAGNOSTICS")
    lines.append("=" * 36)
    lines.append(f"config: {report.get('live_config_path')}")
    lines.append(f"timeframe: {report.get('timeframe')} | bars: {report.get('history_bars_requested')}")
    account = report.get("account") or {}
    if isinstance(account, Mapping):
        lines.append(
            "account: "
            f"login={account.get('login')} server={account.get('server')} "
            f"trade_mode={account.get('trade_mode')} equity={account.get('equity')}"
        )

    dxy = report.get("dxy") or {}
    dxy_summary = dxy.get("summary") or {}
    lines.append("")
    lines.append("DXY")
    lines.append("---")
    lines.append(
        f"rows={dxy_summary.get('rows')} first={dxy_summary.get('first_time_utc')} last={dxy_summary.get('last_time_utc')}"
    )
    components = dxy.get("components") or {}
    if components:
        lines.append("components last times:")
        for name, info in components.items():
            if not isinstance(info, Mapping):
                continue
            if "error" in info:
                lines.append(f"- {name}: ERROR {info.get('error')}")
            else:
                lines.append(
                    f"- {name}({info.get('broker_symbol')}): rows={info.get('rows')} last={info.get('last_time_utc')} close={info.get('last_close')}"
                )

    lines.append("")
    lines.append("SYMBOLS")
    lines.append("-------")
    for symbol, payload in (report.get("symbols_detail") or {}).items():
        if not isinstance(payload, Mapping):
            continue
        lines.append(f"{symbol}:")
        lines.append(
            f"  closed_bar={payload.get('closed_bar_time_utc')} current_bar={payload.get('current_bar_time_utc')} dxy_exact={payload.get('dxy_has_exact_closed_bar')}"
        )
        if payload.get("error"):
            lines.append(f"  ERROR: {payload.get('error')}")

    problems = list(report.get("problems") or [])
    if only_candidate_pass:
        problems = [p for p in problems if isinstance(p, Mapping) and p.get("candidate_pass") is True]

    lines.append("")
    lines.append("PROBLEMS")
    lines.append("--------")
    if not problems:
        if only_candidate_pass:
            lines.append("No invalid features among candidate-passed setups.")
        else:
            lines.append("No invalid features detected.")
    for p in problems:
        if not isinstance(p, Mapping):
            continue
        missing = p.get("missing_features") or []
        bad = p.get("bad_features") or []
        lines.append(
            f"- {p.get('symbol')} {p.get('setup_id')} side={p.get('side')} candidate_pass={p.get('candidate_pass')}"
        )
        lines.append(f"  closed_bar={p.get('closed_bar_time_utc')} decision_bar={p.get('decision_bar_time_utc')}")
        if p.get("candidate_reason"):
            lines.append(f"  candidate_reason={p.get('candidate_reason')}")
        lines.append(f"  missing_count={len(missing)} bad_count={len(bad)}")
        if missing:
            lines.append(f"  missing={', '.join(str(x) for x in missing[:30])}")
        if bad:
            lines.append(f"  bad={', '.join(str(x) for x in bad[:30])}")
        groups = p.get("bad_feature_groups") or {}
        if groups:
            group_text = "; ".join(f"{k}: {', '.join(v[:12])}" for k, v in groups.items())
            lines.append(f"  groups={group_text}")
        lines.append(f"  likely_source={p.get('likely_source')}")

    lines.append("")
    lines.append("DECISION")
    lines.append("--------")
    candidate_problem_count = int((report.get("summary") or {}).get("candidate_pass_problem_count", 0) or 0)
    if candidate_problem_count:
        lines.append("NOT READY: at least one candidate-passed setup has invalid model features. Do not start demo orders yet.")
    else:
        lines.append("FEATURE GATE OK: no candidate-passed setup has invalid model features in this snapshot.")
    return "\n".join(lines)


def dump_feature_diagnostics_json(report: Mapping[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
