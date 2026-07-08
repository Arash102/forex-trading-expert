from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import tempfile

from .config import (
    load_json,
    load_live_router_config,
    resolve_paths,
    selected_setup_ids_from_spec,
    validate_router_bundle,
)
from .model_registry import LiveModelRegistry
from .router import ForwardDemoRouter
from .scheduler import detect_new_bar


@dataclass
class HealthcheckResult:
    ready: bool
    issues: list[str]
    warnings: list[str]
    details: dict[str, Any]


def _check_writable_dir(path: str | Path, name: str, issues: list[str]) -> None:
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
        tmp = p / f".debco_write_test_{next(tempfile._get_candidate_names())}.tmp"
        tmp.write_text("ok", encoding="utf-8")
        tmp.unlink(missing_ok=True)
    except Exception as exc:
        issues.append(f"{name} is not writable: {p} error={type(exc).__name__}:{exc}")


def _is_demo_account(client: Any) -> tuple[bool, str]:
    info = client.account_info()
    trade_mode = getattr(info, "trade_mode", None)
    mt5 = client.mt5
    demo_const = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0) if mt5 is not None else 0
    login = getattr(info, "login", None)
    server = getattr(info, "server", "")
    equity = getattr(info, "equity", None)
    text = f"login={login} server={server} trade_mode={trade_mode} equity={equity}"
    return bool(trade_mode == demo_const), text


def run_startup_healthcheck(
    live_config_path: str | Path,
    *,
    force_inference_enabled: bool | None = None,
    force_demo_orders_enabled: bool | None = None,
    print_report: bool = True,
) -> HealthcheckResult:
    issues: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    live_config_path = Path(live_config_path)

    try:
        cfg = load_live_router_config(live_config_path)
        if force_inference_enabled is not None:
            cfg.setdefault("inference", {})["enabled"] = bool(force_inference_enabled)
        if force_demo_orders_enabled:
            execution = cfg.setdefault("execution", {})
            execution["enable_orders"] = True
            execution["dry_run"] = False
            execution["demo_only"] = True
            execution["runtime_demo_orders_confirmed"] = True

        paths = resolve_paths(live_config_path, cfg)
        spec = load_json(paths.live_execution_spec_path)
        bundle_issues = validate_router_bundle(cfg, spec)
        issues.extend(bundle_issues)

        broker_time = cfg.get("broker_time", {}) or {}
        details["broker_time"] = {
            "server_utc_offset_hours": broker_time.get("server_utc_offset_hours"),
            "timestamps_note": broker_time.get(
                "timestamps_note",
                "MT5 bar timestamps are treated as broker-server time for training/live consistency.",
            ),
        }

        _check_writable_dir(paths.chart_event_dir, "chart_event_dir", issues)
        report_dir = (cfg.get("reports", {}) or {}).get("output_dir", "data/live_reports")
        _check_writable_dir(report_dir, "reports.output_dir", issues)

        inf = cfg.get("inference", {}) or {}
        if bool(inf.get("enabled", False)):
            models_dir = Path(str(inf.get("live_models_dir", "data/live_models")))
            if not models_dir.exists():
                issues.append(f"live_models_dir missing: {models_dir}")
            else:
                registry = LiveModelRegistry(models_dir)
                setup_ids = selected_setup_ids_from_spec(spec)
                missing = []
                invalid = []
                for sid in setup_ids:
                    try:
                        art = registry.load_artifact(sid)
                        if not art.is_valid_for_inference:
                            invalid.append(sid)
                    except Exception:
                        missing.append(sid)
                if missing:
                    issues.append(f"missing live model artifacts: {missing}")
                if invalid:
                    issues.append(f"invalid live model artifacts: {invalid}")
                details["live_model_artifact_count"] = len(list(models_dir.glob("*/artifact.json")))

        router = ForwardDemoRouter(
            live_config_path,
            force_inference_enabled=force_inference_enabled,
            force_demo_orders_enabled=force_demo_orders_enabled,
        )

        client = None
        try:
            client = router.connect_mt5()
            demo_ok, account_text = _is_demo_account(client)
            details["account"] = account_text

            execution = router.cfg.get("execution", {}) or {}
            orders_enabled = bool(execution.get("enable_orders", False)) and not bool(execution.get("dry_run", True))
            if orders_enabled and not demo_ok:
                issues.append("demo orders requested but account is not detected as MT5 demo account")
            elif not demo_ok:
                warnings.append("MT5 account is not detected as demo; dry-run can continue, demo orders will be blocked")

            timeframe = str(router.cfg.get("timeframe", "M15")).upper()
            bars = int((router.cfg.get("mt5", {}) or {}).get("history_bars", 5000))
            symbols = [str(s).upper() for s in router.cfg.get("symbols", [])]
            events = {}
            rates_by_symbol = {}

            for symbol in symbols:
                client.symbol_info(symbol)
                client.symbol_info_tick(symbol)
                rates = client.latest_rates(symbol, timeframe, count=min(bars, 5000))
                rates_by_symbol[symbol] = rates
                if rates is None or len(rates) < 100:
                    issues.append(f"{symbol}: insufficient MT5 rates returned for {timeframe}")
                    continue
                event = detect_new_bar(symbol=symbol, timeframe=timeframe, rates=rates, last_seen_current_bar_time=None)
                if event is None:
                    issues.append(f"{symbol}: could not detect current/closed bar from MT5 rates")
                else:
                    events[symbol] = event

            if router.inference_engine is not None and events:
                try:
                    dxy_frame = router._collect_dxy_frame(client, timeframe, min(bars, 5000))
                except Exception as exc:
                    issues.append(f"DXY collection failed: {type(exc).__name__}:{exc}")
                    dxy_frame = None

                candidate_error_count = 0
                model_feature_invalid_count = 0
                decision_count = 0
                for symbol, event in events.items():
                    snapshot = router._build_feature_snapshot_safe(
                        symbol=symbol,
                        rates=rates_by_symbol[symbol],
                        event=event,
                        dxy_frame=dxy_frame,
                    )
                    if snapshot is None:
                        issues.append(f"{symbol}: live feature snapshot failed")
                        continue

                    decisions = router.engine.evaluate_closed_bar(
                        symbol=symbol,
                        timeframe=timeframe,
                        signal_bar_time_utc=event.closed_bar_time_utc,
                        decision_bar_time_utc=event.current_bar_time_utc,
                        feature_snapshot=snapshot,
                    )
                    decision_count += len(decisions)
                    for d in decisions:
                        reason = str(getattr(d, "reason", ""))
                        if "candidate_filter_error" in reason:
                            candidate_error_count += 1
                        if "model_feature_invalid" in reason:
                            model_feature_invalid_count += 1

                details["startup_decision_count"] = decision_count
                details["startup_candidate_filter_error_count"] = candidate_error_count
                details["startup_model_feature_invalid_count"] = model_feature_invalid_count

                if candidate_error_count:
                    issues.append(f"candidate_filter_error detected during startup: {candidate_error_count}")
                if model_feature_invalid_count:
                    issues.append(f"model_feature_invalid detected for candidate-passed setup(s): {model_feature_invalid_count}")

        finally:
            if client is not None:
                client.shutdown()

    except Exception as exc:
        issues.append(f"startup healthcheck exception: {type(exc).__name__}:{exc}")

    ready = len(issues) == 0
    result = HealthcheckResult(ready=ready, issues=issues, warnings=warnings, details=details)

    if print_report:
        print("DEBCO STARTUP HEALTHCHECK")
        print("STARTUP READY" if ready else "STARTUP NOT READY")
        for w in warnings:
            print("WARNING:", w)
        for i in issues:
            print("ERROR:", i)
        print("DETAILS:", json.dumps(details, ensure_ascii=False, indent=2, default=str))

    return result
