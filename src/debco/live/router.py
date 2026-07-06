from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import load_json, load_live_router_config, resolve_paths, validate_router_bundle
from .mt5_data import MT5ConnectionConfig, MT5DataClient, MT5NotAvailableError
from .order_executor import DryRunOrderExecutor
from .scheduler import BarEvent, choose_sleep_seconds, detect_new_bar, next_expected_bar_time
from .signal_engine import LiveSignalEngine
from .state_store import LiveStateStore


class ForwardDemoRouter:
    def __init__(self, live_config_path: str | Path, *, inject_test_signal: str | None = None):
        self.live_config_path = Path(live_config_path)
        self.cfg = load_live_router_config(self.live_config_path)
        self.paths = resolve_paths(self.live_config_path, self.cfg)
        self.spec = load_json(self.paths.live_execution_spec_path)
        issues = validate_router_bundle(self.cfg, self.spec)
        if issues:
            raise ValueError("Live router configuration/spec validation failed:\n- " + "\n- ".join(issues))
        self.state = LiveStateStore(self.paths.state_db_path)
        self.engine = LiveSignalEngine(
            self.spec,
            self.cfg.get("setup_magic_numbers", {}),
            dry_run=bool(self.cfg.get("execution", {}).get("dry_run", True)),
        )
        self.executor = DryRunOrderExecutor(
            state=self.state,
            chart_event_dir=str(self.paths.chart_event_dir),
            chart_config=self.cfg.get("chart_markers", {}),
        )
        self.inject_test_signal = inject_test_signal
        self.last_seen_current_bar: dict[str, datetime] = {}
        self.next_bar_time: dict[str, datetime] = {}
        self.mt5_client: MT5DataClient | None = None

    def connect_mt5(self) -> MT5DataClient:
        mt5_cfg = self.cfg.get("mt5", {}) or {}
        client = MT5DataClient(
            MT5ConnectionConfig(
                terminal_path=mt5_cfg.get("terminal_path"),
                login=mt5_cfg.get("login"),
                password=mt5_cfg.get("password"),
                server=mt5_cfg.get("server"),
                timeout_seconds=int(mt5_cfg.get("initialize_timeout_seconds", 30)),
            )
        )
        client.initialize()
        self.mt5_client = client
        return client

    def process_bar_event(self, event: BarEvent) -> dict[str, Any]:
        if self.state.has_processed_bar(event.symbol, event.timeframe, event.closed_bar_time_utc):
            return {"symbol": event.symbol, "closed_bar_time_utc": event.closed_bar_time_utc, "status": "already_processed"}
        decisions = self.engine.evaluate_closed_bar(
            symbol=event.symbol,
            timeframe=event.timeframe,
            signal_bar_time_utc=event.closed_bar_time_utc,
            decision_bar_time_utc=event.current_bar_time_utc,
            inject_test_signal=self.inject_test_signal,
        )
        created = []
        for decision in decisions:
            result = self.executor.handle_decision(decision)
            if result:
                created.append(result)
        self.state.mark_processed_bar(
            symbol=event.symbol,
            timeframe=event.timeframe,
            closed_bar_time_utc=event.closed_bar_time_utc,
            current_bar_time_utc=event.current_bar_time_utc,
            status="processed",
            raw={"decision_count": len(decisions), "created_order_intents": len(created)},
        )
        return {
            "symbol": event.symbol,
            "closed_bar_time_utc": event.closed_bar_time_utc,
            "current_bar_time_utc": event.current_bar_time_utc,
            "decision_count": len(decisions),
            "created_order_intents": len(created),
            "status": "processed",
        }

    def poll_once(self, client: MT5DataClient) -> list[dict[str, Any]]:
        timeframe = str(self.cfg.get("timeframe", "M15")).upper()
        bars = int((self.cfg.get("mt5", {}) or {}).get("history_bars", 5000))
        out: list[dict[str, Any]] = []
        for symbol in [str(s).upper() for s in self.cfg.get("symbols", [])]:
            rates = client.latest_rates(symbol, timeframe, count=bars)
            event = detect_new_bar(
                symbol=symbol,
                timeframe=timeframe,
                rates=rates,
                last_seen_current_bar_time=self.last_seen_current_bar.get(symbol),
            )
            if event is None:
                out.append({"symbol": symbol, "status": "no_new_bar"})
                continue
            self.last_seen_current_bar[symbol] = event.current_bar_time
            self.next_bar_time[symbol] = next_expected_bar_time(event.current_bar_time, timeframe)
            out.append(self.process_bar_event(event))
        return out

    def sleep_seconds(self) -> float:
        polling = self.cfg.get("polling", {}) or {}
        now = datetime.now(tz=timezone.utc)
        candidates = list(self.next_bar_time.values())
        next_bar = min(candidates) if candidates else None
        return choose_sleep_seconds(
            now=now,
            next_bar_time=next_bar,
            normal_poll_seconds=float(polling.get("normal_poll_seconds", 5.0)),
            fast_poll_seconds=float(polling.get("fast_poll_seconds", 0.5)),
            pre_bar_fast_window_seconds=float(polling.get("pre_bar_fast_window_seconds", 8.0)),
            post_bar_fast_window_seconds=float(polling.get("post_bar_fast_window_seconds", 10.0)),
        )

    def run(self, *, once: bool = False) -> None:  # pragma: no cover - integration on user machine
        client = self.connect_mt5()
        try:
            while True:
                try:
                    results = self.poll_once(client)
                    if bool((self.cfg.get("logging", {}) or {}).get("print_heartbeat", True)):
                        print({"router_id": self.cfg.get("router_id"), "results": results})
                except MT5NotAvailableError:
                    raise
                except Exception as exc:
                    print({"router_id": self.cfg.get("router_id"), "error": repr(exc)})
                if once:
                    break
                time.sleep(self.sleep_seconds())
        finally:
            client.shutdown()
