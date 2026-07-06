from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from debco.utils.io import read_json

from .config import load_json, load_live_router_config, resolve_paths, validate_router_bundle
from .dxy import build_dxy_from_component_closes, rates_to_ohlc_frame
from .inference import LiveInferenceEngine
from .live_features import LiveFeatureSnapshot, build_live_feature_snapshot
from .mt5_data import MT5ConnectionConfig, MT5DataClient, MT5NotAvailableError
from .order_executor import DryRunOrderExecutor
from .scheduler import BarEvent, choose_sleep_seconds, detect_new_bar, next_expected_bar_time
from .signal_engine import LiveSignalEngine
from .state_store import LiveStateStore


class ForwardDemoRouter:
    def __init__(self, live_config_path: str | Path, *, inject_test_signal: str | None = None, force_inference_enabled: bool | None = None):
        self.live_config_path = Path(live_config_path)
        self.cfg = load_live_router_config(self.live_config_path)
        if force_inference_enabled is not None:
            self.cfg.setdefault("inference", {})["enabled"] = bool(force_inference_enabled)
        self.paths = resolve_paths(self.live_config_path, self.cfg)
        self.spec = load_json(self.paths.live_execution_spec_path)
        issues = validate_router_bundle(self.cfg, self.spec)
        if issues:
            raise ValueError("Live router configuration/spec validation failed:\n- " + "\n- ".join(issues))
        self.state = LiveStateStore(self.paths.state_db_path)
        self.inference_engine = self._build_inference_engine()
        self.engine = LiveSignalEngine(
            self.spec,
            self.cfg.get("setup_magic_numbers", {}),
            dry_run=bool(self.cfg.get("execution", {}).get("dry_run", True)),
            inference_engine=self.inference_engine,
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
        self.feature_config = self._load_feature_config()

    def _build_inference_engine(self) -> LiveInferenceEngine | None:
        inf = self.cfg.get("inference", {}) or {}
        if not bool(inf.get("enabled", False)):
            return None
        return LiveInferenceEngine(
            ml_config_path=inf.get("ml_config_path", "configs/ml_config.local.json"),
            models_dir=inf.get("live_models_dir", "data/live_models"),
            enabled=True,
        )

    def _load_feature_config(self) -> dict[str, Any] | None:
        inf = self.cfg.get("inference", {}) or {}
        if not bool(inf.get("enabled", False)):
            return None

        configured = Path(str(inf.get("feature_config_path", "configs/features_config.example.json")))
        candidates: list[Path] = []

        def add_candidate(path: Path) -> None:
            if path not in candidates:
                candidates.append(path)

        add_candidate(configured)
        # Common project layouts seen during the research pipeline.  Some local
        # workspaces have a list-like placeholder at configs/features_config.example.json;
        # live inference needs the full feature-engineering config object instead.
        if configured.parent:
            add_candidate(configured.parent / "features_config.local.json")
            add_candidate(configured.parent / "feature_config.local.json")
            add_candidate(configured.parent / "features_config.example.json")
            add_candidate(configured.parent / "feature_config.example.json")
        add_candidate(Path("configs/features_config.local.json"))
        add_candidate(Path("configs/feature_config.local.json"))
        add_candidate(Path("features_config.local.json"))
        add_candidate(Path("feature_config.local.json"))

        found_invalid: list[str] = []
        for path in candidates:
            if not path.exists():
                continue
            data = read_json(path)
            if isinstance(data, dict):
                return data
            found_invalid.append(f"{path} ({type(data).__name__})")

        searched = ", ".join(str(p) for p in candidates)
        invalid = "; invalid non-object configs: " + ", ".join(found_invalid) if found_invalid else ""
        raise FileNotFoundError(
            "Live inference needs a full feature-engineering JSON object/dict. "
            f"Searched: {searched}{invalid}"
        )

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

    def _collect_dxy_frame(self, client: MT5DataClient, timeframe: str, count: int) -> Any:
        dxy_cfg = self.cfg.get("dxy", {}) or {}
        if not bool(dxy_cfg.get("enabled", False)):
            return None
        symbol_map = dxy_cfg.get("component_symbol_map", {}) or {}
        frames: dict[str, Any] = {}
        for dxy_symbol in ["EURUSD", "USDJPY", "GBPUSD", "USDCAD", "USDSEK", "USDCHF"]:
            broker_symbol = str(symbol_map.get(dxy_symbol, dxy_symbol))
            try:
                rates = client.latest_rates(broker_symbol, timeframe, count=count)
                frames[dxy_symbol] = rates_to_ohlc_frame(rates, symbol=dxy_symbol)
            except Exception as exc:
                if bool(dxy_cfg.get("fail_if_missing", False)):
                    raise
                print({"router_id": self.cfg.get("router_id"), "dxy_component_warning": dxy_symbol, "error": repr(exc)})
                return None
        return build_dxy_from_component_closes(frames)

    def _build_feature_snapshot_safe(self, *, symbol: str, rates: Any, event: BarEvent, dxy_frame: Any) -> LiveFeatureSnapshot | None:
        if self.feature_config is None or self.inference_engine is None:
            return None
        required_cols: list[str] = []
        for setup in self.engine.setups_for_symbol(symbol):
            if self.inference_engine.models.has_artifact(setup.setup_id):
                try:
                    art = self.inference_engine.models.load_artifact(setup.setup_id)
                    for c in art.feature_columns:
                        if c not in required_cols:
                            required_cols.append(c)
                except Exception:
                    pass
        if not required_cols:
            # Build all model columns if artifacts are not present yet. The inference
            # engine will safely return no_signal with live_model_artifact_missing.
            required_cols = None  # type: ignore[assignment]
        try:
            return build_live_feature_snapshot(
                symbol=symbol,
                rates=rates,
                feature_config=self.feature_config,
                signal_bar_time_utc=event.closed_bar_time_utc,
                dxy_frame=dxy_frame,
                required_feature_columns=required_cols,  # type: ignore[arg-type]
            )
        except Exception as exc:
            print({"router_id": self.cfg.get("router_id"), "feature_snapshot_error": symbol, "error": repr(exc)})
            return None

    def process_bar_event(self, event: BarEvent, *, feature_snapshot: LiveFeatureSnapshot | None = None) -> dict[str, Any]:
        if self.state.has_processed_bar(event.symbol, event.timeframe, event.closed_bar_time_utc):
            return {"symbol": event.symbol, "closed_bar_time_utc": event.closed_bar_time_utc, "status": "already_processed"}
        decisions = self.engine.evaluate_closed_bar(
            symbol=event.symbol,
            timeframe=event.timeframe,
            signal_bar_time_utc=event.closed_bar_time_utc,
            decision_bar_time_utc=event.current_bar_time_utc,
            inject_test_signal=self.inject_test_signal,
            feature_snapshot=feature_snapshot,
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
        symbol_rates: dict[str, Any] = {}
        events: dict[str, BarEvent] = {}
        for symbol in [str(s).upper() for s in self.cfg.get("symbols", [])]:
            rates = client.latest_rates(symbol, timeframe, count=bars)
            symbol_rates[symbol] = rates
            event = detect_new_bar(
                symbol=symbol,
                timeframe=timeframe,
                rates=rates,
                last_seen_current_bar_time=self.last_seen_current_bar.get(symbol),
            )
            if event is None:
                out.append({"symbol": symbol, "status": "no_new_bar"})
                continue
            events[symbol] = event
            self.last_seen_current_bar[symbol] = event.current_bar_time
            self.next_bar_time[symbol] = next_expected_bar_time(event.current_bar_time, timeframe)
        dxy_frame = self._collect_dxy_frame(client, timeframe, bars) if events and self.inference_engine is not None else None
        for symbol, event in events.items():
            snapshot = self._build_feature_snapshot_safe(symbol=symbol, rates=symbol_rates[symbol], event=event, dxy_frame=dxy_frame)
            out.append(self.process_bar_event(event, feature_snapshot=snapshot))
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
