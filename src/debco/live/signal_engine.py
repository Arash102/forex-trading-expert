from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import selected_setup_ids_from_spec


@dataclass(frozen=True)
class SetupRuntimeSpec:
    setup_id: str
    symbol: str
    side: str
    magic: int
    policy: str | None = None
    probability_column: str | None = None
    threshold: float | None = None
    top_percentile: float | None = None
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SignalDecision:
    symbol: str
    timeframe: str
    setup_id: str
    side: str
    magic: int
    signal_bar_time_utc: str
    decision_bar_time_utc: str
    action: str
    reason: str
    probability: float | None = None
    threshold: float | None = None
    dry_run: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "setup_id": self.setup_id,
            "side": self.side,
            "magic": self.magic,
            "signal_bar_time_utc": self.signal_bar_time_utc,
            "decision_bar_time_utc": self.decision_bar_time_utc,
            "action": self.action,
            "reason": self.reason,
            "probability": self.probability,
            "threshold": self.threshold,
            "dry_run": self.dry_run,
        }


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        import math

        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


class LiveSignalEngine:
    """Runtime signal engine for dry-run and live-inference modes.

    v0.1.13a validated timing/state/chart plumbing. v0.1.13b optionally
    attaches a LiveInferenceEngine that applies setup candidate filters,
    live model artifacts, and executable probability cutoffs. If inference is
    disabled or artifacts are missing, decisions remain safe ``no_signal``.
    """

    def __init__(
        self,
        live_spec: Mapping[str, Any],
        magic_numbers: Mapping[str, Any],
        *,
        dry_run: bool = True,
        inference_engine: Any | None = None,
    ):
        self.live_spec = live_spec
        self.magic_numbers = {str(k): int(v) for k, v in magic_numbers.items()}
        self.dry_run = bool(dry_run)
        self.inference_engine = inference_engine
        self.setups = self._build_setups()

    def _build_setups(self) -> list[SetupRuntimeSpec]:
        rows = self.live_spec.get("selected_setups", []) or []
        out: list[SetupRuntimeSpec] = []
        for row in rows:
            sid = str(row.get("setup_id", "")).strip()
            if not sid:
                continue
            out.append(
                SetupRuntimeSpec(
                    setup_id=sid,
                    symbol=str(row.get("symbol", "")).upper(),
                    side=str(row.get("side", "")).lower(),
                    magic=int(self.magic_numbers[sid]),
                    policy=str(row.get("policy")) if row.get("policy") is not None else None,
                    probability_column=str(row.get("probability_column")) if row.get("probability_column") is not None else None,
                    threshold=_to_optional_float(row.get("threshold")),
                    top_percentile=_to_optional_float(row.get("top_percentile")),
                    raw=row,
                )
            )
        return out

    @property
    def setup_ids(self) -> list[str]:
        return selected_setup_ids_from_spec(self.live_spec)

    def setups_for_symbol(self, symbol: str) -> list[SetupRuntimeSpec]:
        sym = str(symbol).upper()
        return [s for s in self.setups if s.symbol == sym]

    def evaluate_closed_bar(
        self,
        *,
        symbol: str,
        timeframe: str,
        signal_bar_time_utc: str,
        decision_bar_time_utc: str,
        inject_test_signal: str | None = None,
        feature_snapshot: Any | None = None,
    ) -> list[SignalDecision]:
        decisions: list[SignalDecision] = []
        for setup in self.setups_for_symbol(symbol):
            if inject_test_signal and setup.setup_id == inject_test_signal:
                decisions.append(
                    SignalDecision(
                        symbol=setup.symbol,
                        timeframe=timeframe,
                        setup_id=setup.setup_id,
                        side=setup.side,
                        magic=setup.magic,
                        signal_bar_time_utc=signal_bar_time_utc,
                        decision_bar_time_utc=decision_bar_time_utc,
                        action="enter",
                        reason="injected_test_signal_for_marker_and_state_validation",
                        probability=0.999,
                        threshold=setup.threshold,
                        dry_run=True,
                    )
                )
                continue
            if self.inference_engine is not None:
                res = self.inference_engine.evaluate(setup, feature_snapshot)
                decisions.append(
                    SignalDecision(
                        symbol=setup.symbol,
                        timeframe=timeframe,
                        setup_id=setup.setup_id,
                        side=setup.side,
                        magic=setup.magic,
                        signal_bar_time_utc=signal_bar_time_utc,
                        decision_bar_time_utc=decision_bar_time_utc,
                        action=res.action,
                        reason=res.reason,
                        probability=res.probability,
                        threshold=res.threshold,
                        dry_run=self.dry_run,
                    )
                )
                continue
            decisions.append(
                SignalDecision(
                    symbol=setup.symbol,
                    timeframe=timeframe,
                    setup_id=setup.setup_id,
                    side=setup.side,
                    magic=setup.magic,
                    signal_bar_time_utc=signal_bar_time_utc,
                    decision_bar_time_utc=decision_bar_time_utc,
                    action="no_signal",
                    reason="model_inference_not_enabled",
                    probability=None,
                    threshold=setup.threshold,
                    dry_run=self.dry_run,
                )
            )
        return decisions
