from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from debco.ml.candidates import candidate_mask_for_job
from debco.ml.setup_inventory import SetupSpec, config_for_setup, list_setup_specs
from debco.utils.io import read_json

from .live_features import LiveFeatureSnapshot
from .model_registry import LiveModelRegistry
from .signal_engine import SetupRuntimeSpec


@dataclass(frozen=True)
class SetupInferenceResult:
    setup_id: str
    enabled: bool
    candidate_pass: bool
    probability: float | None
    threshold: float | None
    action: str
    reason: str


class LiveInferenceEngine:
    """Feature/model inference bridge for the forward-demo router.

    This engine is intentionally strict: if the live model artifact is missing,
    it returns ``no_signal`` instead of guessing. That lets v0.1.13b be used in
    dry-run mode safely while the user trains/verifies final live models.
    """

    def __init__(self, *, ml_config_path: str | Path, models_dir: str | Path, enabled: bool = True):
        self.enabled = bool(enabled)
        self.ml_config_path = Path(ml_config_path)
        self.ml_config = read_json(self.ml_config_path) if self.ml_config_path.exists() else {}
        self.models = LiveModelRegistry(models_dir)
        self.specs_by_id: dict[str, SetupSpec] = {s.setup_id: s for s in list_setup_specs(self.ml_config)} if self.ml_config else {}

    def _candidate_pass(self, setup: SetupRuntimeSpec, feature_row: pd.DataFrame) -> tuple[bool, str]:
        spec = self.specs_by_id.get(setup.setup_id)
        if spec is None:
            return False, "setup_spec_missing_in_ml_config"
        cfg = config_for_setup(self.ml_config, spec)
        try:
            mask = candidate_mask_for_job(feature_row.copy(), symbol=setup.symbol, side=setup.side, config=cfg)
            ok = bool(pd.Series(mask).fillna(False).astype(bool).iloc[-1]) if len(mask) else False
            return ok, "candidate_filter_pass" if ok else "candidate_filter_false"
        except Exception as exc:
            return False, f"candidate_filter_error:{type(exc).__name__}"

    def evaluate(self, setup: SetupRuntimeSpec, snapshot: LiveFeatureSnapshot | None) -> SetupInferenceResult:
        if not self.enabled:
            return SetupInferenceResult(setup.setup_id, False, False, None, None, "no_signal", "live_inference_disabled")
        if snapshot is None:
            return SetupInferenceResult(setup.setup_id, True, False, None, None, "no_signal", "feature_snapshot_missing")
        if not self.models.has_artifact(setup.setup_id):
            return SetupInferenceResult(setup.setup_id, True, False, None, None, "no_signal", "live_model_artifact_missing")
        try:
            artifact = self.models.load_artifact(setup.setup_id)
        except Exception as exc:
            return SetupInferenceResult(setup.setup_id, True, False, None, None, "no_signal", f"live_model_artifact_error:{type(exc).__name__}")
        candidate_ok, candidate_reason = self._candidate_pass(setup, snapshot.model_feature_row)
        if not candidate_ok:
            return SetupInferenceResult(setup.setup_id, True, False, None, artifact.live_probability_cutoff, "no_signal", candidate_reason)
        try:
            probability, artifact = self.models.predict_probability(setup.setup_id, snapshot.model_feature_row)
        except Exception as exc:
            return SetupInferenceResult(setup.setup_id, True, True, None, artifact.live_probability_cutoff, "no_signal", f"model_predict_error:{type(exc).__name__}")
        threshold = float(artifact.live_probability_cutoff)
        action = "enter" if probability >= threshold else "no_signal"
        reason = "probability_passed_live_cutoff" if action == "enter" else "probability_below_live_cutoff"
        return SetupInferenceResult(setup.setup_id, True, True, probability, threshold, action, reason)
