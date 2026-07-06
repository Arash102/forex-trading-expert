from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:  # joblib is installed with scikit-learn in this project environment.
    import joblib
except Exception:  # pragma: no cover
    joblib = None


@dataclass(frozen=True)
class LiveModelArtifact:
    setup_id: str
    symbol: str
    side: str
    probability_column: str
    live_probability_cutoff: float
    feature_columns: list[str]
    model_path: Path | None
    calibrator_path: Path | None
    artifact_path: Path
    metadata: Mapping[str, Any]

    @property
    def is_valid_for_inference(self) -> bool:
        return self.model_path is not None and self.model_path.exists() and math.isfinite(float(self.live_probability_cutoff))


class LiveModelRegistry:
    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self._cache: dict[str, LiveModelArtifact] = {}
        self._model_cache: dict[str, Any] = {}
        self._cal_cache: dict[str, Any] = {}

    def artifact_dir(self, setup_id: str) -> Path:
        return self.models_dir / str(setup_id)

    def has_artifact(self, setup_id: str) -> bool:
        return (self.artifact_dir(setup_id) / "artifact.json").exists()

    def load_artifact(self, setup_id: str) -> LiveModelArtifact:
        sid = str(setup_id)
        if sid in self._cache:
            return self._cache[sid]
        path = self.artifact_dir(sid) / "artifact.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing live model artifact for {sid}: {path}")
        meta = json.loads(path.read_text(encoding="utf-8"))
        model_path = self.artifact_dir(sid) / str(meta.get("model_file", "model.joblib"))
        cal_file = meta.get("calibrator_file")
        cal_path = (self.artifact_dir(sid) / str(cal_file)) if cal_file else None
        artifact = LiveModelArtifact(
            setup_id=sid,
            symbol=str(meta.get("symbol", "")).upper(),
            side=str(meta.get("side", "")).lower(),
            probability_column=str(meta.get("probability_column", "y_prob_raw")),
            live_probability_cutoff=float(meta.get("live_probability_cutoff")),
            feature_columns=[str(x) for x in meta.get("feature_columns", [])],
            model_path=model_path,
            calibrator_path=cal_path,
            artifact_path=path,
            metadata=meta,
        )
        self._cache[sid] = artifact
        return artifact

    def _load_joblib(self, path: Path) -> Any:
        if joblib is None:
            raise RuntimeError("joblib is required to load live model artifacts.")
        return joblib.load(path)

    def predict_probability(self, setup_id: str, feature_row: pd.DataFrame) -> tuple[float, LiveModelArtifact]:
        artifact = self.load_artifact(setup_id)
        if not artifact.is_valid_for_inference:
            raise ValueError(f"Live model artifact for {setup_id} is not valid for inference.")
        x = feature_row.copy()
        for c in artifact.feature_columns:
            if c not in x.columns:
                x[c] = np.nan
        x = x[artifact.feature_columns].apply(pd.to_numeric, errors="coerce")
        sid = artifact.setup_id
        if sid not in self._model_cache:
            self._model_cache[sid] = self._load_joblib(artifact.model_path)  # type: ignore[arg-type]
        model = self._model_cache[sid]
        raw_prob = float(model.predict_proba(x)[:, 1][0])
        prob = raw_prob
        if artifact.probability_column == "y_prob_calibrated" and artifact.calibrator_path and artifact.calibrator_path.exists():
            if sid not in self._cal_cache:
                self._cal_cache[sid] = self._load_joblib(artifact.calibrator_path)
            cal = self._cal_cache[sid]
            if hasattr(cal, "predict"):
                prob = float(cal.predict(np.asarray([raw_prob], dtype=float))[0])
        return prob, artifact


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None
