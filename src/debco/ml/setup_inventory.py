from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from debco.ml.candidates import candidate_mask_for_job
from debco.ml.xgb_optuna import (
    TrainingJob,
    apply_missing_strategy,
    build_candidate_aware_validation_folds,
    candidate_stats_from_mask,
    load_job_frames,
    split_xy,
)


@dataclass(frozen=True)
class SetupSpec:
    setup_id: str
    symbol: str
    side: str
    profile: str
    description: str
    family: str
    enabled: bool
    candidate_filter: dict[str, Any]

    @property
    def job(self) -> TrainingJob:
        return TrainingJob(symbol=self.symbol, profile=self.profile, side=self.side)

    @property
    def candidate_set_name(self) -> str:
        return self.setup_id.lower()


def list_setup_specs(config: Mapping[str, Any], *, enabled_only: bool = True) -> list[SetupSpec]:
    inv = config.get("setup_inventory", {})
    specs: list[SetupSpec] = []
    for item in inv.get("setups", []):
        enabled = bool(item.get("enabled", True))
        if enabled_only and not enabled:
            continue
        setup_id = str(item["setup_id"])
        symbol = str(item["symbol"]).upper()
        side = str(item["side"]).lower()
        profile = str(item["profile"])
        cf = copy.deepcopy(dict(item.get("candidate_filter", {})))
        cf.setdefault("enabled", True)
        cf.setdefault("preset", "setup_specific_v1")
        cf["set_name"] = setup_id.lower()
        cf["setup_id"] = setup_id
        if "setup_params" in item:
            cf["setup_params"] = copy.deepcopy(item.get("setup_params", {}))
        specs.append(
            SetupSpec(
                setup_id=setup_id,
                symbol=symbol,
                side=side,
                profile=profile,
                description=str(item.get("description", "")),
                family=str(item.get("family", "")),
                enabled=enabled,
                candidate_filter=cf,
            )
        )
    return specs


def setup_matches_filters(
    spec: SetupSpec,
    *,
    setup_id: str | None = None,
    symbol: str | None = None,
    side: str | None = None,
    family: str | None = None,
) -> bool:
    if setup_id and spec.setup_id != setup_id:
        return False
    if symbol and spec.symbol != symbol.upper():
        return False
    if side and spec.side != side.lower():
        return False
    if family and spec.family != family:
        return False
    return True


def config_for_setup(config: Mapping[str, Any], spec: SetupSpec) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(config))
    cfg["candidate_filter"] = copy.deepcopy(spec.candidate_filter)
    out = cfg.setdefault("output", {})
    base_name = str(config.get("setup_inventory", {}).get("base_experiment_name", "xgb_v0_1_10_setup_inventory"))
    out["experiment_name"] = f"{base_name}_{spec.setup_id.lower()}"
    # Disable candidate_experiments so the standard training function uses this exact filter.
    cfg.setdefault("candidate_experiments", {})
    cfg["candidate_experiments"]["enabled"] = False
    return cfg


def setup_candidate_summary(config: Mapping[str, Any], spec: SetupSpec) -> dict[str, Any]:
    cfg = config_for_setup(config, spec)
    job = spec.job
    ml_ready, metadata = load_job_frames(cfg, job)
    target = str(cfg.get("sanity", {}).get("target_column", "label"))
    x, y = split_xy(ml_ready, target_col=target)
    metadata = metadata.iloc[: len(x)].reset_index(drop=True)
    missing_cfg = cfg.get("missing_values", {})
    x, y, metadata = apply_missing_strategy(x, y, metadata, dropna=bool(missing_cfg.get("dropna", False)))
    mask = candidate_mask_for_job(x, symbol=job.symbol, side=job.side, config=cfg)
    mask = pd.Series(mask, index=x.index).fillna(False).astype(bool).reset_index(drop=True)
    y2 = y.loc[mask].reset_index(drop=True)
    folds, fold_stats = build_candidate_aware_validation_folds(cfg, metadata, mask, y)
    row: dict[str, Any] = {
        "setup_id": spec.setup_id,
        "candidate_set": spec.candidate_set_name,
        "family": spec.family,
        "description": spec.description,
        "job": job.name,
        "symbol": spec.symbol,
        "profile": spec.profile,
        "side": spec.side,
        "rows_before": len(x),
        "rows_after": int(mask.sum()),
        "keep_ratio": float(mask.mean()) if len(mask) else 0.0,
        "positive_rate_before": float((y == 1).mean()) if len(y) else 0.0,
        "positive_rate_after": float((y2 == 1).mean()) if len(y2) else 0.0,
        "positive_lift": (float((y2 == 1).mean()) / float((y == 1).mean())) if len(y2) and float((y == 1).mean()) > 0 else 0.0,
        "positive_count_after": int((y2 == 1).sum()) if len(y2) else 0,
        "negative_count_after": int((y2 == 0).sum()) if len(y2) else 0,
        "fold_count_after": len(folds),
    }
    row.update(candidate_stats_from_mask(x, y, mask, config=cfg))
    row.update(fold_stats)
    if int(mask.sum()) and "session_block_id" in x.columns:
        row["session_block_counts_after"] = str(x.loc[mask, "session_block_id"].value_counts(dropna=False).sort_index().to_dict())
    return row
