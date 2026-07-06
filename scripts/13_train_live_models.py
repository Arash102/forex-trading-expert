from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    import joblib
except Exception as exc:  # pragma: no cover
    raise SystemExit("joblib is required for live model export. Install scikit-learn/joblib first.") from exc

from debco.ml.calibration import fit_probability_calibrator
from debco.ml.candidates import apply_candidate_filter
from debco.ml.setup_inventory import SetupSpec, config_for_setup, list_setup_specs
from debco.ml.xgb_optuna import _make_xgb_classifier, apply_missing_strategy, load_job_frames, split_xy
from debco.utils.io import ensure_dir, read_json


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _selected_setups(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(x) for x in spec.get("selected_setups", []) or [] if x.get("setup_id")]


def _best_params_from_fold_file(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    out: dict[str, Any] = {}
    search_space = config.get("model", {}).get("search_space", {}) or {}
    for col in df.columns:
        if col == "fold":
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        val = float(vals.median())
        bounds = search_space.get(col)
        if isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(x, int) for x in bounds):
            out[col] = int(round(val))
        else:
            out[col] = val
    return out


def _oof_cutoff_for_setup(row: Mapping[str, Any], setup_cfg: Mapping[str, Any], output_root: Path) -> tuple[float, str]:
    threshold = _finite_or_none(row.get("threshold"))
    top_pct = _finite_or_none(row.get("top_percentile"))
    prob_col = str(row.get("probability_column", "y_prob_raw"))
    if threshold is not None:
        return float(threshold), "fixed_threshold_from_live_execution_spec"
    if top_pct is None:
        raise ValueError(f"{row.get('setup_id')} has neither threshold nor top_percentile.")
    job = str(row["job"])
    preds_path = output_root / str(row["experiment"]) / job / "oof_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing OOF predictions for top-percentile cutoff: {preds_path}")
    preds = pd.read_csv(preds_path)
    if prob_col not in preds.columns:
        raise ValueError(f"{preds_path} lacks probability column {prob_col!r}.")
    probs = pd.to_numeric(preds[prob_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if probs.empty:
        raise ValueError(f"No finite probabilities in {preds_path}:{prob_col}")
    q = max(0.0, min(1.0, 1.0 - float(top_pct) / 100.0))
    cutoff = float(probs.quantile(q))
    return cutoff, f"oof_quantile_from_top_percentile_{top_pct:g}"


def _time_tail_split(n: int, tail_fraction: float, min_tail: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 10:
        idx = np.arange(n)
        return idx, np.asarray([], dtype=int)
    tail = max(int(round(n * float(tail_fraction))), int(min_tail))
    tail = min(tail, max(1, n // 3))
    split = n - tail
    return np.arange(split, dtype=int), np.arange(split, n, dtype=int)


def train_one_setup(
    *,
    base_cfg: Mapping[str, Any],
    live_spec: Mapping[str, Any],
    setup_row: Mapping[str, Any],
    spec_by_id: Mapping[str, SetupSpec],
    models_dir: Path,
) -> dict[str, Any]:
    setup_id = str(setup_row["setup_id"])
    spec = spec_by_id[setup_id]
    cfg = config_for_setup(base_cfg, spec)
    job = spec.job
    ml_ready, metadata = load_job_frames(cfg, job)
    target_col = str(cfg.get("sanity", {}).get("target_column", "label"))
    x, y = split_xy(ml_ready, target_col=target_col)
    metadata = metadata.iloc[: len(x)].reset_index(drop=True)
    missing_cfg = cfg.get("missing_values", {})
    x, y, metadata = apply_missing_strategy(x, y, metadata, dropna=bool(missing_cfg.get("dropna", False)))
    x, y, metadata, candidate_stats = apply_candidate_filter(x, y, metadata, symbol=job.symbol, side=job.side, config=cfg)
    if len(x) < 50 or len(np.unique(y)) < 2:
        raise ValueError(f"Not enough candidate rows/classes to train live model for {setup_id}: rows={len(x)} classes={sorted(set(y))}")
    output_root = Path(base_cfg.get("output", {}).get("results_dir", "data/ml_results"))
    fold_params_path = output_root / str(setup_row["experiment"]) / job.name / "best_params_by_fold.csv"
    best_params = _best_params_from_fold_file(fold_params_path, cfg)
    cal_cfg = cfg.get("calibration", {}) or {}
    train_idx, cal_idx = _time_tail_split(
        len(x),
        float(cal_cfg.get("calibration_tail_fraction", 0.2)),
        int(cal_cfg.get("min_calibration_bars", 300)),
    )
    if len(cal_idx) == 0 or len(np.unique(y.iloc[train_idx])) < 2:
        train_idx = np.arange(len(x), dtype=int)
        cal_idx = np.asarray([], dtype=int)
    clf = _make_xgb_classifier(cfg, best_params, y.iloc[train_idx])
    clf.fit(x.iloc[train_idx], y.iloc[train_idx])
    raw_cal = clf.predict_proba(x.iloc[cal_idx])[:, 1] if len(cal_idx) else np.asarray([], dtype=float)
    calibrator = fit_probability_calibrator(
        raw_cal,
        y.iloc[cal_idx].to_numpy(dtype=int) if len(cal_idx) else np.asarray([], dtype=int),
        method=str(cal_cfg.get("method", "sigmoid")) if bool(cal_cfg.get("enabled", False)) else "none",
        fallback_to_raw_if_single_class=bool(cal_cfg.get("fallback_to_raw_if_single_class", True)),
        random_seed=int(cfg.get("model", {}).get("random_seed", 42)),
    )
    cutoff, cutoff_reason = _oof_cutoff_for_setup(setup_row, cfg, output_root)
    setup_dir = ensure_dir(models_dir / setup_id)
    model_file = setup_dir / "model.joblib"
    cal_file = setup_dir / "calibrator.joblib"
    joblib.dump(clf, model_file)
    joblib.dump(calibrator, cal_file)
    artifact = {
        "setup_id": setup_id,
        "symbol": spec.symbol,
        "side": spec.side,
        "job": job.name,
        "experiment": str(setup_row["experiment"]),
        "model_file": "model.joblib",
        "calibrator_file": "calibrator.joblib",
        "probability_column": str(setup_row.get("probability_column", "y_prob_raw")),
        "live_probability_cutoff": float(cutoff),
        "cutoff_reason": cutoff_reason,
        "policy": setup_row.get("policy"),
        "threshold": _finite_or_none(setup_row.get("threshold")),
        "top_percentile": _finite_or_none(setup_row.get("top_percentile")),
        "feature_columns": list(x.columns),
        "train_rows": int(len(train_idx)),
        "calibration_rows": int(len(cal_idx)),
        "candidate_rows": int(len(x)),
        "candidate_positive_rate": float((y == 1).mean()),
        "candidate_stats": candidate_stats,
        "best_params": best_params,
        "calibration_reason": getattr(calibrator, "reason", "unknown"),
    }
    (setup_dir / "artifact.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return {
        "setup_id": setup_id,
        "symbol": spec.symbol,
        "side": spec.side,
        "job": job.name,
        "candidate_rows": int(len(x)),
        "train_rows": int(len(train_idx)),
        "calibration_rows": int(len(cal_idx)),
        "probability_column": artifact["probability_column"],
        "live_probability_cutoff": float(cutoff),
        "cutoff_reason": cutoff_reason,
        "calibration_reason": artifact["calibration_reason"],
        "artifact_dir": str(setup_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/export one final live model artifact per selected setup.")
    parser.add_argument("--ml-config", default="configs/ml_config.local.json")
    parser.add_argument("--live-spec", default="data/final_strategy_report/live_execution_spec.json")
    parser.add_argument("--models-dir", default="data/live_models")
    parser.add_argument("--setup-id", default=None)
    args = parser.parse_args()

    base_cfg = read_json(args.ml_config)
    live_spec = read_json(args.live_spec)
    selected = _selected_setups(live_spec)
    if args.setup_id:
        selected = [r for r in selected if str(r.get("setup_id")) == args.setup_id]
    if not selected:
        raise SystemExit("No selected setup rows matched.")
    spec_by_id = {s.setup_id: s for s in list_setup_specs(base_cfg)}
    missing = [str(r["setup_id"]) for r in selected if str(r["setup_id"]) not in spec_by_id]
    if missing:
        raise SystemExit(f"Missing setup definitions in ml config: {missing}")
    models_dir = ensure_dir(Path(args.models_dir))
    rows = []
    for row in selected:
        print(f"\n=== TRAIN LIVE MODEL: {row['setup_id']} ===")
        rows.append(train_one_setup(base_cfg=base_cfg, live_spec=live_spec, setup_row=row, spec_by_id=spec_by_id, models_dir=models_dir))
    out = pd.DataFrame(rows)
    out.to_csv(models_dir / "live_model_training_summary.csv", index=False)
    print(f"\nsaved: {models_dir / 'live_model_training_summary.csv'}")


if __name__ == "__main__":
    main()
