from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from debco.ml.calibration import calibration_metrics, fit_probability_calibrator
from debco.ml.candidates import candidate_mask_for_job
from debco.ml.thresholding import threshold_sweep_from_predictions
from debco.validation.cpcv import make_cpcv_splits
from debco.validation.walk_forward import make_walk_forward_splits


@dataclass(frozen=True)
class TrainingJob:
    symbol: str
    profile: str
    side: str

    @property
    def name(self) -> str:
        return f"{self.symbol}_{self.profile}_{self.side}"


def enabled_jobs(config: Mapping[str, Any]) -> list[TrainingJob]:
    out: list[TrainingJob] = []
    for item in config.get("jobs", []):
        if not bool(item.get("enabled", False)):
            continue
        out.append(TrainingJob(symbol=str(item["symbol"]), profile=str(item["profile"]), side=str(item["side"])))
    return out


def _job_path(base: Path, template: str, job: TrainingJob) -> Path:
    return base / template.format(symbol=job.symbol, profile=job.profile, side=job.side)


def load_job_frames(config: Mapping[str, Any], job: TrainingJob) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_cfg = config.get("input", {})
    ml_ready_dir = Path(input_cfg.get("ml_ready_dir", "data/ml_ready"))
    metadata_dir = Path(input_cfg.get("metadata_dir", "data/label_metadata"))
    ml_template = str(input_cfg.get("ml_ready_file_template", "{symbol}_{profile}_ml_ready_{side}.csv"))
    meta_template = str(input_cfg.get("metadata_file_template", "{symbol}_{profile}_metadata_{side}.csv"))
    ml_path = _job_path(ml_ready_dir, ml_template, job)
    meta_path = _job_path(metadata_dir, meta_template, job)
    if not ml_path.exists():
        raise FileNotFoundError(f"ML-ready file not found: {ml_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    ml = pd.read_csv(ml_path)
    meta_header = pd.read_csv(meta_path, nrows=0)
    date_cols = [c for c in ["date", "entry_date", "exit_date"] if c in meta_header.columns]
    meta = pd.read_csv(meta_path, parse_dates=date_cols)
    if len(ml) != len(meta):
        raise ValueError(f"ML-ready and metadata row counts differ for {job.name}: {len(ml)} vs {len(meta)}")
    return ml, meta


def split_xy(ml_ready: pd.DataFrame, target_col: str = "label") -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in ml_ready.columns:
        raise ValueError(f"Target column {target_col!r} is missing.")
    x = ml_ready.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(ml_ready[target_col], errors="coerce").astype("Int64")
    keep = y.notna()
    return x.loc[keep].reset_index(drop=True), y.loc[keep].astype(int).reset_index(drop=True)


def apply_missing_strategy(
    x: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    *,
    dropna: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if not dropna:
        return x.reset_index(drop=True), y.reset_index(drop=True), metadata.reset_index(drop=True)
    keep = ~x.isna().any(axis=1)
    return x.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True), metadata.loc[keep].reset_index(drop=True)


def build_validation_folds(config: Mapping[str, Any], metadata: pd.DataFrame) -> list[Any]:
    val = config.get("validation", {})
    method = str(val.get("method", "walk_forward")).lower()
    if method == "walk_forward":
        wf = val.get("walk_forward", {})
        return make_walk_forward_splits(
            len(metadata),
            train_window_bars=int(wf.get("train_window_bars", 12000)),
            test_window_bars=int(wf.get("test_window_bars", 2000)),
            step_bars=int(wf.get("step_bars", wf.get("test_window_bars", 2000))),
            expanding=bool(wf.get("expanding", False)),
            min_train_bars=int(wf.get("min_train_bars", wf.get("train_window_bars", 12000))),
            purge_bars=int(wf.get("purge_bars", 0)),
        )
    if method == "cpcv":
        cpcv = val.get("cpcv", {})
        return make_cpcv_splits(
            metadata,
            n_groups=int(cpcv.get("n_groups", 6)),
            n_test_groups=int(cpcv.get("n_test_groups", 2)),
            embargo_bars=int(cpcv.get("embargo_bars", 0)),
            purge=bool(cpcv.get("purge", True)),
            max_splits=cpcv.get("max_splits", None),
        )
    raise ValueError(f"Unsupported validation method: {method}")


def candidate_stats_from_mask(
    x: pd.DataFrame,
    y: pd.Series,
    candidate_mask: pd.Series,
    *,
    config: Mapping[str, Any],
) -> dict[str, float | str]:
    """Summarize candidate filtering without changing row indices.

    Candidate-based/meta-label validation should usually keep the original
    chronological row index and then select candidate rows inside each
    train/test time block. This avoids collapsing time and losing walk-forward
    folds when candidate rows are sparse.
    """
    before = int(len(x))
    mask = pd.Series(candidate_mask, index=x.index).fillna(False).astype(bool)
    after = int(mask.sum())
    y2 = y.loc[mask]
    return {
        "candidate_set": str(config.get("candidate_filter", {}).get("set_name", "all_candles")),
        "candidate_preset": str(config.get("candidate_filter", {}).get("preset", "all_candles")),
        "candidate_filter_enabled": float(bool(config.get("candidate_filter", {}).get("enabled", False))),
        "candidate_rows_before": float(before),
        "candidate_rows_after": float(after),
        "candidate_keep_ratio": float(after / before) if before else np.nan,
        "candidate_positive_rate": float((y2 == 1).mean()) if after else np.nan,
        "candidate_positive_count": float((y2 == 1).sum()) if after else 0.0,
        "candidate_negative_count": float((y2 == 0).sum()) if after else 0.0,
    }


def build_candidate_aware_validation_folds(
    config: Mapping[str, Any],
    metadata: pd.DataFrame,
    candidate_mask: pd.Series,
    y: pd.Series | None = None,
) -> tuple[list[Any], dict[str, float]]:
    """Build validation folds on the original timeline, then keep candidates.

    This is the safe default for candidate-based/meta-label training. We do not
    first compress the dataset to candidate rows, because that destroys the
    original time spacing and can reduce 9 walk-forward folds to 0 or 1 fold.
    """
    from debco.validation.walk_forward import ValidationFold

    cand_cfg = config.get("candidate_validation", {})
    mode = str(cand_cfg.get("mode", "base_timeline")).lower()
    if mode not in {"base_timeline", "filter_before_split"}:
        raise ValueError(f"Unsupported candidate_validation.mode: {mode}")

    mask = pd.Series(candidate_mask).reset_index(drop=True).fillna(False).astype(bool).to_numpy()

    if mode == "filter_before_split":
        filtered_meta = metadata.loc[mask].reset_index(drop=True)
        folds = build_validation_folds(config, filtered_meta)
        return folds, {
            "candidate_validation_mode": 0.0,
            "base_fold_count": float(len(folds)),
            "candidate_fold_count": float(len(folds)),
            "candidate_folds_skipped": 0.0,
        }

    base_folds = build_validation_folds(config, metadata)
    min_train = int(cand_cfg.get("min_train_candidates", 300))
    min_test = int(cand_cfg.get("min_test_candidates", 30))
    min_train_pos = int(cand_cfg.get("min_train_positives", 20))
    min_test_pos = int(cand_cfg.get("min_test_positives", 3))

    out = []
    skipped = 0
    for fold in base_folds:
        train_base = np.asarray(fold.train_idx, dtype=int)
        test_base = np.asarray(fold.test_idx, dtype=int)
        train_idx = train_base[mask[train_base]]
        test_idx = test_base[mask[test_base]]

        ok = len(train_idx) >= min_train and len(test_idx) >= min_test
        if ok and y is not None:
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            ok = ok and int((y_train == 1).sum()) >= min_train_pos and int((y_train == 0).sum()) >= min_train_pos
            ok = ok and int((y_test == 1).sum()) >= min_test_pos and int((y_test == 0).sum()) >= min_test_pos
        if ok:
            out.append(ValidationFold(fold_id=fold.fold_id, train_idx=train_idx, test_idx=test_idx))
        else:
            skipped += 1

    return out, {
        "candidate_validation_mode": 1.0,
        "base_fold_count": float(len(base_folds)),
        "candidate_fold_count": float(len(out)),
        "candidate_folds_skipped": float(skipped),
        "candidate_min_train_candidates": float(min_train),
        "candidate_min_test_candidates": float(min_test),
    }


def _safe_metric(fn, default: float = np.nan) -> float:
    try:
        value = float(fn())
        return value if np.isfinite(value) else default
    except Exception:
        return default


def binary_classification_metrics(y_true: np.ndarray, proba: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= float(threshold)).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else np.nan
    out = {
        "n": float(len(y_true)),
        "positive_rate": float(np.mean(y_true == 1)) if len(y_true) else np.nan,
        "predicted_positive_rate": float(np.mean(y_pred == 1)) if len(y_pred) else np.nan,
        "threshold": float(threshold),
        "accuracy": _safe_metric(lambda: accuracy_score(y_true, y_pred)),
        "precision": _safe_metric(lambda: precision_score(y_true, y_pred, zero_division=0)),
        "recall": _safe_metric(lambda: recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1": _safe_metric(lambda: f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": _safe_metric(lambda: balanced_accuracy_score(y_true, y_pred)),
        "mcc": _safe_metric(lambda: matthews_corrcoef(y_true, y_pred)),
        "roc_auc": _safe_metric(lambda: roc_auc_score(y_true, proba)),
        "average_precision": _safe_metric(lambda: average_precision_score(y_true, proba)),
        "log_loss": _safe_metric(lambda: log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "brier_score": _safe_metric(lambda: brier_score_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6))),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    return out


def _suggest_param(trial: Any, name: str, bounds: list[Any]) -> Any:
    lo, hi = bounds
    if isinstance(lo, int) and isinstance(hi, int):
        return trial.suggest_int(name, int(lo), int(hi))
    return trial.suggest_float(name, float(lo), float(hi))


def _scale_pos_weight(y: pd.Series, mode: Any) -> float | None:
    if mode != "auto":
        return None if mode is None else float(mode)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos <= 0:
        return 1.0
    return float(neg / pos)


def _make_xgb_classifier(config: Mapping[str, Any], params: Mapping[str, Any], y_train: pd.Series):
    from xgboost import XGBClassifier

    model_cfg = config.get("model", {})
    base = dict(model_cfg.get("xgboost_base_params", {}))
    base["random_state"] = int(model_cfg.get("random_seed", 42))
    base.update(params)
    spw = _scale_pos_weight(y_train, model_cfg.get("scale_pos_weight", "auto"))
    if spw is not None:
        base["scale_pos_weight"] = spw
    if str(base.get("missing", "nan")).lower() == "nan":
        base["missing"] = np.nan
    return XGBClassifier(**base)


def _default_metric_probability_column(config: Mapping[str, Any]) -> str:
    """Choose probability column used for fold_metrics and live y_pred.

    Calibration is useful for probability interpretation and Brier/ECE, but in
    candidate-based/meta-label experiments raw XGBoost scores are usually the
    safer ranking signal. Calibrated probabilities can be compressed around the
    base rate and a fixed high threshold may silently produce zero signals.
    """
    model_cfg = config.get("model", {})
    explicit = model_cfg.get("evaluation_probability_column")
    if explicit:
        return str(explicit)
    candidate_enabled = bool(config.get("candidate_filter", {}).get("enabled", False))
    if candidate_enabled:
        return "y_prob_raw"
    cal_enabled = bool(config.get("calibration", {}).get("enabled", False))
    return "y_prob_calibrated" if cal_enabled else "y_prob_raw"


def _select_probability_column(raw: np.ndarray, calibrated: np.ndarray, probability_column: str) -> np.ndarray:
    col = str(probability_column)
    if col == "y_prob_raw":
        return raw
    if col == "y_prob_calibrated":
        return calibrated
    raise ValueError(f"Unsupported evaluation_probability_column: {probability_column!r}")


def _time_inner_split(train_idx: np.ndarray, valid_fraction: float, min_valid: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(train_idx)
    valid_size = max(int(round(n * float(valid_fraction))), int(min_valid))
    valid_size = min(valid_size, max(1, n // 3)) if n >= 3 else 1
    split = n - valid_size
    return train_idx[:split], train_idx[split:]


def _train_calibration_split(train_idx: np.ndarray, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cal_cfg = config.get("calibration", {})
    if not bool(cal_cfg.get("enabled", False)):
        return train_idx, np.asarray([], dtype=int)
    return _time_inner_split(
        train_idx,
        float(cal_cfg.get("calibration_tail_fraction", 0.2)),
        int(cal_cfg.get("min_calibration_bars", 800)),
    )


def optimize_params_for_fold(x: pd.DataFrame, y: pd.Series, train_idx: np.ndarray, config: Mapping[str, Any]) -> dict[str, Any]:
    import optuna

    model_cfg = config.get("model", {})
    trials = int(model_cfg.get("optuna_trials", 25))
    if trials <= 0:
        return {}
    objective_metric = str(model_cfg.get("objective_metric", "average_precision"))
    search_space = model_cfg.get("search_space", {})
    inner_train_idx, inner_valid_idx = _time_inner_split(
        train_idx,
        float(model_cfg.get("inner_valid_fraction", 0.2)),
        int(model_cfg.get("min_inner_valid_bars", 500)),
    )

    def objective(trial: Any) -> float:
        params = {name: _suggest_param(trial, name, bounds) for name, bounds in search_space.items()}
        clf = _make_xgb_classifier(config, params, y.iloc[inner_train_idx])
        clf.fit(x.iloc[inner_train_idx], y.iloc[inner_train_idx])
        proba = clf.predict_proba(x.iloc[inner_valid_idx])[:, 1]
        metrics = binary_classification_metrics(y.iloc[inner_valid_idx].to_numpy(), proba, threshold=float(model_cfg.get("threshold", 0.6)))
        value = metrics.get(objective_metric, np.nan)
        if not np.isfinite(value):
            return -1e9
        return float(value)

    sampler = optuna.samplers.TPESampler(seed=int(model_cfg.get("random_seed", 42)))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return dict(study.best_params)


def _metadata_columns_for_predictions(metadata: pd.DataFrame, test_idx: np.ndarray) -> pd.DataFrame:
    keep = [c for c in ["date", "entry_date", "exit_date", "symbol", "profile", "side", "target_name"] if c in metadata.columns]
    if not keep:
        return pd.DataFrame(index=np.arange(len(test_idx)))
    out = metadata.iloc[test_idx][keep].reset_index(drop=True).copy()
    return out


def train_predict_fold(x: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame, fold: Any, config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any], dict[str, Any]]:
    train_idx = np.asarray(fold.train_idx, dtype=int)
    test_idx = np.asarray(fold.test_idx, dtype=int)
    if len(np.unique(y.iloc[train_idx])) < 2:
        raise ValueError(f"Fold {fold.fold_id} train labels have only one class.")

    model_train_idx, cal_idx = _train_calibration_split(train_idx, config)
    if len(np.unique(y.iloc[model_train_idx])) < 2:
        model_train_idx = train_idx
        cal_idx = np.asarray([], dtype=int)

    best_params = optimize_params_for_fold(x, y, model_train_idx, config)
    clf = _make_xgb_classifier(config, best_params, y.iloc[model_train_idx])
    clf.fit(x.iloc[model_train_idx], y.iloc[model_train_idx])

    raw_test = clf.predict_proba(x.iloc[test_idx])[:, 1]
    raw_cal = clf.predict_proba(x.iloc[cal_idx])[:, 1] if len(cal_idx) else np.asarray([], dtype=float)
    cal_cfg = config.get("calibration", {})
    calibrator = fit_probability_calibrator(
        raw_cal,
        y.iloc[cal_idx].to_numpy(dtype=int) if len(cal_idx) else np.asarray([], dtype=int),
        method=str(cal_cfg.get("method", "sigmoid")) if bool(cal_cfg.get("enabled", False)) else "none",
        fallback_to_raw_if_single_class=bool(cal_cfg.get("fallback_to_raw_if_single_class", True)),
        allow_inverted_sigmoid=bool(cal_cfg.get("allow_inverted_sigmoid", False)),
        random_seed=int(config.get("model", {}).get("random_seed", 42)),
    )
    calibrated_test = calibrator.predict(raw_test)

    model_cfg = config.get("model", {})
    threshold = float(model_cfg.get("threshold", 0.3))
    evaluation_probability_column = _default_metric_probability_column(config)
    prob_for_metrics = _select_probability_column(raw_test, calibrated_test, evaluation_probability_column)
    metrics = binary_classification_metrics(y.iloc[test_idx].to_numpy(), prob_for_metrics, threshold=threshold)
    metrics["fold"] = fold.fold_id
    metrics["probability_column"] = evaluation_probability_column
    metrics["calibration_method"] = str(cal_cfg.get("method", "none")) if bool(cal_cfg.get("enabled", False)) else "none"
    metrics["calibration_reason"] = calibrator.reason
    metrics["calibration_rows"] = float(len(cal_idx))
    metrics.update({f"calibrated_{k}": v for k, v in calibration_metrics(y.iloc[test_idx].to_numpy(), calibrated_test, bins=int(cal_cfg.get("ece_bins", 10))).items()})
    metrics.update({f"raw_{k}": v for k, v in calibration_metrics(y.iloc[test_idx].to_numpy(), raw_test, bins=int(cal_cfg.get("ece_bins", 10))).items()})

    meta_cols = _metadata_columns_for_predictions(metadata, test_idx)
    preds = pd.DataFrame({
        "fold": fold.fold_id,
        "row_idx": test_idx,
        "y_true": y.iloc[test_idx].to_numpy(dtype=int),
        "y_prob_raw": raw_test,
        "y_prob_calibrated": calibrated_test,
        "y_pred": (prob_for_metrics >= threshold).astype(int),
    })
    if not meta_cols.empty:
        preds = pd.concat([preds, meta_cols], axis=1)

    cal_summary = {
        "fold": fold.fold_id,
        "method": metrics["calibration_method"],
        "reason": calibrator.reason,
        "calibration_rows": int(len(cal_idx)),
        "test_rows": int(len(test_idx)),
    }
    cal_summary.update({f"raw_{k}": v for k, v in calibration_metrics(y.iloc[test_idx].to_numpy(), raw_test, bins=int(cal_cfg.get("ece_bins", 10))).items()})
    cal_summary.update({f"calibrated_{k}": v for k, v in calibration_metrics(y.iloc[test_idx].to_numpy(), calibrated_test, bins=int(cal_cfg.get("ece_bins", 10))).items()})
    return preds, metrics, best_params, cal_summary


def _mean_std_metrics(metrics: list[dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(metrics)
    numeric_cols = [c for c in df.columns if c != "fold" and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for c in numeric_cols:
        rows.append({"metric": c, "mean": df[c].mean(), "std": df[c].std(ddof=1), "min": df[c].min(), "max": df[c].max()})
    return pd.DataFrame(rows)


def _write_threshold_sweep(predictions: pd.DataFrame, job_dir: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    sw_cfg = config.get("threshold_sweep", {})
    if not bool(sw_cfg.get("enabled", True)):
        return pd.DataFrame()
    thresholds = [float(x) for x in sw_cfg.get("thresholds", [0.5, 0.6, 0.7])]
    prob_cols = [str(x) for x in sw_cfg.get("probability_columns", ["y_prob_calibrated", "y_prob_raw"])]
    sweep = threshold_sweep_from_predictions(predictions, thresholds=thresholds, probability_columns=prob_cols, y_col="y_true")
    if not sweep.empty:
        sweep.to_csv(job_dir / "threshold_sweep.csv", index=False)
    return sweep


def run_training_job(config: Mapping[str, Any], job: TrainingJob, output_root: Path) -> dict[str, Any]:
    ml_ready, metadata = load_job_frames(config, job)
    target_col = str(config.get("sanity", {}).get("target_column", "label"))
    x, y = split_xy(ml_ready, target_col=target_col)
    metadata = metadata.iloc[: len(x)].reset_index(drop=True)

    missing_cfg = config.get("missing_values", {})
    x, y, metadata = apply_missing_strategy(x, y, metadata, dropna=bool(missing_cfg.get("dropna", False)))

    candidate_mask = candidate_mask_for_job(x, symbol=job.symbol, side=job.side, config=config)
    candidate_mask = pd.Series(candidate_mask, index=x.index).fillna(False).astype(bool).reset_index(drop=True)
    candidate_stats = candidate_stats_from_mask(x, y, candidate_mask, config=config)

    folds, candidate_fold_stats = build_candidate_aware_validation_folds(config, metadata, candidate_mask, y)
    candidate_stats.update(candidate_fold_stats)
    if not folds:
        raise ValueError(
            f"No candidate-aware validation folds generated for {job.name}. "
            "Relax candidate_filter or lower candidate_validation minimums."
        )

    job_dir = output_root / job.name
    job_dir.mkdir(parents=True, exist_ok=True)
    all_predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, float]] = []
    fold_params: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for fold in folds:
        preds, metrics, best_params, cal_summary = train_predict_fold(x, y, metadata, fold, config)
        all_predictions.append(preds)
        fold_metrics.append(metrics)
        calibration_rows.append(cal_summary)
        row = {"fold": fold.fold_id}
        row.update(best_params)
        fold_params.append(row)
        print(
            f"{job.name} {fold.fold_id}: "
            f"n_test={int(metrics['n'])} prob={metrics.get('probability_column', '')} "
            f"thr={metrics.get('threshold', np.nan):.2f} ap={metrics.get('average_precision', np.nan):.4f} "
            f"auc={metrics.get('roc_auc', np.nan):.4f} precision={metrics.get('precision', np.nan):.4f} "
            f"recall={metrics.get('recall', np.nan):.4f} mcc={metrics.get('mcc', np.nan):.4f} "
            f"cal={metrics.get('calibration_reason', '')}"
        )

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    metrics_df = pd.DataFrame(fold_metrics)
    summary_df = _mean_std_metrics(fold_metrics)
    params_df = pd.DataFrame(fold_params)
    calibration_df = pd.DataFrame(calibration_rows)
    sweep_df = _write_threshold_sweep(predictions, job_dir, config)

    predictions.to_csv(job_dir / "oof_predictions.csv", index=False)
    # Compatibility with v0.1.4 name.
    predictions.to_csv(job_dir / "predictions.csv", index=False)
    metrics_df.to_csv(job_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(job_dir / "metrics_summary.csv", index=False)
    params_df.to_csv(job_dir / "best_params_by_fold.csv", index=False)
    calibration_df.to_csv(job_dir / "calibration_summary.csv", index=False)
    with (job_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    best_sweep = {}
    if not sweep_df.empty and "mcc" in sweep_df.columns:
        all_sweep = sweep_df[sweep_df["fold"].eq("ALL")]
        if not all_sweep.empty:
            best = all_sweep.sort_values(["mcc", "precision"], ascending=False).iloc[0]
            best_sweep = {
                "best_threshold": float(best["threshold"]),
                "best_threshold_prob_col": str(best["probability_column"]),
                "best_threshold_mcc": float(best["mcc"]),
                "best_threshold_precision": float(best["precision"]),
                "best_threshold_recall": float(best["recall"]),
                "best_threshold_signal_rate": float(best["signal_rate"]),
            }

    out = {
        "job": job.name,
        "rows": int(len(x)),
        "folds": int(len(folds)),
        "positive_rate": float((y == 1).mean()) if len(y) else np.nan,
        "output_dir": str(job_dir),
    }
    out.update(candidate_stats)
    out.update(best_sweep)
    return out
