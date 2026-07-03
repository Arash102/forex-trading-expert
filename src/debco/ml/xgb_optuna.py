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
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from debco.validation.cpcv import make_cpcv_splits
from debco.validation.walk_forward import ValidationFold, make_walk_forward_splits


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
        out.append(
            TrainingJob(
                symbol=str(item["symbol"]),
                profile=str(item["profile"]),
                side=str(item["side"]),
            )
        )
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
    meta = pd.read_csv(meta_path, parse_dates=[c for c in ["date", "entry_date", "exit_date"] if c in pd.read_csv(meta_path, nrows=0).columns])
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
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
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
    if name in {"learning_rate", "reg_alpha", "reg_lambda", "gamma"}:
        # gamma/reg params include zero in search space; use linear scale to allow zero.
        return trial.suggest_float(name, float(lo), float(hi))
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
    # XGBoost expects np.nan for missing. JSON config uses "nan" for readability.
    if str(base.get("missing", "nan")).lower() == "nan":
        base["missing"] = np.nan
    return XGBClassifier(**base)


def _time_inner_split(train_idx: np.ndarray, valid_fraction: float, min_valid: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(train_idx)
    valid_size = max(int(round(n * float(valid_fraction))), int(min_valid))
    valid_size = min(valid_size, max(1, n // 3)) if n >= 3 else 1
    split = n - valid_size
    return train_idx[:split], train_idx[split:]


def optimize_params_for_fold(
    x: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
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
    # If the inner validation has one class only, still run but objective may be NaN.
    def objective(trial: Any) -> float:
        params = {name: _suggest_param(trial, name, bounds) for name, bounds in search_space.items()}
        clf = _make_xgb_classifier(config, params, y.iloc[inner_train_idx])
        clf.fit(x.iloc[inner_train_idx], y.iloc[inner_train_idx])
        proba = clf.predict_proba(x.iloc[inner_valid_idx])[:, 1]
        metrics = binary_classification_metrics(
            y.iloc[inner_valid_idx].to_numpy(),
            proba,
            threshold=float(model_cfg.get("threshold", 0.6)),
        )
        value = metrics.get(objective_metric, np.nan)
        if not np.isfinite(value):
            return -1e9
        return float(value)

    sampler = optuna.samplers.TPESampler(seed=int(model_cfg.get("random_seed", 42)))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return dict(study.best_params)


def train_predict_fold(
    x: pd.DataFrame,
    y: pd.Series,
    fold: Any,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
    train_idx = np.asarray(fold.train_idx, dtype=int)
    test_idx = np.asarray(fold.test_idx, dtype=int)
    if len(np.unique(y.iloc[train_idx])) < 2:
        raise ValueError(f"Fold {fold.fold_id} train labels have only one class.")
    best_params = optimize_params_for_fold(x, y, train_idx, config)
    clf = _make_xgb_classifier(config, best_params, y.iloc[train_idx])
    clf.fit(x.iloc[train_idx], y.iloc[train_idx])
    proba = clf.predict_proba(x.iloc[test_idx])[:, 1]
    threshold = float(config.get("model", {}).get("threshold", 0.6))
    metrics = binary_classification_metrics(y.iloc[test_idx].to_numpy(), proba, threshold=threshold)
    metrics["fold"] = fold.fold_id
    preds = pd.DataFrame({
        "fold": fold.fold_id,
        "row_idx": test_idx,
        "y_true": y.iloc[test_idx].to_numpy(dtype=int),
        "proba": proba,
        "y_pred": (proba >= threshold).astype(int),
    })
    return preds, metrics, best_params


def _mean_std_metrics(metrics: list[dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(metrics)
    numeric_cols = [c for c in df.columns if c != "fold" and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for c in numeric_cols:
        rows.append({"metric": c, "mean": df[c].mean(), "std": df[c].std(ddof=1), "min": df[c].min(), "max": df[c].max()})
    return pd.DataFrame(rows)


def run_training_job(config: Mapping[str, Any], job: TrainingJob, output_root: Path) -> dict[str, Any]:
    ml_ready, metadata = load_job_frames(config, job)
    target_col = str(config.get("sanity", {}).get("target_column", "label"))
    x, y = split_xy(ml_ready, target_col=target_col)
    # Align metadata after split_xy. In current pipeline labels are complete, but this keeps it safe.
    metadata = metadata.iloc[: len(x)].reset_index(drop=True)
    missing_cfg = config.get("missing_values", {})
    x, y, metadata = apply_missing_strategy(
        x,
        y,
        metadata,
        dropna=bool(missing_cfg.get("dropna", False)),
    )
    folds = build_validation_folds(config, metadata)
    if not folds:
        raise ValueError(f"No validation folds generated for {job.name}.")

    job_dir = output_root / job.name
    job_dir.mkdir(parents=True, exist_ok=True)
    all_predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, float]] = []
    fold_params: list[dict[str, Any]] = []
    for fold in folds:
        preds, metrics, best_params = train_predict_fold(x, y, fold, config)
        all_predictions.append(preds)
        fold_metrics.append(metrics)
        row = {"fold": fold.fold_id}
        row.update(best_params)
        fold_params.append(row)
        print(
            f"{job.name} {fold.fold_id}: "
            f"n_test={int(metrics['n'])} ap={metrics.get('average_precision', np.nan):.4f} "
            f"auc={metrics.get('roc_auc', np.nan):.4f} precision={metrics.get('precision', np.nan):.4f} "
            f"recall={metrics.get('recall', np.nan):.4f} mcc={metrics.get('mcc', np.nan):.4f}"
        )

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    metrics_df = pd.DataFrame(fold_metrics)
    summary_df = _mean_std_metrics(fold_metrics)
    params_df = pd.DataFrame(fold_params)

    predictions.to_csv(job_dir / "predictions.csv", index=False)
    metrics_df.to_csv(job_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(job_dir / "metrics_summary.csv", index=False)
    params_df.to_csv(job_dir / "best_params_by_fold.csv", index=False)
    with (job_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return {
        "job": job.name,
        "rows": int(len(x)),
        "folds": int(len(folds)),
        "positive_rate": float((y == 1).mean()),
        "output_dir": str(job_dir),
    }
