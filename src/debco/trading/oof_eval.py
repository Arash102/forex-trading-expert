from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from debco.ml.xgb_optuna import TrainingJob
from debco.trading.risk_metrics import summarize_trade_pnl
from debco.trading.threshold_policy import (
    fixed_threshold_mask,
    rolling_oof_target_precision_mask,
    threshold_metrics_for_selection,
    top_percentile_mask_by_fold,
)


def job_name_to_training_job(job_name: str) -> TrainingJob:
    if job_name.startswith("EURUSD_"):
        symbol = "EURUSD"
        rest = job_name[len("EURUSD_") :]
    elif job_name.startswith("XAUUSD_"):
        symbol = "XAUUSD"
        rest = job_name[len("XAUUSD_") :]
    else:
        parts = job_name.split("_")
        symbol, rest = parts[0], "_".join(parts[1:])
    side = "long" if rest.endswith("_long") else "short"
    profile = rest[: -len(f"_{side}")]
    return TrainingJob(symbol=symbol, profile=profile, side=side)


def metadata_path_for_job(config: Mapping[str, Any], job: TrainingJob) -> Path:
    input_cfg = config.get("input", {})
    metadata_dir = Path(input_cfg.get("metadata_dir", "data/label_metadata"))
    template = str(input_cfg.get("metadata_file_template", "{symbol}_{profile}_metadata_{side}.csv"))
    return metadata_dir / template.format(symbol=job.symbol, profile=job.profile, side=job.side)


def load_metadata_for_job(config: Mapping[str, Any], job: TrainingJob) -> pd.DataFrame:
    path = metadata_path_for_job(config, job)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    header = pd.read_csv(path, nrows=0)
    date_cols = [c for c in ["date", "entry_date", "exit_date"] if c in header.columns]
    return pd.read_csv(path, parse_dates=date_cols)


def load_oof_predictions(job_dir: Path) -> pd.DataFrame:
    for name in ["oof_predictions.csv", "predictions.csv"]:
        path = job_dir / name
        if path.exists():
            header = pd.read_csv(path, nrows=0)
            date_cols = [c for c in ["date", "entry_date", "exit_date"] if c in header.columns]
            return pd.read_csv(path, parse_dates=date_cols)
    raise FileNotFoundError(f"No oof_predictions.csv or predictions.csv found in {job_dir}")


def enrich_predictions_with_trade_outcomes(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if "row_idx" not in predictions.columns:
        raise ValueError("Predictions must include row_idx to join metadata outcomes.")
    out = predictions.copy()
    row_idx = pd.to_numeric(out["row_idx"], errors="coerce").astype("Int64")
    valid = row_idx.notna() & (row_idx >= 0) & (row_idx < len(metadata))
    if not bool(valid.all()):
        out = out.loc[valid].copy()
        row_idx = row_idx.loc[valid]
    meta_take = metadata.iloc[row_idx.astype(int).to_numpy()].reset_index(drop=True)
    for col in ["realized_pips", "tp_pips", "sl_pips", "event_type", "outcome_label", "bars_to_event", "entry_price", "exit_price"]:
        if col in meta_take.columns and col not in out.columns:
            out[col] = meta_take[col].to_numpy()
    for col in ["date", "entry_date", "exit_date", "symbol", "profile", "side"]:
        if col in meta_take.columns and col not in out.columns:
            out[col] = meta_take[col].to_numpy()
    out["pnl_pips"] = pd.to_numeric(out.get("realized_pips"), errors="coerce")
    sl = pd.to_numeric(out.get("sl_pips"), errors="coerce").replace(0, np.nan).abs()
    out["pnl_R"] = out["pnl_pips"] / sl
    return out


def _selected_trade_frame(enriched: pd.DataFrame, mask: pd.Series, *, policy_name: str, probability_column: str, threshold: float | None = None, top_percentile: float | None = None) -> pd.DataFrame:
    trades = enriched.loc[mask.fillna(False).astype(bool)].copy()
    trades["policy"] = policy_name
    trades["probability_column"] = probability_column
    if threshold is not None:
        trades["threshold"] = float(threshold)
    if top_percentile is not None:
        trades["top_percentile"] = float(top_percentile)
    return trades


def evaluate_trade_selection(
    enriched: pd.DataFrame,
    mask: pd.Series,
    *,
    policy: str,
    probability_column: str,
    initial_capital: float,
    risk_per_trade: float,
    ruin_drawdowns: list[float],
    n_ruin_sims: int,
    random_seed: int,
    threshold: float | None = None,
    top_percentile: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    trades = _selected_trade_frame(enriched, mask, policy_name=policy, probability_column=probability_column, threshold=threshold, top_percentile=top_percentile)
    summary = summarize_trade_pnl(
        trades,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        ruin_drawdowns=ruin_drawdowns,
        n_ruin_sims=n_ruin_sims,
        random_seed=random_seed,
    )
    cls = threshold_metrics_for_selection(enriched["y_true"], mask) if "y_true" in enriched.columns else {}
    summary.update({f"label_{k}": v for k, v in cls.items()})
    summary.update({"policy": policy, "probability_column": probability_column})
    if threshold is not None:
        summary["threshold"] = float(threshold)
    if top_percentile is not None:
        summary["top_percentile"] = float(top_percentile)
    return summary, trades


def fold_level_trade_metrics(
    enriched: pd.DataFrame,
    mask: pd.Series,
    *,
    policy: str,
    probability_column: str,
    initial_capital: float,
    risk_per_trade: float,
    ruin_drawdowns: list[float],
    threshold: float | None = None,
    top_percentile: float | None = None,
) -> pd.DataFrame:
    if "fold" not in enriched.columns:
        return pd.DataFrame()
    rows = []
    for fold, idx in enriched.groupby("fold").groups.items():
        sub = enriched.loc[idx]
        sub_mask = mask.loc[idx]
        summary = summarize_trade_pnl(
            sub.loc[sub_mask.fillna(False).astype(bool)].copy(),
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            ruin_drawdowns=ruin_drawdowns,
            n_ruin_sims=0,
        )
        summary.update({"fold": fold, "policy": policy, "probability_column": probability_column})
        if threshold is not None:
            summary["threshold"] = float(threshold)
        if top_percentile is not None:
            summary["top_percentile"] = float(top_percentile)
        rows.append(summary)
    return pd.DataFrame(rows)


def evaluate_job_trading(
    config: Mapping[str, Any],
    *,
    job_dir: Path,
    job_name: str | None = None,
) -> dict[str, pd.DataFrame]:
    te = config.get("trading_eval", {})
    job = job_name_to_training_job(job_name or job_dir.name)
    predictions = load_oof_predictions(job_dir)
    metadata = load_metadata_for_job(config, job)
    enriched = enrich_predictions_with_trade_outcomes(predictions, metadata)

    initial_capital = float(te.get("initial_capital", 1000.0))
    risk_per_trade = float(te.get("risk_per_trade", 0.02))
    ruin_drawdowns = [float(x) for x in te.get("ruin_drawdowns", [0.25, 0.30])]
    n_ruin_sims = int(te.get("risk_of_ruin_simulations", 5000))
    random_seed = int(te.get("random_seed", 42))
    prob_cols = [str(c) for c in te.get("probability_columns", ["y_prob_raw", "y_prob_calibrated"])]
    fixed_thresholds = [float(x) for x in te.get("fixed_thresholds", [0.3, 0.4, 0.5, 0.6, 0.7])]
    top_percentiles = [float(x) for x in te.get("top_percentiles", [1, 2, 5, 10])]

    fixed_rows = []
    fold_rows = []
    trade_rows = []
    for col in prob_cols:
        if col not in enriched.columns:
            continue
        for thr in fixed_thresholds:
            mask = fixed_threshold_mask(enriched, probability_column=col, threshold=thr)
            summary, trades = evaluate_trade_selection(
                enriched,
                mask,
                policy="fixed_threshold",
                probability_column=col,
                threshold=thr,
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade,
                ruin_drawdowns=ruin_drawdowns,
                n_ruin_sims=n_ruin_sims,
                random_seed=random_seed,
            )
            fixed_rows.append(summary)
            fm = fold_level_trade_metrics(
                enriched,
                mask,
                policy="fixed_threshold",
                probability_column=col,
                threshold=thr,
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade,
                ruin_drawdowns=ruin_drawdowns,
            )
            fold_rows.append(fm)
            if not trades.empty:
                trade_rows.append(trades)

    top_rows = []
    for col in prob_cols:
        if col not in enriched.columns:
            continue
        for pct in top_percentiles:
            mask = top_percentile_mask_by_fold(enriched, probability_column=col, top_percentile=pct)
            summary, trades = evaluate_trade_selection(
                enriched,
                mask,
                policy="top_percentile_by_fold",
                probability_column=col,
                top_percentile=pct,
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade,
                ruin_drawdowns=ruin_drawdowns,
                n_ruin_sims=n_ruin_sims,
                random_seed=random_seed,
            )
            top_rows.append(summary)
            fm = fold_level_trade_metrics(
                enriched,
                mask,
                policy="top_percentile_by_fold",
                probability_column=col,
                top_percentile=pct,
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade,
                ruin_drawdowns=ruin_drawdowns,
            )
            fold_rows.append(fm)
            if not trades.empty:
                trade_rows.append(trades)

    rolling_rows = []
    rolling_threshold_rows = []
    target_cfg = te.get("target_precision_policy", {})
    if bool(target_cfg.get("enabled", True)):
        for col in prob_cols:
            if col not in enriched.columns:
                continue
            mask, threshold_log = rolling_oof_target_precision_mask(
                enriched,
                probability_column=col,
                thresholds=fixed_thresholds,
                target_precision=float(target_cfg.get("target_precision", 0.60)),
                min_past_trades=int(target_cfg.get("min_past_trades", 30)),
                fallback_threshold=target_cfg.get("fallback_threshold", None),
            )
            summary, trades = evaluate_trade_selection(
                enriched,
                mask,
                policy="rolling_oof_target_precision",
                probability_column=col,
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade,
                ruin_drawdowns=ruin_drawdowns,
                n_ruin_sims=n_ruin_sims,
                random_seed=random_seed,
            )
            rolling_rows.append(summary)
            threshold_log.insert(0, "probability_column", col)
            rolling_threshold_rows.append(threshold_log)
            fm = fold_level_trade_metrics(
                enriched,
                mask,
                policy="rolling_oof_target_precision",
                probability_column=col,
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade,
                ruin_drawdowns=ruin_drawdowns,
            )
            fold_rows.append(fm)
            if not trades.empty:
                trade_rows.append(trades)

    return {
        "enriched_predictions": enriched,
        "fixed_threshold_summary": pd.DataFrame(fixed_rows),
        "top_percentile_summary": pd.DataFrame(top_rows),
        "rolling_target_precision_summary": pd.DataFrame(rolling_rows),
        "rolling_threshold_log": pd.concat(rolling_threshold_rows, ignore_index=True) if rolling_threshold_rows else pd.DataFrame(),
        "trading_fold_metrics": pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame(),
        "selected_trades": pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame(),
    }
