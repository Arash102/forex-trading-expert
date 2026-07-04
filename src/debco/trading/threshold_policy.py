from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_threshold_mask(predictions: pd.DataFrame, *, probability_column: str, threshold: float) -> pd.Series:
    if probability_column not in predictions.columns:
        raise ValueError(f"Missing probability column: {probability_column}")
    return pd.to_numeric(predictions[probability_column], errors="coerce") >= float(threshold)


def top_percentile_mask_by_fold(predictions: pd.DataFrame, *, probability_column: str, top_percentile: float) -> pd.Series:
    """Select the top X percent of probability scores inside each test fold.

    This is primarily a high-confidence ranking diagnostic. It is not a direct
    deployable rule unless converted into a threshold chosen from past data.
    """
    if probability_column not in predictions.columns:
        raise ValueError(f"Missing probability column: {probability_column}")
    pct = float(top_percentile)
    if pct <= 0 or pct > 100:
        raise ValueError("top_percentile must be in (0, 100].")
    scores = pd.to_numeric(predictions[probability_column], errors="coerce")
    out = pd.Series(False, index=predictions.index)
    fold_col = "fold" if "fold" in predictions.columns else None
    groups = predictions.groupby(fold_col).groups.items() if fold_col else [("ALL", predictions.index)]
    for _, idx in groups:
        idx = list(idx)
        n = len(idx)
        if n == 0:
            continue
        k = max(1, int(np.ceil(n * pct / 100.0)))
        top_idx = scores.loc[idx].sort_values(ascending=False).head(k).index
        out.loc[top_idx] = True
    return out


def threshold_metrics_for_selection(y_true: pd.Series, selected: pd.Series) -> dict[str, float]:
    y = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    s = selected.fillna(False).astype(bool)
    tp = int(((y == 1) & s).sum())
    fp = int(((y == 0) & s).sum())
    fn = int(((y == 1) & ~s).sum())
    tn = int(((y == 0) & ~s).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    signal_rate = float(s.mean()) if len(s) else float("nan")
    return {"tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn), "precision": precision, "recall": recall, "signal_rate": signal_rate}


def rolling_oof_target_precision_mask(
    predictions: pd.DataFrame,
    *,
    probability_column: str,
    thresholds: list[float],
    target_precision: float,
    min_past_trades: int,
    fallback_threshold: float | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Choose threshold from prior OOF folds, then apply to the next fold.

    This is a practical approximation of selecting thresholds only from past
    out-of-sample evidence. For fold 0 there is no past OOF evidence, so the
    fallback threshold is used if provided; otherwise no trades are selected.
    """
    if "fold" not in predictions.columns:
        raise ValueError("rolling_oof_target_precision_mask requires a fold column.")
    if probability_column not in predictions.columns:
        raise ValueError(f"Missing probability column: {probability_column}")
    thresholds = sorted([float(x) for x in thresholds])
    selected = pd.Series(False, index=predictions.index)
    rows = []
    folds = list(pd.Series(predictions["fold"]).drop_duplicates())
    past_idx: list[int] = []
    for fold in folds:
        cur_idx = predictions.index[predictions["fold"].eq(fold)].tolist()
        chosen = fallback_threshold if not past_idx else None
        reason = "fallback" if not past_idx and fallback_threshold is not None else "no_past_data"
        if past_idx:
            past = predictions.loc[past_idx]
            candidates = []
            for thr in thresholds:
                mask = fixed_threshold_mask(past, probability_column=probability_column, threshold=thr)
                m = threshold_metrics_for_selection(past["y_true"], mask)
                trades = int(m["tp"] + m["fp"])
                if trades >= int(min_past_trades) and np.isfinite(m["precision"]) and m["precision"] >= float(target_precision):
                    candidates.append((thr, m, trades))
            if candidates:
                # Use the least restrictive threshold that satisfies the target;
                # this preserves more trade opportunities while meeting precision.
                chosen, chosen_metrics, _ = sorted(candidates, key=lambda x: x[0])[0]
                reason = "target_precision_met"
            elif fallback_threshold is not None:
                chosen = fallback_threshold
                reason = "fallback_no_threshold_met_target"
            else:
                reason = "no_threshold_met_target"
        if chosen is not None:
            cur = predictions.loc[cur_idx]
            cur_mask = fixed_threshold_mask(cur, probability_column=probability_column, threshold=float(chosen))
            selected.loc[cur.index] = cur_mask.to_numpy()
        rows.append({"fold": fold, "chosen_threshold": float(chosen) if chosen is not None else np.nan, "reason": reason, "past_rows": float(len(past_idx))})
        past_idx.extend(cur_idx)
    return selected, pd.DataFrame(rows)
