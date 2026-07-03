from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def candidate_mask_for_job(x: pd.DataFrame, *, symbol: str, side: str, config: Mapping[str, Any]) -> pd.Series:
    """Return candidate rows for optional candidate-based/meta-label training.

    This is intentionally disabled by default in ``ml_config``. It provides a
    clean hook for the next research step: primary candidate generation first,
    XGBoost as a meta-label / confirmation model second.
    """
    cfg = config.get("candidate_filter", {})
    mask = pd.Series(True, index=x.index)
    if not bool(cfg.get("enabled", False)):
        return mask

    session_ids = cfg.get("session_block_ids", None)
    if session_ids is not None and "session_block_id" in x.columns:
        mask &= x["session_block_id"].isin([int(v) for v in session_ids])

    max_spread_by_symbol = cfg.get("max_spread_pips_by_symbol", {})
    max_spread = max_spread_by_symbol.get(symbol, None)
    if max_spread is not None and "spread_pips" in x.columns:
        mask &= pd.to_numeric(x["spread_pips"], errors="coerce") <= float(max_spread)

    preset = str(cfg.get("preset", "session_tradeable_v1"))
    if preset == "directional_context_v1":
        dc = cfg.get("directional_context_v1", {})
        if "gmma_distance" in x.columns:
            gmma = pd.to_numeric(x["gmma_distance"], errors="coerce")
            if side == "long":
                mask &= gmma >= float(dc.get("long_min_gmma_distance", -20))
            elif side == "short":
                mask &= gmma <= float(dc.get("short_max_gmma_distance", 20))
        if "atr_regime" in x.columns:
            atr = pd.to_numeric(x["atr_regime"], errors="coerce")
            mask &= atr >= float(dc.get("min_atr_regime", 1))
            mask &= atr <= float(dc.get("max_atr_regime", 4))

    return mask.fillna(False).astype(bool)


def apply_candidate_filter(
    x: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    *,
    symbol: str,
    side: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, float]]:
    mask = candidate_mask_for_job(x, symbol=symbol, side=side, config=config)
    before = len(x)
    x2 = x.loc[mask].reset_index(drop=True)
    y2 = y.loc[mask].reset_index(drop=True)
    m2 = metadata.loc[mask].reset_index(drop=True)
    after = len(x2)
    stats = {
        "candidate_filter_enabled": float(bool(config.get("candidate_filter", {}).get("enabled", False))),
        "candidate_rows_before": float(before),
        "candidate_rows_after": float(after),
        "candidate_keep_ratio": float(after / before) if before else np.nan,
        "candidate_positive_rate": float((y2 == 1).mean()) if after else np.nan,
    }
    return x2, y2, m2, stats
