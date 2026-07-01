from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    side: str,
    take_profit_pips: float,
    stop_loss_pips: float,
    horizon_bars: int,
    pip_size: float,
    same_bar_policy: str = "sl_first",
) -> pd.DataFrame:
    """Triple Barrier labeling after Lopez de Prado style event horizon.

    Labels:
    +1: take profit first
    -1: stop loss first
     0: vertical barrier reached without TP/SL

    The label is computed from bar i entry at next bar open by default.
    This function only creates labels; trading simulation is separate.
    """
    if side not in {"long", "short"}:
        raise ValueError("side must be 'long' or 'short'")
    if same_bar_policy not in {"sl_first", "tp_first"}:
        raise ValueError("same_bar_policy must be sl_first or tp_first")

    out = df.copy().sort_values("date").reset_index(drop=True)
    labels = np.full(len(out), np.nan)
    hit_time = [pd.NaT] * len(out)
    hit_reason = [None] * len(out)

    for i in range(len(out) - 1):
        entry_idx = i + 1
        end_idx = min(entry_idx + horizon_bars, len(out) - 1)
        entry = float(out.loc[entry_idx, "open"])
        if side == "long":
            tp = entry + take_profit_pips * pip_size
            sl = entry - stop_loss_pips * pip_size
        else:
            tp = entry - take_profit_pips * pip_size
            sl = entry + stop_loss_pips * pip_size

        label = 0
        reason = "vertical"
        htime = out.loc[end_idx, "date"]
        for j in range(entry_idx, end_idx + 1):
            high = float(out.loc[j, "high"])
            low = float(out.loc[j, "low"])
            if side == "long":
                tp_hit = high >= tp
                sl_hit = low <= sl
            else:
                tp_hit = low <= tp
                sl_hit = high >= sl

            if tp_hit and sl_hit:
                if same_bar_policy == "sl_first":
                    label, reason = -1, "sl_same_bar"
                else:
                    label, reason = 1, "tp_same_bar"
                htime = out.loc[j, "date"]
                break
            if tp_hit:
                label, reason = 1, "tp"
                htime = out.loc[j, "date"]
                break
            if sl_hit:
                label, reason = -1, "sl"
                htime = out.loc[j, "date"]
                break

        labels[i] = label
        hit_time[i] = htime
        hit_reason[i] = reason

    out["tb_label"] = labels
    out["tb_hit_time"] = hit_time
    out["tb_hit_reason"] = hit_reason
    out["tb_side"] = side
    out["tb_tp_pips"] = take_profit_pips
    out["tb_sl_pips"] = stop_loss_pips
    out["tb_horizon_bars"] = horizon_bars
    return out
