from __future__ import annotations

import pandas as pd


def summarize_trades(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {"trades": 0, "win_rate": 0.0, "net_pips": 0.0, "profit_factor": 0.0, "payoff": 0.0}
    pnl = trades["pnl_pips"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "net_pips": float(pnl.sum()),
        "profit_factor": gross_profit / gross_loss if gross_loss else float("inf"),
        "payoff": float(wins.mean() / (-losses.mean())) if len(wins) and len(losses) else 0.0,
    }
