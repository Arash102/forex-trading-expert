from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

_TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


@dataclass(frozen=True)
class MT5ConnectionConfig:
    login: int | None = None
    password: str | None = None
    server: str | None = None
    path: str | None = None


class MT5DataLoader:
    """Thin MT5 data loader.

    This module only fetches raw broker candles. It must not calculate trading
    features or signals.
    """

    def __init__(self, connection: MT5ConnectionConfig | None = None) -> None:
        self.connection = connection or MT5ConnectionConfig()
        self.mt5 = None

    def connect(self) -> None:
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError as exc:
            raise RuntimeError("MetaTrader5 package is not installed. Run: pip install MetaTrader5") from exc

        kwargs: dict[str, Any] = {}
        if self.connection.path:
            kwargs["path"] = self.connection.path
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

        if self.connection.login and self.connection.password and self.connection.server:
            if not mt5.login(self.connection.login, password=self.connection.password, server=self.connection.server):
                raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
        self.mt5 = mt5

    def shutdown(self) -> None:
        if self.mt5 is not None:
            self.mt5.shutdown()

    def _tf(self, timeframe: str) -> int:
        if self.mt5 is None:
            raise RuntimeError("MT5 is not connected")
        key = timeframe.upper()
        if key not in _TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return getattr(self.mt5, _TIMEFRAME_MAP[key])

    def fetch_bars(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        if self.mt5 is None:
            raise RuntimeError("MT5 is not connected")
        if not self.mt5.symbol_select(symbol, True):
            raise RuntimeError(f"MT5 cannot select symbol: {symbol}. last_error={self.mt5.last_error()}")

        rates = self.mt5.copy_rates_from_pos(symbol, self._tf(timeframe), 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates returned for {symbol}. last_error={self.mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
        df = df.rename(columns={"tick_volume": "tick_volume", "real_volume": "real_volume"})
        cols = ["date", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
        return df[cols].sort_values("date").reset_index(drop=True)


def fetch_many_from_mt5(symbol_map: dict[str, str], timeframe: str, bars: int) -> dict[str, pd.DataFrame]:
    loader = MT5DataLoader()
    loader.connect()
    try:
        return {name: loader.fetch_bars(broker_symbol, timeframe, bars) for name, broker_symbol in symbol_map.items()}
    finally:
        loader.shutdown()
