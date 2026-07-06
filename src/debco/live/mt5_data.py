from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MT5_TIMEFRAMES = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
}


class MT5NotAvailableError(RuntimeError):
    pass


@dataclass
class MT5ConnectionConfig:
    terminal_path: str | None = None
    login: int | None = None
    password: str | None = None
    server: str | None = None
    timeout_seconds: int = 30


class MT5DataClient:
    def __init__(self, config: MT5ConnectionConfig | None = None):
        self.config = config or MT5ConnectionConfig()
        self.mt5 = None

    def import_mt5(self):
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on user machine
            raise MT5NotAvailableError("MetaTrader5 Python package is not available in this environment") from exc
        self.mt5 = mt5
        return mt5

    def initialize(self) -> None:  # pragma: no cover - requires MT5 terminal
        mt5 = self.import_mt5()
        kwargs: dict[str, Any] = {}
        if self.config.terminal_path:
            kwargs["path"] = self.config.terminal_path
        if self.config.login is not None:
            kwargs["login"] = int(self.config.login)
        if self.config.password:
            kwargs["password"] = self.config.password
        if self.config.server:
            kwargs["server"] = self.config.server
        ok = mt5.initialize(**kwargs)
        if not ok:
            raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")

    def shutdown(self) -> None:  # pragma: no cover - requires MT5 terminal
        if self.mt5 is not None:
            self.mt5.shutdown()

    def timeframe_constant(self, timeframe: str):  # pragma: no cover - simple with MT5
        mt5 = self.mt5 or self.import_mt5()
        name = MT5_TIMEFRAMES.get(str(timeframe).upper())
        if not name:
            raise ValueError(f"Unsupported MT5 timeframe: {timeframe}")
        return getattr(mt5, name)

    def latest_rates(self, symbol: str, timeframe: str, count: int = 500):  # pragma: no cover - requires MT5 terminal
        mt5 = self.mt5 or self.import_mt5()
        tf = self.timeframe_constant(timeframe)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(count))
        if rates is None:
            raise RuntimeError(f"copy_rates_from_pos failed for {symbol} {timeframe}: {mt5.last_error()}")
        return rates
