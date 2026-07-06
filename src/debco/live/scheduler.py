from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Iterable, Mapping
from numbers import Real


TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
}


def timeframe_seconds(timeframe: str) -> int:
    tf = str(timeframe).upper()
    if tf not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return TIMEFRAME_SECONDS[tf]


def utc_dt(value: Any) -> datetime:
    # MT5/numpy rows often expose scalar timestamps as numpy scalar objects
    # (for example numpy.int64). Convert them to plain Python scalars without
    # adding a hard dependency on numpy in the live router.
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, Real) and not isinstance(value, bool):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        s = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    raise TypeError(f"Cannot convert to UTC datetime: {value!r}")


def iso_utc(dt: datetime) -> str:
    return utc_dt(dt).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class BarEvent:
    symbol: str
    timeframe: str
    current_bar_time: datetime
    closed_bar_time: datetime
    previous_current_bar_time: datetime | None = None

    @property
    def current_bar_time_utc(self) -> str:
        return iso_utc(self.current_bar_time)

    @property
    def closed_bar_time_utc(self) -> str:
        return iso_utc(self.closed_bar_time)


def extract_bar_time(row: Any) -> datetime:
    if isinstance(row, Mapping):
        return utc_dt(row["time"])

    # MetaTrader5.copy_rates_from_pos returns a numpy structured array.
    # Iterating it yields numpy.void rows with dtype names such as "time",
    # "open", "high"...  Accessing row["time"] is valid, but the
    # value is usually numpy.int64, so utc_dt must support numpy scalars too.
    dtype = getattr(row, "dtype", None)
    names = getattr(dtype, "names", None)
    if names and "time" in names:
        return utc_dt(row["time"])

    if hasattr(row, "__getitem__"):
        try:
            return utc_dt(row["time"])
        except Exception:
            pass
        # Fallback for tuple-like MT5 rows where the first field is epoch time.
        try:
            return utc_dt(row[0])
        except Exception:
            pass
    raise TypeError(f"Cannot extract MT5 bar time from row: {row!r}")


def detect_new_bar(
    *,
    symbol: str,
    timeframe: str,
    rates: Iterable[Any],
    last_seen_current_bar_time: datetime | None,
) -> BarEvent | None:
    rows = list(rates)
    if len(rows) < 2:
        return None
    # MT5 copy_rates_from_pos(..., 0, N) normally returns current forming bar first.
    # Some mocks/dataframes may be chronological. We handle both by sorting unique times.
    times = sorted({extract_bar_time(r) for r in rows})
    if len(times) < 2:
        return None
    current_bar_time = times[-1]
    closed_bar_time = times[-2]
    if last_seen_current_bar_time is None or current_bar_time > utc_dt(last_seen_current_bar_time):
        return BarEvent(
            symbol=symbol,
            timeframe=timeframe,
            current_bar_time=current_bar_time,
            closed_bar_time=closed_bar_time,
            previous_current_bar_time=last_seen_current_bar_time,
        )
    return None


def next_expected_bar_time(last_current_bar_time: datetime, timeframe: str) -> datetime:
    return utc_dt(last_current_bar_time) + timedelta(seconds=timeframe_seconds(timeframe))


def in_fast_poll_window(
    *,
    now: datetime,
    next_bar_time: datetime,
    pre_window_seconds: float,
    post_window_seconds: float,
) -> bool:
    now_u = utc_dt(now)
    next_u = utc_dt(next_bar_time)
    return (next_u - timedelta(seconds=float(pre_window_seconds))) <= now_u <= (next_u + timedelta(seconds=float(post_window_seconds)))


def choose_sleep_seconds(
    *,
    now: datetime,
    next_bar_time: datetime | None,
    normal_poll_seconds: float,
    fast_poll_seconds: float,
    pre_bar_fast_window_seconds: float,
    post_bar_fast_window_seconds: float,
) -> float:
    if next_bar_time is None:
        return float(normal_poll_seconds)
    if in_fast_poll_window(
        now=now,
        next_bar_time=next_bar_time,
        pre_window_seconds=pre_bar_fast_window_seconds,
        post_window_seconds=post_bar_fast_window_seconds,
    ):
        return float(fast_poll_seconds)
    return float(normal_poll_seconds)
