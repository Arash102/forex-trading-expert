from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def load_csv(path: str | Path, parse_dates: tuple[str, ...] = ("date",)) -> pd.DataFrame:
    p = Path(path)
    kwargs: dict[str, Any] = {}
    if parse_dates:
        kwargs["parse_dates"] = list(parse_dates)
    return pd.read_csv(p, **kwargs)
