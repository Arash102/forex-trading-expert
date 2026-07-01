from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def walk_forward_splits(n_rows: int, train_bars: int, test_bars: int, step_bars: int, purge_bars: int = 0) -> list[Split]:
    splits: list[Split] = []
    fold = 0
    train_start = 0
    while True:
        train_end = train_start + train_bars
        test_start = train_end + purge_bars
        test_end = test_start + test_bars
        if test_end > n_rows:
            break
        splits.append(Split(fold, train_start, train_end, test_start, test_end))
        fold += 1
        train_start += step_bars
    return splits
