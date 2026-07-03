from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValidationFold:
    fold_id: str
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_walk_forward_splits(
    n_rows: int,
    *,
    train_window_bars: int,
    test_window_bars: int,
    step_bars: int | None = None,
    expanding: bool = False,
    min_train_bars: int | None = None,
    purge_bars: int = 0,
) -> list[ValidationFold]:
    """Build causal walk-forward splits over row order.

    The row order must already be chronological. Test windows are always strictly
    after the train window. ``purge_bars`` removes the last rows before each test
    window from the train set. This is useful for Triple Barrier labels whose
    event horizon extends beyond the feature row.
    """
    n = int(n_rows)
    train_window = int(train_window_bars)
    test_window = int(test_window_bars)
    step = int(step_bars or test_window_bars)
    min_train = int(min_train_bars or train_window)
    purge = max(0, int(purge_bars))
    if n <= 0:
        return []
    if train_window <= 0 or test_window <= 0 or step <= 0:
        raise ValueError("train_window_bars, test_window_bars, and step_bars must be positive.")
    if min_train <= 0:
        raise ValueError("min_train_bars must be positive.")

    folds: list[ValidationFold] = []
    test_start = train_window
    fold_no = 0
    while test_start < n:
        train_start = 0 if expanding else max(0, test_start - train_window)
        train_end = max(train_start, test_start - purge)
        test_end = min(n, test_start + test_window)
        if train_end - train_start >= min_train and test_end > test_start:
            folds.append(
                ValidationFold(
                    fold_id=f"wf_{fold_no:03d}",
                    train_idx=np.arange(train_start, train_end, dtype=int),
                    test_idx=np.arange(test_start, test_end, dtype=int),
                )
            )
            fold_no += 1
        if test_end >= n:
            break
        test_start += step
    return folds
