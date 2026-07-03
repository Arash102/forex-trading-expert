from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CPCVFold:
    fold_id: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_groups: tuple[int, ...]


def _contiguous_group_ids(n_rows: int, n_groups: int) -> np.ndarray:
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if n_rows < n_groups:
        raise ValueError("n_rows must be >= n_groups")
    edges = np.linspace(0, n_rows, n_groups + 1, dtype=int)
    group_ids = np.empty(n_rows, dtype=int)
    for group_id in range(n_groups):
        group_ids[edges[group_id] : edges[group_id + 1]] = group_id
    return group_ids


def _event_start_end(metadata: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "entry_date" in metadata.columns:
        start = pd.to_datetime(metadata["entry_date"], errors="coerce")
    elif "date" in metadata.columns:
        start = pd.to_datetime(metadata["date"], errors="coerce")
    else:
        start = pd.Series(pd.RangeIndex(len(metadata)), index=metadata.index)

    if "exit_date" in metadata.columns:
        end = pd.to_datetime(metadata["exit_date"], errors="coerce")
    elif "date" in metadata.columns:
        end = pd.to_datetime(metadata["date"], errors="coerce")
    else:
        end = pd.Series(pd.RangeIndex(len(metadata)), index=metadata.index)
    end = end.fillna(start)
    start = start.fillna(end)
    return start, end


def _apply_purge(
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    metadata: pd.DataFrame,
) -> np.ndarray:
    """Purge train events whose [entry, exit] interval overlaps test intervals."""
    start, end = _event_start_end(metadata)
    test_start = start[test_mask]
    test_end = end[test_mask]
    if len(test_start) == 0:
        return train_mask

    purged = train_mask.copy()
    # Combine each contiguous test group into a time interval to keep this fast.
    test_intervals = pd.DataFrame({"start": test_start, "end": test_end}).dropna()
    if test_intervals.empty:
        return purged
    # The split groups are contiguous in row space, but combinations may include
    # multiple separated blocks. Coalesce only adjacent/overlapping time intervals.
    test_intervals = test_intervals.sort_values("start")
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for row in test_intervals.itertuples(index=False):
        s = row.start
        e = row.end
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    for s, e in merged:
        overlap = (start <= e) & (end >= s)
        purged &= ~overlap.to_numpy(dtype=bool)
    return purged


def _apply_embargo(
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    embargo_bars: int,
) -> np.ndarray:
    if embargo_bars <= 0:
        return train_mask
    out = train_mask.copy()
    test_idx = np.flatnonzero(test_mask)
    if len(test_idx) == 0:
        return out
    # Embargo bars immediately after every contiguous test block.
    breaks = np.where(np.diff(test_idx) > 1)[0]
    block_starts = np.r_[test_idx[0], test_idx[breaks + 1]]
    block_ends = np.r_[test_idx[breaks], test_idx[-1]]
    n = len(train_mask)
    for end in block_ends:
        embargo_start = int(end) + 1
        embargo_end = min(n, embargo_start + int(embargo_bars))
        out[embargo_start:embargo_end] = False
    return out


def make_cpcv_splits(
    metadata: pd.DataFrame,
    *,
    n_groups: int,
    n_test_groups: int,
    embargo_bars: int = 0,
    purge: bool = True,
    max_splits: int | None = None,
) -> list[CPCVFold]:
    """Build combinatorial purged CV folds in the spirit of Lopez de Prado.

    Data must be sorted chronologically. The implementation returns all
    combinations of ``n_test_groups`` out of ``n_groups`` unless ``max_splits`` is
    specified. Train events overlapping test event intervals are purged, and an
    optional row-based embargo is applied after each contiguous test block.
    """
    n = len(metadata)
    if n_test_groups <= 0 or n_test_groups >= n_groups:
        raise ValueError("n_test_groups must be in [1, n_groups-1].")
    group_ids = _contiguous_group_ids(n, n_groups)
    folds: list[CPCVFold] = []
    for combo_no, combo in enumerate(combinations(range(n_groups), n_test_groups)):
        if max_splits is not None and combo_no >= int(max_splits):
            break
        test_mask = np.isin(group_ids, np.asarray(combo, dtype=int))
        train_mask = ~test_mask
        if purge:
            train_mask = _apply_purge(train_mask, test_mask, metadata)
        train_mask = _apply_embargo(train_mask, test_mask, int(embargo_bars))
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        folds.append(
            CPCVFold(
                fold_id="cpcv_" + "_".join(str(x) for x in combo),
                train_idx=train_idx.astype(int),
                test_idx=test_idx.astype(int),
                test_groups=tuple(int(x) for x in combo),
            )
        )
    return folds
