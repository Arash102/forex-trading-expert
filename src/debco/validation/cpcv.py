from __future__ import annotations

from itertools import combinations
from dataclasses import dataclass


@dataclass(frozen=True)
class CPCVSplit:
    fold: int
    train_groups: list[int]
    test_groups: list[int]


def cpcv_splits(n_groups: int, k_test_groups: int) -> list[CPCVSplit]:
    """Combinatorial Purged Cross-Validation group combinations.

    Purge/embargo are applied later when mapping groups to row intervals.
    """
    all_groups = list(range(n_groups))
    out: list[CPCVSplit] = []
    for fold, test_tuple in enumerate(combinations(all_groups, k_test_groups)):
        test = list(test_tuple)
        train = [g for g in all_groups if g not in test]
        out.append(CPCVSplit(fold, train, test))
    return out
