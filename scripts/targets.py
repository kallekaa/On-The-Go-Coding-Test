"""Utilities for computing and managing forward-return targets."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, MutableMapping, Sequence

import pandas as pd


DAILY_TARGET_STEPS: "OrderedDict[str, int]" = OrderedDict(
    [
        ("target_1d", 1),
        ("target_2d", 2),
        ("target_3d", 3),
        ("target_5d", 5),
        ("target_1w", 5),
    ]
)

WEEKLY_TARGET_STEPS: "OrderedDict[str, int]" = OrderedDict(
    [
        ("target_1d", 1),
        ("target_2d", 2),
        ("target_3d", 3),
        ("target_5d", 5),
        ("target_1w", 1),
    ]
)


def get_horizon_steps(frequency: str) -> "OrderedDict[str, int]":
    """Return ordered mapping of target column names to look-ahead steps for a frequency."""
    frequency = frequency.lower()
    if frequency == "weekly":
        return WEEKLY_TARGET_STEPS.copy()
    return DAILY_TARGET_STEPS.copy()


def ordered_target_names(frequency: str | None = None) -> List[str]:
    """List target column names for a frequency, or all unique names if None."""
    if frequency:
        return list(get_horizon_steps(frequency).keys())
    names: List[str] = []
    for mapping in (DAILY_TARGET_STEPS, WEEKLY_TARGET_STEPS):
        for name in mapping:
            if name not in names:
                names.append(name)
    return names


def compute_forward_returns(close_series: pd.Series, frequency: str) -> pd.DataFrame:
    """Compute forward percentage returns for all configured horizons."""
    steps = get_horizon_steps(frequency)
    result = {}
    for column, step in steps.items():
        shifted = close_series.shift(-step)
        result[column] = shifted.div(close_series).sub(1.0)
    return pd.DataFrame(result, index=close_series.index)


def target_columns_for_window(
    frequency: str,
    start_offset: int,
    end_offset: int,
) -> List[str]:
    """Return target column names whose look-ahead steps fall in the given window."""
    steps = get_horizon_steps(frequency)
    return [col for col, step in steps.items() if start_offset <= step <= end_offset]
