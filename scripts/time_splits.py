"""Centralized utilities for managing chronological train/validation/test splits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
SPLITS_DIR = MODELS_DIR / "splits"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class TimeSplit:
    """Represents chronological train/validation/test windows."""

    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "val_start": self.val_start,
            "val_end": self.val_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "TimeSplit":
        return cls(
            train_start=data["train_start"],
            train_end=data["train_end"],
            val_start=data["val_start"],
            val_end=data["val_end"],
            test_start=data["test_start"],
            test_end=data["test_end"],
        )

    def _parse(self, value: str) -> pd.Timestamp:
        return pd.to_datetime(value)

    def _section_bounds(self, section: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        section = section.lower()
        mapping = {
            "train": (self._parse(self.train_start), self._parse(self.train_end)),
            "validation": (self._parse(self.val_start), self._parse(self.val_end)),
            "val": (self._parse(self.val_start), self._parse(self.val_end)),
            "test": (self._parse(self.test_start), self._parse(self.test_end)),
        }
        if section not in mapping:
            raise ValueError(f"Unknown section '{section}'.")
        return mapping[section]

    def select(self, df: pd.DataFrame, section: str, date_col: str = "date") -> pd.DataFrame:
        """Return rows belonging to the selected section."""
        if df.empty:
            return df.copy()
        start, end = self._section_bounds(section)
        dates = pd.to_datetime(df[date_col])
        mask = (dates >= start) & (dates <= end)
        return df.loc[mask].copy()

    def to_frames(self, df: pd.DataFrame, date_col: str = "date") -> Dict[str, pd.DataFrame]:
        """Split the given DataFrame into train/validation/test frames."""
        return {
            "train": self.select(df, "train", date_col=date_col),
            "validation": self.select(df, "validation", date_col=date_col),
            "test": self.select(df, "test", date_col=date_col),
        }


def _split_path(symbol: str, frequency: str) -> Path:
    return SPLITS_DIR / frequency / f"{symbol.upper()}.json"


def save_time_split(symbol: str, frequency: str, split: TimeSplit) -> Path:
    path = _split_path(symbol, frequency)
    _ensure_dir(path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(split.to_dict(), fh, indent=2)
    return path


def load_time_split(symbol: str, frequency: str) -> Optional[TimeSplit]:
    path = _split_path(symbol, frequency)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return TimeSplit.from_dict(data)


def compute_time_split(
    dates: Sequence[pd.Timestamp] | pd.Series,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> TimeSplit:
    """Compute chronological split boundaries from ratios."""
    series = pd.Series(pd.to_datetime(dates)).sort_values().reset_index(drop=True)
    total = len(series)
    if total == 0:
        raise ValueError("Cannot compute time split on empty date series.")
    ratios_sum = train_ratio + val_ratio + test_ratio
    train_ratio = train_ratio / ratios_sum
    val_ratio = val_ratio / ratios_sum
    test_ratio = test_ratio / ratios_sum
    train_count = max(1, int(total * train_ratio))
    val_count = max(1, int(total * val_ratio))
    remainder = total - train_count - val_count
    if remainder <= 0:
        remainder = 1
        if val_count > 1:
            val_count -= 1
        else:
            train_count = max(1, train_count - 1)
    test_count = remainder
    train_end_idx = train_count - 1
    val_start_idx = train_end_idx + 1
    val_end_idx = val_start_idx + val_count - 1
    test_start_idx = val_end_idx + 1
    test_end_idx = total - 1
    return TimeSplit(
        train_start=str(series.iloc[0].date()),
        train_end=str(series.iloc[min(train_end_idx, total - 1)].date()),
        val_start=str(series.iloc[min(val_start_idx, total - 1)].date()),
        val_end=str(series.iloc[min(val_end_idx, total - 1)].date()),
        test_start=str(series.iloc[min(test_start_idx, total - 1)].date()),
        test_end=str(series.iloc[test_end_idx].date()),
    )


def get_or_create_time_split(
    symbol: str,
    frequency: str,
    dates: Sequence[pd.Timestamp] | pd.Series,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> TimeSplit:
    split = load_time_split(symbol, frequency)
    if split is not None:
        return split
    split = compute_time_split(dates, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    save_time_split(symbol, frequency, split)
    return split


def generate_walk_forward_splits(
    dates: Sequence[pd.Timestamp] | pd.Series,
    *,
    train_window: int,
    val_window: int,
    test_window: int,
    step: Optional[int] = None,
    max_splits: Optional[int] = None,
) -> List[TimeSplit]:
    """Generate sequential walk-forward splits based on window sizes (in rows)."""
    if train_window <= 0 or val_window <= 0 or test_window <= 0:
        raise ValueError("Window sizes must be positive integers.")
    series = pd.Series(pd.to_datetime(dates)).sort_values().reset_index(drop=True)
    total = len(series)
    splits: List[TimeSplit] = []
    start = 0
    step = step or test_window
    while True:
        train_end_idx = start + train_window - 1
        val_start_idx = train_end_idx + 1
        val_end_idx = val_start_idx + val_window - 1
        test_start_idx = val_end_idx + 1
        test_end_idx = test_start_idx + test_window - 1
        if test_end_idx >= total:
            break
        split = TimeSplit(
            train_start=str(series.iloc[start].date()),
            train_end=str(series.iloc[train_end_idx].date()),
            val_start=str(series.iloc[val_start_idx].date()),
            val_end=str(series.iloc[val_end_idx].date()),
            test_start=str(series.iloc[test_start_idx].date()),
            test_end=str(series.iloc[test_end_idx].date()),
        )
        splits.append(split)
        if max_splits and len(splits) >= max_splits:
            break
        start += step
    return splits
