"""Regression training pipeline with time-series aware CV."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
import sys
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.diagnostic_outputs import save_regression_summary
from scripts.targets import ordered_target_names
from scripts.time_splits import (
    TimeSplit,
    generate_walk_forward_splits,
    get_or_create_time_split,
)

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REGRESSION_MODEL_DIR = MODELS_DIR / "regression"
REGRESSION_METRICS_DIR = MODELS_DIR / "metrics" / "regression"
DEFAULT_DB_PATH = DATA_DIR / "stocks.db"

FEATURE_TABLES = {"daily": "features_daily", "weekly": "features_weekly"}
METADATA_COLUMNS = ["symbol", "date"]
TARGET_HORIZONS = ordered_target_names()
MODEL_TYPES = {"lasso": Lasso, "ridge": Ridge}


@dataclass
class FeatureDataset:
    """Container describing features, targets, and metadata columns."""

    data: pd.DataFrame
    feature_cols: List[str]
    target_cols: List[str]
    metadata_cols: List[str]


@dataclass
class SplitData:
    """Chronological split of the dataset."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass
class ModelTrainingResult:
    """Summary of a trained regression model."""

    horizon: str
    best_alpha: float | None
    metrics: Dict[str, float | None]
    model_path: Path | None
    metrics_path: Path | None


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Establish and return a SQLite connection."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def load_feature_data(symbol: str, frequency: str, conn: sqlite3.Connection) -> FeatureDataset:
    """Load engineered features for a symbol/frequency pair sorted chronologically."""
    frequency = frequency.lower()
    table = FEATURE_TABLES[frequency]
    query = f"""
        SELECT *
        FROM {table}
        WHERE symbol = ?
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(symbol.upper(),))
    if df.empty:
        return FeatureDataset(df, [], [], METADATA_COLUMNS.copy())

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    metadata_cols = [col for col in METADATA_COLUMNS if col in df.columns]
    target_cols = [col for col in df.columns if col.startswith("target_")]
    excluded = set(metadata_cols + target_cols)
    feature_cols = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    return FeatureDataset(data=df, feature_cols=feature_cols, target_cols=target_cols, metadata_cols=metadata_cols)


def split_dataset_by_timesplit(dataset: pd.DataFrame, split: TimeSplit) -> SplitData:
    frames = split.to_frames(dataset, date_col="date")
    return SplitData(train=frames["train"], validation=frames["validation"], test=frames["test"])


def run_walk_forward_analysis(
    symbol: str,
    frequency: str,
    dataset: pd.DataFrame,
    feature_cols: Sequence[str],
    wf_cfg: Dict[str, Any],
    model_type: str,
    alpha_grid: Sequence[float],
    random_seed: int,
) -> None:
    """Evaluate regression stability via walk-forward splits."""
    if not wf_cfg:
        return
    splits = generate_walk_forward_splits(
        dataset["date"],
        train_window=wf_cfg.get("train_window", 200),
        val_window=wf_cfg.get("val_window", 50),
        test_window=wf_cfg.get("test_window", 50),
        step=wf_cfg.get("step"),
        max_splits=wf_cfg.get("max_splits"),
    )
    if not splits:
        message = "Walk-forward configuration produced no valid splits."
        logger.warning(message)
        print(message)
        return
    for horizon in TARGET_HORIZONS:
        if horizon not in dataset.columns:
            continue
        horizon_results = []
        for idx, wf_split in enumerate(splits):
            split_data = split_dataset_by_timesplit(dataset, wf_split)
            result = train_single_horizon(
                symbol=symbol,
                frequency=frequency,
                split_data=split_data,
                feature_cols=feature_cols,
                target_col=horizon,
                model_type=model_type,
                alpha_grid=alpha_grid,
                persist=False,
                random_seed=random_seed,
            )
            if result:
                horizon_results.append({"split_index": idx, "split": wf_split.to_dict(), "metrics": result.metrics})
        if horizon_results:
            path = REGRESSION_METRICS_DIR / frequency / f"{symbol.upper()}_{horizon}_walk_forward.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                json.dump(horizon_results, fh, indent=2)


def rolling_expanding_window_indices(
    n_samples: int,
    min_train_size: int,
    val_window: int,
    step_size: int = 0,
    max_splits: int | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window folds for time-series cross-validation."""
    if n_samples < min_train_size + val_window:
        return []
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    train_end = min_train_size
    while True:
        val_start = train_end
        val_end = val_start + val_window
        if val_end > n_samples:
            break
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        splits.append((train_idx, val_idx))
        if max_splits is not None and len(splits) >= max_splits:
            break
        train_end = val_end + step_size
    return splits


def _prepare_matrix(df: pd.DataFrame, feature_cols: Sequence[str], target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and aligned target vector, dropping NaNs."""
    filtered = df.replace([np.inf, -np.inf], np.nan)
    subset = filtered[feature_cols + [target_col]].copy()
    subset[feature_cols] = subset[feature_cols].ffill().fillna(0.0)
    subset = subset.dropna(subset=[target_col])
    features = subset[feature_cols].reset_index(drop=True)
    target = subset[target_col].reset_index(drop=True)
    return features, target


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, and R^2 metrics."""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def save_model(model, path: Path) -> None:
    """Persist a trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)


def load_model(symbol: str, frequency: str, model_type: str, horizon: str) -> object:
    """Load a persisted regression model."""
    path = build_model_path(symbol, frequency, model_type, horizon)
    return load(path)


def build_model_path(symbol: str, frequency: str, model_type: str, horizon: str) -> Path:
    """Generate a consistent model path."""
    safe_horizon = horizon.replace("target_", "")
    filename = f"{symbol.upper()}_{frequency}_{model_type}_{safe_horizon}.joblib"
    return REGRESSION_MODEL_DIR / frequency / filename


def build_metrics_path(symbol: str, frequency: str, model_type: str, horizon: str) -> Path:
    """Generate a consistent metrics path."""
    safe_horizon = horizon.replace("target_", "")
    filename = f"{symbol.upper()}_{frequency}_{model_type}_{safe_horizon}.json"
    return REGRESSION_METRICS_DIR / frequency / filename


def store_metrics(metrics: Dict[str, float | None], path: Path) -> None:
    """Write evaluation metrics to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def time_series_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    alpha_grid: Sequence[float],
    min_train_size: int,
    val_window: int,
    max_splits: int = 5,
) -> float | None:
    """Return the alpha value with the best average RMSE."""
    folds = rolling_expanding_window_indices(
        len(X), min_train_size=min_train_size, val_window=val_window, max_splits=max_splits
    )
    if not folds:
        return alpha_grid[0] if alpha_grid else None
    model_cls = MODEL_TYPES[model_type]
    best_alpha = None
    best_score = np.inf
    for alpha in alpha_grid:
        fold_scores: List[float] = []
        for train_idx, val_idx in folds:
            model = model_cls(alpha=alpha, max_iter=10000)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            rmse = mean_squared_error(y.iloc[val_idx], preds) ** 0.5
            fold_scores.append(rmse)
        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha


def train_single_horizon(
    symbol: str,
    frequency: str,
    split_data: SplitData,
    feature_cols: Sequence[str],
    target_col: str,
    model_type: str,
    alpha_grid: Sequence[float],
    *,
    persist: bool = True,
    random_seed: int = 667,
) -> ModelTrainingResult | None:
    """Train, tune, and evaluate a regression model for a single horizon."""
    train_X, train_y = _prepare_matrix(split_data.train, feature_cols, target_col)
    if train_y.empty:
        return None

    n_samples = len(train_X)
    val_window = max(1, min(60, n_samples // 5 or 1))
    max_train = max(1, n_samples - val_window)
    min_train = min(max(val_window * 2, 50), max_train)
    if min_train < val_window:
        min_train = val_window
    best_alpha = time_series_cv(
        train_X,
        train_y,
        model_type=model_type,
        alpha_grid=alpha_grid,
        min_train_size=min_train,
        val_window=val_window,
    )
    if best_alpha is None:
        return None

    train_val_df = pd.concat([split_data.train, split_data.validation], ignore_index=True)
    final_X, final_y = _prepare_matrix(train_val_df, feature_cols, target_col)
    if final_y.empty:
        return None
    model_cls = MODEL_TYPES[model_type]
    model_params = model_cls().get_params()
    if "random_state" in model_params:
        model = model_cls(alpha=best_alpha, max_iter=10000, random_state=random_seed)
    else:
        model = model_cls(alpha=best_alpha, max_iter=10000)
    model.fit(final_X, final_y)

    trained_feature_cols = list(final_X.columns)

    test_X, test_y = _prepare_matrix(split_data.test, feature_cols, target_col)
    error_series: List[Dict[str, float | str | None]] = []
    if test_y.empty:
        metrics = {"rmse": None, "mae": None, "r2": None}
        predictions = np.array([])
    else:
        predictions = model.predict(test_X)
        metrics = evaluate_predictions(test_y.values, predictions)
        test_dates = split_data.test["date"].reset_index(drop=True)
        for idx, (actual, pred) in enumerate(zip(test_y.values, predictions)):
            date_value = test_dates.iloc[idx] if idx < len(test_dates) else None
            if hasattr(date_value, "isoformat"):
                date_str = date_value.isoformat()
            else:
                date_str = str(date_value)
            error_series.append(
                {
                    "date": date_str,
                    "actual": float(actual),
                    "predicted": float(pred),
                    "error": float(pred - actual),
                }
            )

    model_path = None
    metrics_path = None
    if persist:
        model_path = build_model_path(symbol, frequency, model_type, target_col)
        metrics_path = build_metrics_path(symbol, frequency, model_type, target_col)
        save_model(model, model_path)
        store_metrics(metrics, metrics_path)

    feature_importance: List[Dict[str, float | str]] = []
    if hasattr(model, "coef_"):
        coefs = getattr(model, "coef_")
        if isinstance(coefs, np.ndarray) and coefs.ndim > 1:
            coefs = coefs.ravel()
        for feature, weight in zip(trained_feature_cols, coefs):
            weight_float = float(weight)
            feature_importance.append(
                {
                    "feature": feature,
                    "weight": weight_float,
                    "abs_weight": abs(weight_float),
                }
            )
        feature_importance.sort(key=lambda item: item["abs_weight"], reverse=True)

    insight_flags: List[str] = []
    if not error_series:
        insight_flags.append("no_test_data")
    if metrics.get("r2") is not None and metrics["r2"] is not None and metrics["r2"] < 0:
        insight_flags.append("negative_r2")
    if metrics.get("rmse") is None:
        insight_flags.append("missing_rmse")

    if persist:
        artifacts: List[Dict[str, str]] = []
        if model_path:
            artifacts.append({"type": "model", "path": str(model_path)})
        if metrics_path:
            artifacts.append({"type": "table", "description": "regression_metrics", "path": str(metrics_path)})
        save_regression_summary(
            symbol=symbol,
            frequency=frequency,
            model_type=model_type,
            horizon=target_col,
            metrics=metrics,
            best_alpha=float(best_alpha),
            feature_importance=feature_importance,
            error_series=error_series,
            artifacts=artifacts,
            insight_flags=insight_flags,
        )

    return ModelTrainingResult(
        horizon=target_col,
        best_alpha=best_alpha,
        metrics=metrics,
        model_path=model_path,
        metrics_path=metrics_path,
    )


def train_regression_models(
    symbol: str,
    frequency: str = "daily",
    model_type: str = "ridge",
    alpha_grid: Sequence[float] | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    *,
    walk_forward: Dict[str, Any] | None = None,
    random_seed: int = 667,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> Dict[str, ModelTrainingResult]:
    """Canonical entry point for training regression models across horizons."""
    model_type = model_type.lower()
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from {list(MODEL_TYPES)}.")
    alpha_grid = alpha_grid or [0.001, 0.01, 0.1, 1.0, 10.0]

    conn = get_connection(db_path)
    try:
        dataset = load_feature_data(symbol, frequency, conn)
    finally:
        conn.close()

    if dataset.data.empty:
        message = f"No feature data for {symbol.upper()} ({frequency})."
        logger.warning(message)
        print(message)
        return {}
    split_definition = get_or_create_time_split(
        symbol,
        frequency,
        dataset.data["date"],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    split_data = split_dataset_by_timesplit(dataset.data, split_definition)
    results: Dict[str, ModelTrainingResult] = {}
    for horizon in TARGET_HORIZONS:
        if horizon not in dataset.target_cols:
            continue
        result = train_single_horizon(
            symbol=symbol,
            frequency=frequency,
            split_data=split_data,
            feature_cols=dataset.feature_cols,
            target_col=horizon,
            model_type=model_type,
            alpha_grid=alpha_grid,
            random_seed=random_seed,
        )
        if result:
            results[horizon] = result
            message = (
                f"{symbol.upper()} {frequency} {horizon}: alpha={result.best_alpha}, "
                f"metrics={result.metrics}"
            )
            logger.info(message)
            print(message)
        else:
            message = f"Skipping horizon {horizon} for {symbol.upper()} ({frequency}) due to insufficient data."
            logger.warning(message)
            print(message)
    if walk_forward:
        run_walk_forward_analysis(
            symbol,
            frequency,
            dataset.data,
            dataset.feature_cols,
            walk_forward,
            model_type,
            alpha_grid,
            random_seed=random_seed,
        )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary_daily = train_regression_models("AAPL", "daily")
    summary_weekly = train_regression_models("AAPL", "weekly")
    daily_message = f"Daily horizons trained: {list(summary_daily)}"
    weekly_message = f"Weekly horizons trained: {list(summary_weekly)}"
    logger.info(daily_message)
    logger.info(weekly_message)
    print(daily_message)
    print(weekly_message)
