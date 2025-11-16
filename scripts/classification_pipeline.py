"""Classification pipeline converting regression outputs into buy/no-buy decisions."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from scripts.targets import (
    get_horizon_steps,
    ordered_target_names,
    target_columns_for_window,
)
from scripts.time_splits import TimeSplit, get_or_create_time_split, load_time_split
from scripts.diagnostic_outputs import save_classification_summary


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIER_MODEL_DIR = MODELS_DIR / "classification" / "models"
CLASSIFIER_SCALER_DIR = MODELS_DIR / "classification" / "scalers"
CLASSIFIER_THRESHOLD_DIR = MODELS_DIR / "classification" / "thresholds"
CLASSIFIER_METRICS_DIR = MODELS_DIR / "metrics" / "classification"
REGRESSION_MODEL_DIR = MODELS_DIR / "regression"
DEFAULT_DB_PATH = DATA_DIR / "stocks.db"

FEATURE_TABLES = {"daily": "features_daily", "weekly": "features_weekly"}
METADATA_COLUMNS = ["symbol", "date"]
REGRESSION_HORIZONS = ordered_target_names()
DEFAULT_TECH_FEATURES = [
    "return_1d",
    "return_5d",
    "rsi_14",
    "macd",
    "volatility_std_20d",
    "volume_ma_10",
    "price_over_ma_20",
]


@dataclass
class ClassificationDataset:
    """Encapsulates loaded feature data and column partitions."""

    data: pd.DataFrame
    feature_cols: List[str]
    metadata_cols: List[str]


@dataclass
class ClassificationSplit:
    """Train/validation/test split for classification."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass
class ClassificationResult:
    """Stores training outcome for later inspection."""

    symbol: str
    frequency: str
    model_path: Path | None
    scaler_path: Path | None
    threshold_path: Path | None
    metrics_path: Path | None
    metrics: Dict[str, float]
    threshold: float
    class_weight: Optional[Dict[int, float] | str] = None


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def load_feature_data(symbol: str, frequency: str, conn: sqlite3.Connection) -> ClassificationDataset:
    """Load engineered features for a symbol/frequency pair."""
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
        return ClassificationDataset(df, [], METADATA_COLUMNS.copy())

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    metadata_cols = [col for col in METADATA_COLUMNS if col in df.columns]
    excluded = set(metadata_cols)
    feature_cols = [
        col
        for col in df.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(df[col])
        and not col.startswith("target_")
    ]
    return ClassificationDataset(data=df, feature_cols=feature_cols, metadata_cols=metadata_cols)


def parse_class_weight(value: Optional[object]) -> Optional[Dict[int, float] | str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parsed = {}
        for key, val in value.items():
            parsed[int(key)] = float(val)
        return parsed
    return None


def build_classification_split(df: pd.DataFrame, split: TimeSplit) -> ClassificationSplit:
    frames = split.to_frames(df, date_col="date")
    return ClassificationSplit(train=frames["train"], validation=frames["validation"], test=frames["test"])


def construct_buy_labels(
    df: pd.DataFrame,
    *,
    frequency: str,
    threshold: float = 0.02,
    start_offset: int = 1,
    end_offset: int = 5,
) -> Tuple[pd.Series, pd.Series]:
    """Create binary labels based on computed forward-return targets."""
    candidate_cols = target_columns_for_window(frequency, start_offset, end_offset)
    available_cols = [col for col in candidate_cols if col in df.columns]
    if not available_cols:
        raise ValueError(
            f"No target columns available for window {start_offset}-{end_offset} in frequency '{frequency}'."
        )
    future_matrix = df[available_cols]
    max_future = future_matrix.max(axis=1)
    labels = (max_future >= threshold).astype(float)
    labels = labels.where(~max_future.isna(), np.nan)
    return labels, max_future


def prepare_numeric_features(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Fill missing values to create a dense numeric feature matrix."""
    features = df[list(columns)].replace([np.inf, -np.inf], np.nan)
    features = features.ffill().fillna(0.0)
    return features


def build_regression_model_path(symbol: str, frequency: str, model_type: str, horizon: str) -> Path:
    """Return the expected path to a regression model artifact."""
    safe_horizon = horizon.replace("target_", "")
    filename = f"{symbol.upper()}_{frequency}_{model_type}_{safe_horizon}.joblib"
    return REGRESSION_MODEL_DIR / frequency / filename


def compute_regression_predictions(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    symbol: str,
    frequency: str,
    horizons: Sequence[str],
    model_type: str = "ridge",
) -> Dict[str, pd.Series]:
    """Load regression models and generate predictions for all requested horizons."""
    prepared = prepare_numeric_features(df, feature_cols)
    predictions: Dict[str, pd.Series] = {}
    for horizon in horizons:
        model_path = build_regression_model_path(symbol, frequency, model_type, horizon)
        if not model_path.exists():
            continue
        model = load(model_path)
        preds = model.predict(prepared)
        predictions[horizon] = pd.Series(preds, index=df.index, name=f"pred_{horizon}")
    return predictions


def build_classification_features(
    df: pd.DataFrame,
    regression_preds: Dict[str, pd.Series],
    *,
    selected_features: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Combine regression outputs with optional technical features."""
    combined = df.copy()
    for horizon, series in regression_preds.items():
        combined[f"pred_{horizon}"] = series
    feature_cols = sorted([f"pred_{h}" for h in regression_preds])
    if selected_features:
        available = [col for col in selected_features if col in combined.columns]
        feature_cols.extend(available)
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_matrix = combined[feature_cols]
    feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return feature_matrix, feature_cols


def rolling_expanding_window_indices(
    n_samples: int,
    min_train_size: int,
    val_window: int,
    step_size: Optional[int] = None,
    max_splits: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window folds for time-series cross-validation."""
    if n_samples < min_train_size + val_window:
        return []
    step = step_size or val_window
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
        train_end = val_end + step
    return splits


def fit_logistic_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    penalty: str = "l2",
    C: float = 1.0,
    class_weight: Optional[Dict[int, float] | str] = None,
) -> LogisticRegression:
    """Instantiate and fit a logistic regression classifier."""
    solver = "saga" if penalty == "l1" else "lbfgs"
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        max_iter=5000,
        random_state=667,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    return model


def time_series_cv_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    penalties: Sequence[str],
    c_values: Sequence[float],
    min_train_size: int,
    val_window: int,
    metric: str = "f1",
    max_splits: int = 5,
    future_returns: Optional[pd.Series] = None,
    decision_threshold: float = 0.5,
    class_weight: Optional[Dict[int, float] | str] = None,
) -> Tuple[str, float]:
    """Return penalty/C combination maximizing the desired metric."""
    folds = rolling_expanding_window_indices(
        len(X), min_train_size=min_train_size, val_window=val_window, max_splits=max_splits
    )
    if not folds:
        return penalties[0], c_values[0]

    best_combo = (penalties[0], c_values[0])
    best_score = -np.inf
    for penalty in penalties:
        for C in c_values:
            fold_scores = []
            for train_idx, val_idx in folds:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X.iloc[train_idx])
                X_val = scaler.transform(X.iloc[val_idx])
                model = fit_logistic_model(
                    X_train,
                    y.iloc[train_idx].values,
                    penalty=penalty,
                    C=C,
                    class_weight=class_weight,
                )
                probs = model.predict_proba(X_val)[:, 1]
                preds = (probs >= decision_threshold).astype(int)
                if metric == "f1":
                    score = f1_score(y.iloc[val_idx], preds, zero_division=0)
                elif metric == "accuracy":
                    score = accuracy_score(y.iloc[val_idx], preds)
                elif metric == "expected_pnl":
                    if future_returns is None:
                        raise ValueError("future_returns must be provided for expected_pnl metric.")
                    returns_slice = future_returns.iloc[val_idx].values
                    score = expected_pnl_score(preds, returns_slice)
                else:
                    score = accuracy_score(y.iloc[val_idx], preds)
                fold_scores.append(score)
            avg_score = float(np.mean(fold_scores))
            if avg_score > best_score:
                best_score = avg_score
                best_combo = (penalty, C)
    return best_combo


def select_probability_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    strategy: str = "f1",
    thresholds: Optional[Sequence[float]] = None,
    min_recall: float = 0.5,
    future_returns: Optional[np.ndarray] = None,
) -> float:
    """Choose the probability cutoff based on the supplied strategy."""
    thresholds = thresholds or np.linspace(0.3, 0.7, 21)
    best_threshold = 0.5
    best_score = -np.inf

    if strategy == "precision_recall":
        precision, recall, thresh = precision_recall_curve(y_true, probs)
        thresh = thresh if len(thresh) else np.array([0.5])
        for p, r, t in zip(precision, recall, np.append(thresh, thresh[-1])):
            if r >= min_recall and p > best_score:
                best_score = p
                best_threshold = t
        return float(best_threshold)

    for t in thresholds:
        preds = (probs >= t).astype(int)
        if strategy == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif strategy == "youden":
            score = recall_score(y_true, preds, zero_division=0) + precision_score(
                y_true, preds, zero_division=0
            ) - 1
        elif strategy == "pnl":
            if future_returns is None:
                raise ValueError("future_returns required for pnl strategy.")
            score = expected_pnl_score(preds, future_returns)
        else:
            score = accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_threshold = t
    return float(best_threshold)


def evaluate_classification(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    future_returns: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute standard classification metrics."""
    preds = (probs >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else np.nan,
    }
    if future_returns is not None:
        metrics["expected_pnl"] = expected_pnl_score(preds, future_returns)
        trades = future_returns[preds == 1]
        metrics["trades"] = int(np.count_nonzero(preds))
        metrics["avg_trade_return"] = float(np.nanmean(trades)) if trades.size else 0.0
    return {key: float(value) for key, value in metrics.items()}


def expected_pnl_score(decisions: np.ndarray, future_returns: np.ndarray) -> float:
    """Compute average realized return for executed trades."""
    trades = future_returns[decisions == 1]
    if trades.size == 0:
        return 0.0
    return float(np.nanmean(trades))


def build_classification_identifier(
    symbol: str,
    frequency: str,
    threshold: float,
    window: Tuple[int, int],
) -> str:
    """Create a unique identifier for classifier artifacts."""
    thr = int(threshold * 10000)
    return f"{symbol.upper()}_{frequency}_thr{thr}_w{window[0]}-{window[1]}"


def save_classifier_artifacts(
    *,
    model: LogisticRegression,
    scaler: StandardScaler,
    threshold: float,
    identifier: str,
    metrics: Dict[str, float],
) -> Tuple[Path, Path, Path, Path]:
    """Persist classifier, scaler, threshold, and metrics."""
    CLASSIFIER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CLASSIFIER_SCALER_DIR.mkdir(parents=True, exist_ok=True)
    CLASSIFIER_THRESHOLD_DIR.mkdir(parents=True, exist_ok=True)
    CLASSIFIER_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = CLASSIFIER_MODEL_DIR / f"{identifier}.joblib"
    scaler_path = CLASSIFIER_SCALER_DIR / f"{identifier}.joblib"
    threshold_path = CLASSIFIER_THRESHOLD_DIR / f"{identifier}.json"
    metrics_path = CLASSIFIER_METRICS_DIR / f"{identifier}.json"

    dump(model, model_path)
    dump(scaler, scaler_path)
    with threshold_path.open("w", encoding="utf-8") as fh:
        json.dump({"threshold": threshold}, fh, indent=2)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return model_path, scaler_path, threshold_path, metrics_path


def load_classifier_artifacts(identifier: str) -> Tuple[LogisticRegression, StandardScaler, float]:
    """Load persisted classifier, scaler, and probability threshold."""
    model_path = CLASSIFIER_MODEL_DIR / f"{identifier}.joblib"
    scaler_path = CLASSIFIER_SCALER_DIR / f"{identifier}.joblib"
    threshold_path = CLASSIFIER_THRESHOLD_DIR / f"{identifier}.json"
    model = load(model_path)
    scaler = load(scaler_path)
    with threshold_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return model, scaler, float(data["threshold"])


def train_buy_classifier(
    symbol: str,
    *,
    frequency: str = "daily",
    threshold: float = 0.02,
    window: Tuple[int, int] = (1, 5),
    regression_model_type: str = "ridge",
    selected_features: Optional[Sequence[str]] = None,
    penalties: Sequence[str] = ("l2", "l1"),
    c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.2,
    probability_strategy: str = "f1",
    cv_metric: str = "f1",
    class_weight: Optional[Dict[int, float] | str] = None,
    random_seed: int = 667,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> ClassificationResult | None:
    """Master orchestration for training the buy/no-buy classifier."""
    conn = get_connection(db_path)
    try:
        dataset = load_feature_data(symbol, frequency, conn)
    finally:
        conn.close()

    if dataset.data.empty:
        message = f"No feature data for {symbol.upper()} ({frequency})."
        logger.warning(message)
        print(message)
        return None
    class_weight = parse_class_weight(class_weight)

    regression_outputs = compute_regression_predictions(
        dataset.data,
        dataset.feature_cols,
        symbol=symbol,
        frequency=frequency,
        horizons=REGRESSION_HORIZONS,
        model_type=regression_model_type,
    )
    if not regression_outputs:
        message = "Regression predictions missing; train regression models first."
        logger.warning(message)
        print(message)
        return None

    labels, future_returns = construct_buy_labels(
        dataset.data,
        frequency=frequency,
        threshold=threshold,
        start_offset=window[0],
        end_offset=window[1],
    )
    dataset.data["label_buy"] = labels
    dataset.data["label_buy_return"] = future_returns
    feature_matrix, feature_cols = build_classification_features(
        dataset.data, regression_outputs, selected_features=selected_features or DEFAULT_TECH_FEATURES
    )
    combined = pd.concat(
        [dataset.data[dataset.metadata_cols], feature_matrix, dataset.data[["label_buy", "label_buy_return"]]], axis=1
    )
    combined = combined.dropna(subset=["label_buy"])
    if combined.empty:
        message = "Label construction removed all rows; adjust threshold/window."
        logger.warning(message)
        print(message)
        return None

    split_def = load_time_split(symbol, frequency)
    if split_def is None:
        split_def = get_or_create_time_split(
            symbol,
            frequency,
            combined["date"],
            train_ratio=train_ratio,
            val_ratio=validation_ratio,
            test_ratio=test_ratio,
        )
    split = build_classification_split(combined, split_def)

    def build_xy(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        X = frame[feature_cols]
        y = frame["label_buy"].astype(int)
        returns = frame["label_buy_return"].astype(float)
        return X, y, returns

    X_train, y_train, ret_train = build_xy(split.train)
    X_val, y_val, ret_val = build_xy(split.validation)
    X_test, y_test, ret_test = build_xy(split.test)

    if len(y_train.unique()) < 2 or len(y_val) == 0 or len(y_test) == 0:
        message = "Insufficient class diversity for training."
        logger.warning(message)
        print(message)
        return None

    n_samples = len(X_train)
    val_window = max(5, min(n_samples // 5, 50))
    min_train_size = max(val_window * 2, 50)
    penalty, C = time_series_cv_classifier(
        X_train,
        y_train,
        penalties=penalties,
        c_values=c_values,
        min_train_size=min_train_size,
        val_window=val_window,
        metric=cv_metric,
        future_returns=ret_train.reset_index(drop=True),
        class_weight=class_weight,
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    model = fit_logistic_model(
        scaler.transform(X_train),
        y_train.values,
        penalty=penalty,
        C=C,
        class_weight=class_weight,
    )

    val_probs = model.predict_proba(scaler.transform(X_val))[:, 1]
    best_threshold = select_probability_threshold(
        y_val.values,
        val_probs,
        strategy=probability_strategy,
        future_returns=ret_val.values if probability_strategy == "pnl" else None,
    )

    train_val_X = pd.concat([X_train, X_val])
    train_val_y = pd.concat([y_train, y_val])
    scaler = StandardScaler()
    scaler.fit(train_val_X)
    model = fit_logistic_model(
        scaler.transform(train_val_X),
        train_val_y.values,
        penalty=penalty,
        C=C,
        class_weight=class_weight,
    )

    test_probs = model.predict_proba(scaler.transform(X_test))[:, 1]
    metrics = evaluate_classification(
        y_test.values,
        test_probs,
        best_threshold,
        future_returns=ret_test.values,
    )

    identifier = build_classification_identifier(symbol, frequency, threshold, window)
    model_path, scaler_path, threshold_path, metrics_path = save_classifier_artifacts(
        model=model,
        scaler=scaler,
        threshold=best_threshold,
        identifier=identifier,
        metrics=metrics,
    )

    preds = (test_probs >= best_threshold).astype(int)
    confusion = confusion_matrix(y_test.values, preds).tolist()
    precision_pts, recall_pts, pr_thresholds = precision_recall_curve(y_test.values, test_probs)
    roc_data: Dict[str, List[float]]
    if len(np.unique(y_test.values)) > 1:
        fpr, tpr, roc_thresholds = roc_curve(y_test.values, test_probs)
        roc_data = {
            "fpr": [float(value) for value in fpr],
            "tpr": [float(value) for value in tpr],
            "thresholds": [float(value) for value in roc_thresholds],
        }
    else:
        roc_data = {"fpr": [], "tpr": [], "thresholds": []}
    curves = {
        "precision_recall": {
            "precision": [float(value) for value in precision_pts],
            "recall": [float(value) for value in recall_pts],
            "thresholds": [float(value) for value in pr_thresholds],
        },
        "roc": roc_data,
    }
    positive_rate = float(y_test.mean()) if len(y_test) else 0.0
    insight_flags: List[str] = []
    if positive_rate and (positive_rate < 0.1 or positive_rate > 0.9):
        insight_flags.append("class_imbalance")
    if metrics.get("f1", 0.0) < 0.5:
        insight_flags.append("low_f1_score")
    roc_auc_value = metrics.get("roc_auc")
    roc_auc_float: Optional[float] = None
    if roc_auc_value is not None:
        try:
            roc_auc_float = float(roc_auc_value)
        except (TypeError, ValueError):
            roc_auc_float = None
    if roc_auc_float is not None and not np.isnan(roc_auc_float) and roc_auc_float < 0.6:
        insight_flags.append("weak_separation")
    expected_pnl = metrics.get("expected_pnl")
    if expected_pnl is not None and expected_pnl < 0:
        insight_flags.append("negative_expected_pnl")
    artifacts = [
        {"type": "model", "path": model_path},
        {"type": "scaler", "path": scaler_path},
        {"type": "table", "description": "classification_threshold", "path": threshold_path},
        {"type": "table", "description": "classification_metrics", "path": metrics_path},
    ]
    save_classification_summary(
        symbol=symbol,
        frequency=frequency,
        identifier=identifier,
        metrics=metrics,
        threshold=best_threshold,
        confusion_matrix=confusion,
        curves=curves,
        artifacts=artifacts,
        insight_flags=insight_flags,
    )

    result_message = f"Classifier trained for {symbol.upper()} ({frequency}); metrics={metrics}"
    logger.info(result_message)
    print(result_message)
    return ClassificationResult(
        symbol=symbol.upper(),
        frequency=frequency,
        model_path=model_path,
        scaler_path=scaler_path,
        threshold_path=threshold_path,
        metrics_path=metrics_path,
        metrics=metrics,
        threshold=best_threshold,
        class_weight=class_weight,
    )


def run_classification_inference(
    symbol: str,
    *,
    frequency: str = "daily",
    threshold: float = 0.02,
    window: Tuple[int, int] = (1, 5),
    regression_model_type: str = "ridge",
    selected_features: Optional[Sequence[str]] = None,
    recent_points: int = 30,
    db_path: Path | str = DEFAULT_DB_PATH,
    split: Optional[TimeSplit] = None,
    random_seed: int = 667,
) -> pd.DataFrame:
    """Generate probabilities and buy/no-buy decisions for recent data."""
    identifier = build_classification_identifier(symbol, frequency, threshold, window)
    model, scaler, prob_threshold = load_classifier_artifacts(identifier)

    conn = get_connection(db_path)
    try:
        dataset = load_feature_data(symbol, frequency, conn)
    finally:
        conn.close()

    regression_outputs = compute_regression_predictions(
        dataset.data,
        dataset.feature_cols,
        symbol=symbol,
        frequency=frequency,
        horizons=REGRESSION_HORIZONS,
        model_type=regression_model_type,
    )
    feature_matrix, feature_cols = build_classification_features(
        dataset.data, regression_outputs, selected_features=selected_features or DEFAULT_TECH_FEATURES
    )
    frame = pd.concat([dataset.data[dataset.metadata_cols], feature_matrix], axis=1)
    if split is not None:
        frame = split.select(frame, "test").reset_index(drop=True)
    else:
        frame = frame.tail(recent_points).reset_index(drop=True)
    scaled = scaler.transform(frame[feature_cols])
    probs = model.predict_proba(scaled)[:, 1]
    frame["prob_buy"] = probs
    frame["decision"] = (frame["prob_buy"] >= prob_threshold).astype(int)
    return frame


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = train_buy_classifier("AAPL", frequency="daily")
    if result:
        inference_df = run_classification_inference("AAPL", frequency="daily")
        logger.info("Inference preview:\n%s", inference_df.tail())
        print(inference_df.tail())
