"""Feature engineering pipeline for daily and weekly stock price data."""
from __future__ import annotations

import logging
from pathlib import Path
import sqlite3
import sys
from typing import Iterable, Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from scripts.targets import compute_forward_returns


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SCALER_DIR = MODELS_DIR / "scalers"
DEFAULT_DB_PATH = DATA_DIR / "stocks.db"

PRICE_TABLES = {"daily": "prices_daily", "weekly": "prices_weekly"}
FEATURE_TABLES = {"daily": "features_daily", "weekly": "features_weekly"}
RETURN_PERIODS = [1, 2, 3, 5, 10]
MA_WINDOWS = [5, 10, 20, 50, 100, 200]
VOLUME_WINDOWS = [5, 10, 20, 50]
MOMENTUM_LAGS = [5, 10, 20]
VOLATILITY_WINDOWS = [10, 20, 50]
VARIANCE_THRESHOLD = 1e-8
CORRELATION_THRESHOLD = 0.97
LEAKAGE_AUDIT_SAMPLES = 8
LEAKAGE_TOLERANCE = 1e-8


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def ensure_feature_table(conn: sqlite3.Connection, frequency: str, columns: Iterable[str]) -> None:
    """Ensure the feature table exists and contains all required columns."""
    table = FEATURE_TABLES[frequency]
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for column in columns:
        if column in ("symbol", "date"):
            continue
        if column not in existing:
            conn.execute(f'ALTER TABLE {table} ADD COLUMN "{column}" REAL')


def load_price_data(symbol: str, frequency: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Load daily or weekly price data for a symbol sorted by date."""
    frequency = frequency.lower()
    table = PRICE_TABLES[frequency]
    date_column = "date" if frequency == "daily" else "week_start"
    query = f"""
        SELECT symbol, {date_column} AS date, open, high, low, close, volume
        FROM {table}
        WHERE symbol = ?
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(symbol.upper(),))
    return df


def add_return_features(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Add simple, log, lagged, and rolling cumulative returns."""
    result = df.copy()
    close = result["close"]
    for period in RETURN_PERIODS:
        result[f"return_{period}d"] = close.pct_change(periods=period)
        result[f"log_return_{period}d"] = np.log1p(result[f"return_{period}d"])
        result[f"lag_return_{period}d"] = result[f"return_{period}d"].shift(1)

    result["return_1d"] = close.pct_change()
    for window in (5, 10, 20):
        rolling = (1 + result["return_1d"]).rolling(window=window).apply(np.prod, raw=True) - 1
        result[f"rolling_cum_return_{window}d"] = rolling

    if frequency == "daily":
        result["return_1w"] = close.pct_change(periods=5)
    else:
        result["return_1w"] = result["return_1d"]

    return result


def add_moving_average_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages, slopes, and price-to-MA ratios."""
    result = df.copy()
    for window in MA_WINDOWS:
        ma_col = f"ma_{window}"
        result[ma_col] = result["close"].rolling(window=window).mean()
        result[f"{ma_col}_slope"] = result[ma_col] - result[ma_col].shift(1)
        result[f"price_over_{ma_col}"] = result["close"] / result[ma_col]
    return result


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, momentum, and stochastic oscillator indicators."""
    result = df.copy()
    delta = result["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    window = 14
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    result["rsi_14"] = 100 - (100 / (1 + rs))

    ema_fast = result["close"].ewm(span=12, adjust=False).mean()
    ema_slow = result["close"].ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    result["macd"] = macd
    result["macd_signal"] = signal
    result["macd_hist"] = macd - signal

    for lag in MOMENTUM_LAGS:
        result[f"momentum_{lag}d"] = result["close"] - result["close"].shift(lag)

    lowest_low = result["low"].rolling(window=window).min()
    highest_high = result["high"].rolling(window=window).max()
    result["stoch_k"] = (result["close"] - lowest_low) / (highest_high - lowest_low)
    result["stoch_d"] = result["stoch_k"].rolling(window=3).mean()

    return result


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility measures such as rolling std and ATR."""
    result = df.copy()
    for window in VOLATILITY_WINDOWS:
        result[f"volatility_std_{window}d"] = result["return_1d"].rolling(window=window).std()

    prev_close = result["close"].shift(1)
    high_low = result["high"] - result["low"]
    high_close = (result["high"] - prev_close).abs()
    low_close = (result["low"] - prev_close).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    result["atr_14"] = true_range.rolling(window=14).mean()
    result["volatility_ratio_10_20"] = result["volatility_std_10d"] / result["volatility_std_20d"]
    result["volatility_ratio_20_50"] = result["volatility_std_20d"] / result["volatility_std_50d"]
    return result


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling volume statistics and OBV."""
    result = df.copy()
    for window in VOLUME_WINDOWS:
        vol_ma = result["volume"].rolling(window=window).mean()
        result[f"volume_ma_{window}"] = vol_ma
        result[f"volume_over_ma_{window}"] = result["volume"] / vol_ma

    price_diff = result["close"].diff().fillna(0)
    direction = np.sign(price_diff)
    result["obv"] = (direction * result["volume"].fillna(0)).cumsum()
    return result


def generate_polynomial_and_interaction_features(
    df: pd.DataFrame,
    base_features: Sequence[str],
    degree: int = 2,
) -> tuple[pd.DataFrame, PolynomialFeatures]:
    """Expand selected features using polynomial/interaction terms."""
    if not base_features:
        return df.copy(), PolynomialFeatures(degree=degree, include_bias=False)

    transformer = PolynomialFeatures(degree=degree, include_bias=False)
    base = df[base_features].fillna(0.0)
    expanded = transformer.fit_transform(base)
    names = transformer.get_feature_names_out(base_features)
    expanded_df = pd.DataFrame(expanded, columns=[f"poly_{name}" for name in names], index=df.index)
    result = pd.concat([df, expanded_df], axis=1)
    return result, transformer


def _prepare_scaling_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Forward/backward fill then zero-fill feature columns for scaling."""
    subset = df[feature_cols].copy()
    subset = subset.ffill().bfill().fillna(0.0)
    return subset


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    scaler_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on train data and transform train/test."""
    if not feature_cols:
        return pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index), StandardScaler()

    train_subset = _prepare_scaling_frame(train_df, feature_cols)
    scaler = StandardScaler()
    scaler.fit(train_subset)

    train_scaled = scaler.transform(train_subset)
    train_scaled_df = pd.DataFrame(
        train_scaled,
        columns=[f"scaled_{col}" for col in feature_cols],
        index=train_df.index,
    )

    test_subset = _prepare_scaling_frame(test_df, feature_cols)
    test_scaled = scaler.transform(test_subset) if not test_subset.empty else np.empty((0, len(feature_cols)))
    test_scaled_df = pd.DataFrame(
        test_scaled,
        columns=[f"scaled_{col}" for col in feature_cols],
        index=test_df.index,
    )

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    dump(scaler, scaler_path)
    return train_scaled_df, test_scaled_df, scaler


def store_features(df: pd.DataFrame, frequency: str, conn: sqlite3.Connection) -> int:
    """Store engineered features back to SQLite with upsert behavior."""
    if df.empty:
        return 0

    columns = ["symbol", "date"] + [col for col in df.columns if col not in ("symbol", "date")]
    ensure_feature_table(conn, frequency, columns)
    column_names = ", ".join(f'"{col}"' for col in columns)
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(f'"{col}"=excluded."{col}"' for col in columns if col not in ("symbol", "date"))
    table = FEATURE_TABLES[frequency]
    sql = f"""
        INSERT INTO {table} ({column_names})
        VALUES ({placeholders})
        ON CONFLICT(symbol, date) DO UPDATE SET
        {updates}
    """

    def _convert(value: object) -> object:
        if pd.isna(value):
            return None
        if isinstance(value, (np.floating, np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.integer, np.int32, np.int64)):
            return int(value)
        return value

    rows = []
    for _, row in df[columns].iterrows():
        rows.append(tuple(_convert(row[col]) for col in columns))

    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def generate_features(
    symbol: str,
    frequency: str = "daily",
    train_end_date: str | None = None,
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    poly_degree: int = 2,
    start_date: str | None = "2022-01-01",
) -> pd.DataFrame:
    """Full feature engineering workflow for a symbol/frequency."""
    frequency = frequency.lower()
    conn = get_connection(db_path)
    try:
        price_df = load_price_data(symbol, frequency, conn)
        if price_df.empty:
            return price_df

        if start_date:
            cutoff = pd.to_datetime(start_date)
            price_df["date"] = pd.to_datetime(price_df["date"])
            price_df = price_df.loc[price_df["date"] >= cutoff].copy()
            price_df["date"] = price_df["date"].dt.strftime("%Y-%m-%d")
        if price_df.empty:
            return price_df

        df = build_raw_feature_frame(price_df, frequency, poly_degree)

        categorical_cols = {"symbol", "date"}
        target_cols = [col for col in df.columns if col.startswith("target_")]
        numeric_cols = [
            col
            for col in df.columns
            if col not in categorical_cols.union(target_cols)
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        numeric_cols = select_feature_columns(df, numeric_cols)
        audit_feature_leakage(price_df, df, frequency, numeric_cols, poly_degree)
        df = df.drop(columns=[col for col in df.columns if col.startswith("target_") and col not in target_cols])

        if train_end_date:
            train_mask = df["date"] <= train_end_date
        else:
            train_mask = pd.Series(True, index=df.index)
        test_mask = ~train_mask

        scaler_path = SCALER_DIR / f"{symbol.upper()}_{frequency}_scaler.pkl"
        train_scaled, test_scaled, _ = scale_features(
            df.loc[train_mask], df.loc[test_mask], numeric_cols, scaler_path
        )
        scaled_cols: list[str] = list(train_scaled.columns)
        if not scaled_cols:
            scaled_cols = list(test_scaled.columns)
        if scaled_cols:
            df.loc[train_mask, scaled_cols] = train_scaled
            df.loc[test_mask, scaled_cols] = test_scaled

        stored = store_features(df, frequency, conn)
        message = f"Stored {stored} {frequency} feature rows for {symbol.upper()}."
        logger.info(message)
        print(message)
        return df
    finally:
        conn.close()


def build_raw_feature_frame(price_df: pd.DataFrame, frequency: str, poly_degree: int) -> pd.DataFrame:
    """Construct the unscaled feature frame for a price dataframe."""
    df = price_df.copy()
    df = add_return_features(df, frequency)
    df = add_moving_average_features(df)
    df = add_momentum_indicators(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    target_df = compute_forward_returns(df["close"], frequency)
    df = pd.concat([df, target_df], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan)

    base_features = [
        "return_1d",
        "return_5d",
        "ma_10",
        "ma_20",
        "macd",
        "rsi_14",
        "volatility_std_20d",
        "volume_ma_10",
    ]
    available_base = [feature for feature in base_features if feature in df.columns]
    df, _ = generate_polynomial_and_interaction_features(df, available_base, degree=poly_degree)
    return df


def select_feature_columns(df: pd.DataFrame, numeric_cols: Sequence[str]) -> Sequence[str]:
    """Apply variance and correlation-based filtering to numeric feature columns."""
    if not numeric_cols:
        return numeric_cols
    variances = df[numeric_cols].var().fillna(0.0)
    candidates = [col for col in numeric_cols if variances[col] >= VARIANCE_THRESHOLD]
    if not candidates:
        return candidates
    corr_matrix = df[candidates].corr().abs()
    dropped = set()
    selected = []
    for col in candidates:
        if col in dropped:
            continue
        selected.append(col)
        for other in candidates:
            if other == col or other in dropped:
                continue
            if corr_matrix.loc[col, other] >= CORRELATION_THRESHOLD:
                dropped.add(other)
    message = f"Feature selection retained {len(selected)} columns (from {len(numeric_cols)})."
    logger.info(message)
    print(message)
    return selected


def audit_feature_leakage(
    price_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    frequency: str,
    numeric_cols: Sequence[str],
    poly_degree: int,
    *,
    samples: int = LEAKAGE_AUDIT_SAMPLES,
) -> None:
    """Verify that feature columns can be reproduced without future information."""
    if feature_df.empty or not numeric_cols:
        return
    n = len(feature_df)
    sample_indices = sorted({int(idx) for idx in np.linspace(max(0, n // 4), n - 1, samples)})
    for idx in sample_indices:
        subset_price = price_df.iloc[: idx + 1]
        recalculated = build_raw_feature_frame(subset_price, frequency, poly_degree).iloc[-1]
        current_row = feature_df.iloc[idx]
        for col in numeric_cols:
            if col not in recalculated or col not in current_row:
                continue
            val_full = current_row[col]
            val_recalc = recalculated[col]
            if pd.isna(val_full) and pd.isna(val_recalc):
                continue
            if abs(val_full - val_recalc) > LEAKAGE_TOLERANCE:
                raise ValueError(
                    f"Feature leakage detected for column '{col}' at index {idx}: "
                    f"full={val_full}, recomputed={val_recalc}"
                )
    message = "Feature leakage audit passed for numeric features."
    logger.info(message)
    print(message)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    daily_features = generate_features("AAPL", "daily")
    daily_message = f"Daily feature shape: {daily_features.shape}"
    logger.info(daily_message)
    print(daily_message)
    weekly_features = generate_features("AAPL", "weekly")
    weekly_message = f"Weekly feature shape: {weekly_features.shape}"
    logger.info(weekly_message)
    print(weekly_message)
