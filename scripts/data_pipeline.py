"""Stock data ingestion and storage pipeline using SQLite."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence
import sqlite3

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "stocks.db"


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection, ensuring the database directory exists."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Create the required price tables if they do not exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prices_daily (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prices_weekly (
            symbol TEXT NOT NULL,
            week_start TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, week_start)
        )
        """
    )
    conn.commit()


def fetch_daily_data(
    symbol: str, start_date: str, end_date: str | None = None
) -> pd.DataFrame:
    """Download daily OHLCV data for a symbol using yfinance."""
    symbol = symbol.upper()
    end = end_date or datetime.now(tz=UTC).date().isoformat()
    raw = yf.download(
        symbol,
        start=start_date,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            raw.columns = raw.columns.droplevel(-1)
        except ValueError:
            raw.columns = ["_".join(str(part) for part in col if part) for col in raw.columns]

    if raw.empty:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])

    df = raw.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    keep_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    return df[keep_cols]


def store_daily_data(df: pd.DataFrame, symbol: str, conn: sqlite3.Connection) -> int:
    """Insert daily price data into SQLite, replacing existing duplicates."""
    if df.empty:
        return 0
    df = df.copy()
    df["symbol"] = symbol.upper()
    df = df.where(pd.notnull(df), None)
    rows: list[tuple] = [
        (
            row.symbol,
            row.date,
            row.open,
            row.high,
            row.low,
            row.close,
            row.volume,
        )
        for row in df.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO prices_daily (symbol, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, date) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def compute_weekly_from_daily(df_daily: pd.DataFrame, week_start_day: int = 0) -> pd.DataFrame:
    """Aggregate a daily DataFrame into weekly OHLCV values.

    Args:
        df_daily: Daily prices containing symbol/date/open/high/low/close/volume.
        week_start_day: Monday=0 ... Sunday=6, used to compute the label stored in week_start.
    """
    if df_daily.empty:
        return pd.DataFrame(columns=["symbol", "week_start", "open", "high", "low", "close", "volume"])

    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    # Align every date to the requested week start (default Monday)
    df["week_start"] = df["date"] - pd.to_timedelta((df["date"].dt.weekday - week_start_day) % 7, unit="D")

    grouped = (
        df.groupby(["symbol", "week_start"], as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
    )
    grouped["week_start"] = grouped["week_start"].dt.strftime("%Y-%m-%d")
    return grouped


def store_weekly_data(df: pd.DataFrame, symbol: str, conn: sqlite3.Connection) -> int:
    """Insert weekly aggregates into SQLite."""
    if df.empty:
        return 0
    df = df.copy()
    df["symbol"] = symbol.upper()
    df = df.where(pd.notnull(df), None)
    rows = [
        (
            row.symbol,
            row.week_start,
            row.open,
            row.high,
            row.low,
            row.close,
            row.volume,
        )
        for row in df.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO prices_weekly (symbol, week_start, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, week_start) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume
        """,
        rows,
    )
    conn.commit()
    return len(rows)


@dataclass
class UpdateResult:
    """Simple container summarizing how many rows were written."""

    symbol: str
    daily_rows: int = 0
    weekly_rows: int = 0


def update_data(
    symbol: str,
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    frequencies: Sequence[str] = ("daily", "weekly"),
) -> UpdateResult:
    """High-level workflow that fetches, aggregates, and stores price data."""
    symbol = symbol.upper()
    end = end_date or datetime.now(tz=UTC).date().isoformat()
    conn = get_connection(db_path)
    ensure_tables(conn)
    try:
        result = UpdateResult(symbol=symbol)
        df_daily = fetch_daily_data(symbol, start_date, end)
        if "daily" in frequencies:
            result.daily_rows = store_daily_data(df_daily, symbol, conn)
        if "weekly" in frequencies:
            df_weekly = compute_weekly_from_daily(df_daily)
            result.weekly_rows = store_weekly_data(df_weekly, symbol, conn)
    finally:
        conn.close()
    return result


if __name__ == "__main__":
    summary = update_data("AAPL")
    print(
        f"Updated {summary.symbol}: "
        f"{summary.daily_rows} daily rows, {summary.weekly_rows} weekly rows."
    )
