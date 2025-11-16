"""Backtesting and validation module for buy/no-buy classification strategy."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.classification_pipeline import (
    CLASSIFIER_METRICS_DIR,
    CLASSIFIER_MODEL_DIR,
    CLASSIFIER_SCALER_DIR,
    CLASSIFIER_THRESHOLD_DIR,
    build_classification_identifier,
    load_classifier_artifacts,
    run_classification_inference,
    get_connection,
    load_feature_data,
)
from scripts.diagnostic_outputs import save_backtest_summary
from scripts.time_splits import load_time_split, TimeSplit

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "stocks.db"


@dataclass
class BacktestConfig:
    """Configuration for the backtest."""

    start_capital: float = 1.0
    transaction_cost: float = 0.0  # Proportional cost per trade (e.g., 0.001 = 0.1%).
    price_column: str = "close"
    execution_price: str = "close"  # Document choice: execute trades at same-day close.
    risk_free_rate: float = 0.0  # Annualized, for Sharpe calculation.
    periods_per_year: int = 252  # Adjust for weekly data.


@dataclass
class BacktestResult:
    """Structured backtest output for documentation/UI."""

    metrics: Dict[str, float]
    benchmark: Dict[str, float]
    equity_curve: pd.DataFrame
    summary: str


def load_test_period_prices(
    symbol: str,
    frequency: str,
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Load the chronologically sorted price data to align with inference outputs."""
    conn = get_connection(db_path)
    try:
        dataset = load_feature_data(symbol, frequency, conn)
    finally:
        conn.close()
    return dataset.data[["date", "close"]].copy()


def align_signals_with_prices(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """Merge price data with classifier outputs on date."""
    merged = pd.merge(prices, signals, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    merged.rename(columns={"decision": "signal"}, inplace=True)
    return merged


def run_strategy_simulation(
    market_data: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Simulate the long-only strategy given aligned data."""
    df = market_data.copy()
    df["position"] = 0
    df["trade"] = 0
    df["portfolio"] = np.nan
    df["cash"] = config.start_capital
    df["holdings"] = 0.0
    df["units"] = 0.0
    df["returns"] = 0.0

    position = 0
    cash = config.start_capital
    units = 0.0
    last_price = None

    for idx, row in df.iterrows():
        price = row[config.price_column]
        signal = row["signal"]

        if position == 0 and signal == 1:
            units = cash / price if price != 0 else 0
            cost = cash * config.transaction_cost
            cash = cash - cost - cash
            holdings = units * price
            portfolio = holdings + cash
            position = 1 if units > 0 else 0
            df.at[idx, "trade"] = 1 if position else 0
        elif position == 1 and signal == 0:
            proceeds = units * price
            cost = proceeds * config.transaction_cost
            cash = cash + proceeds - cost
            holdings = 0.0
            units = 0.0
            portfolio = cash
            position = 0
            df.at[idx, "trade"] = -1
        else:
            holdings = units * price
            portfolio = holdings + cash
        df.at[idx, "position"] = position
        df.at[idx, "cash"] = cash
        df.at[idx, "holdings"] = holdings
        df.at[idx, "portfolio"] = portfolio
        if last_price is not None and df.at[idx - 1, "portfolio"] != 0:
            df.at[idx, "returns"] = (portfolio / df.at[idx - 1, "portfolio"]) - 1
        last_price = price

    df["portfolio"] = df["portfolio"].fillna(method="ffill").fillna(config.start_capital)
    df["returns"] = df["returns"].fillna(0.0)
    return df


def compute_performance_metrics(
    equity_curve: pd.DataFrame,
    config: BacktestConfig,
) -> Dict[str, float]:
    """Calculate key performance indicators from the equity curve."""
    returns = equity_curve["returns"]
    total_return = equity_curve["portfolio"].iloc[-1] / equity_curve["portfolio"].iloc[0] - 1
    periods = len(equity_curve)
    annualized_return = (1 + total_return) ** (config.periods_per_year / max(periods, 1)) - 1 if periods > 0 else 0
    volatility = returns.std() * np.sqrt(config.periods_per_year)
    sharpe = (
        (returns.mean() * config.periods_per_year - config.risk_free_rate)
        / (volatility if volatility else np.nan)
    )
    cumulative = equity_curve["portfolio"]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    trades = equity_curve.loc[equity_curve["trade"] != 0]
    wins = trades[trades["returns"] > 0]
    losses = trades[trades["returns"] < 0]
    win_rate = len(wins) / len(trades) if len(trades) else np.nan
    avg_gain = wins["returns"].mean() if len(wins) else 0.0
    avg_loss = losses["returns"].mean() if len(losses) else 0.0
    avg_holding = (
        equity_curve["position"].diff().ne(0).cumsum().value_counts().mean() if not equity_curve.empty else np.nan
    )

    signals = equity_curve["signal"].sum()
    metrics = {
        "final_portfolio": float(equity_curve["portfolio"].iloc[-1]),
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate) if win_rate == win_rate else np.nan,
        "avg_gain": float(avg_gain),
        "avg_loss": float(avg_loss),
        "buy_signals": int(signals),
        "trades": int(len(trades)),
        "avg_holding_period": float(avg_holding) if avg_holding == avg_holding else np.nan,
    }
    return metrics


def compute_benchmark_metrics(
    prices: pd.Series,
    config: BacktestConfig,
) -> Dict[str, float]:
    """Compute buy-and-hold metrics for comparison."""
    returns = prices.pct_change().dropna()
    total_return = prices.iloc[-1] / prices.iloc[0] - 1
    annualized_return = (1 + total_return) ** (config.periods_per_year / len(prices)) - 1 if len(prices) else 0
    volatility = returns.std() * np.sqrt(config.periods_per_year)
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() if not drawdown.empty else 0
    return {
        "benchmark_total_return": float(total_return),
        "benchmark_annualized_return": float(annualized_return),
        "benchmark_volatility": float(volatility),
        "benchmark_max_drawdown": float(max_drawdown),
    }


def equity_curves_for_plotting(
    backtest_df: pd.DataFrame,
    prices: pd.Series,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Build a DataFrame suitable for plotting both strategy and benchmark curves."""
    benchmark_equity = (prices / prices.iloc[0]) * config.start_capital
    return pd.DataFrame(
        {
            "date": backtest_df["date"],
            "strategy_equity": backtest_df["portfolio"],
            "benchmark_equity": benchmark_equity.reindex(backtest_df.index).fillna(method="ffill"),
            "position": backtest_df["position"],
            "price": prices.reindex(backtest_df.index),
        }
    )


def summarize_backtest(metrics: Dict[str, float], benchmark: Dict[str, float]) -> str:
    """Generate a human-readable summary."""
    summary = [
        f"Final Portfolio Value: {metrics['final_portfolio']:.2f}",
        f"Total Return: {metrics['total_return']*100:.2f}%",
        f"Annualized Return: {metrics['annualized_return']*100:.2f}%",
        f"Volatility: {metrics['volatility']*100:.2f}%",
        f"Sharpe Ratio: {metrics['sharpe']:.2f}",
        f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%",
        f"Win Rate: {metrics['win_rate']*100 if metrics['win_rate']==metrics['win_rate'] else np.nan:.2f}%",
        f"Average Gain: {metrics['avg_gain']*100:.2f}%",
        f"Average Loss: {metrics['avg_loss']*100:.2f}%",
        f"Buy Signals: {metrics['buy_signals']}",
        f"Trades Executed: {metrics['trades']}",
        f"Average Holding Period: {metrics['avg_holding_period']:.2f}",
        "--- Benchmark ---",
        f"Buy & Hold Total Return: {benchmark['benchmark_total_return']*100:.2f}%",
        f"Buy & Hold Volatility: {benchmark['benchmark_volatility']*100:.2f}%",
        f"Buy & Hold Max Drawdown: {benchmark['benchmark_max_drawdown']*100:.2f}%",
    ]
    return "\n".join(summary)


def run_backtest(
    symbol: str,
    *,
    frequency: str = "daily",
    threshold: float = 0.02,
    window: Tuple[int, int] = (1, 5),
    regression_model_type: str = "ridge",
    config: Optional[BacktestConfig] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> BacktestResult | None:
    """Top-level orchestration for backtesting the classifier strategy."""
    config = config or BacktestConfig()
    prices = load_test_period_prices(symbol, frequency, db_path=db_path)
    if prices.empty:
        message = f"No price data for {symbol.upper()} ({frequency})."
        logger.warning(message)
        print(message)
        return None

    split_def = load_time_split(symbol, frequency)
    if split_def is None:
        message = "Time split metadata missing. Run regression training first."
        logger.warning(message)
        print(message)
        return None
    identifier = build_classification_identifier(symbol, frequency, threshold, window)
    try:
        _ = load_classifier_artifacts(identifier)
    except FileNotFoundError:
        message = "Classifier artifacts not found. Train classifier first."
        logger.warning(message)
        print(message)
        return None

    signals = run_classification_inference(
        symbol,
        frequency=frequency,
        threshold=threshold,
        window=window,
        regression_model_type=regression_model_type,
        split=split_def,
        db_path=db_path,
    )
    if signals.empty:
        message = "No signals generated for backtest."
        logger.warning(message)
        print(message)
        return None

    market_data = align_signals_with_prices(prices, signals)
    market_data = split_def.select(market_data, "test").reset_index(drop=True)
    if market_data.empty:
        message = "No overlapping data in test period for backtest."
        logger.warning(message)
        print(message)
        return None
    backtest_df = run_strategy_simulation(market_data, config)
    metrics = compute_performance_metrics(backtest_df, config)
    benchmark = compute_benchmark_metrics(market_data["close"], config)
    equity = equity_curves_for_plotting(backtest_df, market_data["close"], config)
    summary = summarize_backtest(metrics, benchmark)
    artifacts = [
        {"type": "model", "description": "classifier_model", "path": CLASSIFIER_MODEL_DIR / f"{identifier}.joblib"},
        {"type": "scaler", "description": "classifier_scaler", "path": CLASSIFIER_SCALER_DIR / f"{identifier}.joblib"},
        {"type": "table", "description": "classifier_threshold", "path": CLASSIFIER_THRESHOLD_DIR / f"{identifier}.json"},
        {"type": "table", "description": "classifier_metrics", "path": CLASSIFIER_METRICS_DIR / f"{identifier}.json"},
    ]
    equity_records: List[Dict[str, float | str]] = []
    for _, row in equity.iterrows():
        date_value = row["date"]
        if hasattr(date_value, "isoformat"):
            date_str = date_value.isoformat()
        else:
            date_str = str(date_value)
        equity_records.append(
            {
                "date": date_str,
                "strategy_equity": float(row["strategy_equity"]),
                "benchmark_equity": float(row["benchmark_equity"]),
                "position": float(row["position"]),
                "price": float(row["price"]),
            }
        )
    insight_flags: List[str] = []
    total_return = metrics.get("total_return")
    if total_return is not None and total_return < 0:
        insight_flags.append("negative_total_return")
    sharpe = metrics.get("sharpe")
    if sharpe is not None and sharpe < 1:
        insight_flags.append("low_sharpe_ratio")
    max_drawdown = metrics.get("max_drawdown")
    if max_drawdown is not None and max_drawdown < -0.3:
        insight_flags.append("deep_drawdown")
    save_backtest_summary(
        symbol=symbol,
        frequency=frequency,
        identifier=identifier,
        metrics=metrics,
        benchmark=benchmark,
        equity_curve=equity_records,
        insight_flags=insight_flags,
        artifacts=artifacts,
    )
    logger.info("Backtest summary:\n%s", summary)
    print(summary)
    return BacktestResult(metrics=metrics, benchmark=benchmark, equity_curve=equity, summary=summary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_backtest("AAPL", frequency="daily")
