"""Dash UI for the On-The-Go stock prediction pipeline."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from scripts.data_pipeline import update_data
from scripts.feature_engineering import (
    DEFAULT_DB_PATH as DATA_DB_PATH,
    generate_features,
    get_connection as feature_db_connection,
    load_price_data,
)
from scripts.classification_pipeline import (
    CLASSIFIER_METRICS_DIR,
    CLASSIFIER_MODEL_DIR,
    CLASSIFIER_SCALER_DIR,
    CLASSIFIER_THRESHOLD_DIR,
    build_classification_identifier,
    load_classifier_artifacts,
    run_classification_inference,
)
from scripts.regression_training import REGRESSION_MODEL_DIR
from scripts.backtesting import run_backtest
from scripts.diagnostics import run_diagnostics
from scripts.targets import get_horizon_steps, ordered_target_names
from scripts.time_splits import load_time_split


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("dash_app")


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REGRESSION_MODEL_DIR.parent
SPLITS_DIR = MODELS_DIR / "splits"
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN"]
CACHE_SENTINEL = "static"
MAX_RESIDUAL_POINTS = 200
MAX_FEATURE_ROWS = 15


@lru_cache(maxsize=1)
def discover_symbols() -> List[str]:
    """Infer available symbols from stored split definitions."""
    symbols: set[str] = set()
    if SPLITS_DIR.exists():
        for freq_dir in SPLITS_DIR.iterdir():
            if not freq_dir.is_dir():
                continue
            for path in freq_dir.glob("*.json"):
                symbols.add(path.stem.upper())
    if not symbols:
        symbols.update(DEFAULT_SYMBOLS)
    return sorted(symbols)


@lru_cache(maxsize=32)
def load_price_history_cached(
    symbol: str,
    frequency: str,
    lookback: int = 180,
    cache_key: str = CACHE_SENTINEL,
) -> pd.DataFrame:
    """Return most recent price history for plotting."""
    conn = feature_db_connection(DATA_DB_PATH)
    try:
        df = load_price_data(symbol, frequency, conn)
    finally:
        conn.close()
    if df.empty:
        return pd.DataFrame(columns=["date", "close"])
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(lookback).reset_index(drop=True)
    return df


def _format_timestamp(ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _maybe_float(value: object) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=64)
def load_pipeline_status_cached(
    symbol: str,
    frequency: str,
    threshold: float,
    window_start: int,
    window_end: int,
    cache_key: str = CACHE_SENTINEL,
) -> Dict[str, object]:
    """Summarize available artifacts for the overview page."""
    regression_dir = REGRESSION_MODEL_DIR / frequency
    pattern = f"{symbol.upper()}_{frequency}_*_*.joblib"
    reg_paths = list(regression_dir.glob(pattern)) if regression_dir.exists() else []
    regression_last = max((path.stat().st_mtime for path in reg_paths), default=None)

    window_tuple = (window_start, window_end)
    identifier = build_classification_identifier(symbol, frequency, threshold, window_tuple)
    clf_model = CLASSIFIER_MODEL_DIR / f"{identifier}.joblib"
    clf_scaler = CLASSIFIER_SCALER_DIR / f"{identifier}.joblib"
    clf_threshold = CLASSIFIER_THRESHOLD_DIR / f"{identifier}.json"
    clf_metrics = CLASSIFIER_METRICS_DIR / f"{identifier}.json"
    clf_files = [clf_model, clf_scaler, clf_threshold, clf_metrics]
    clf_exists = all(path.exists() for path in clf_files)
    clf_last = max((path.stat().st_mtime for path in clf_files if path.exists()), default=None)

    split = load_time_split(symbol, frequency)
    training_dates: Dict[str, str] = {}
    if clf_metrics.exists():
        try:
            with clf_metrics.open("r", encoding="utf-8") as fh:
                metrics_data = json.load(fh)
            training_dates["classification_metrics"] = metrics_data.get("trained_at", "—")
        except json.JSONDecodeError:
            training_dates["classification_metrics"] = "—"

    return {
        "regression_available": bool(reg_paths),
        "regression_last_trained": _format_timestamp(regression_last),
        "classification_available": clf_exists,
        "classification_last_trained": _format_timestamp(clf_last),
        "classification_identifier": identifier,
        "split_available": split is not None,
        "split": split.to_dict() if split else None,
        "training_dates": training_dates,
    }


def price_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Compute key price stats for overview & forecast."""
    if df.empty:
        return {"last_price": None, "last_date": None, "return_30d": None}
    last_price = float(df["close"].iloc[-1])
    last_date = df["date"].iloc[-1]
    rolling = df.tail(30)
    change = float(rolling["close"].iloc[-1] / rolling["close"].iloc[0] - 1) if len(rolling) > 1 else None
    return {
        "last_price": last_price,
        "last_date": last_date,
        "return_30d": change,
    }


def build_input_data_chart(price_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if price_df.empty:
        fig.update_layout(title="No price data available", margin=dict(l=10, r=10, t=40, b=10))
        return fig
    df = price_df.copy()
    df = df.sort_values("date")
    for window in (20, 50, 100):
        df[f"ma_{window}"] = df["close"].rolling(window=window).mean()
    rolling_std = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["ma_20"] + 2 * rolling_std
    df["bb_lower"] = df["ma_20"] - 2 * rolling_std
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma_20"], mode="lines", name="MA 20", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma_50"], mode="lines", name="MA 50", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma_100"], mode="lines", name="MA 100", line=dict(dash="dashdot")))
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["bb_upper"],
            mode="lines",
            name="Bollinger Upper",
            line=dict(color="rgba(31,119,180,0.2)"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["bb_lower"],
            mode="lines",
            name="Bollinger Lower",
            fill="tonexty",
            fillcolor="rgba(31,119,180,0.1)",
            line=dict(color="rgba(31,119,180,0.2)"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["volume"],
            name="Volume",
            marker=dict(color="rgba(128,128,128,0.3)"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        title="Price & Technical Context",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2=dict(
            overlaying="y",
            side="right",
            title="Volume",
            showgrid=False,
        ),
        legend=dict(orientation="h"),
    )
    return fig


def format_percent(value: Optional[float], digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def format_currency(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"${float(value):,.2f}"


def format_number(value: Optional[float], digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def render_nav_link(path: str, label: str) -> html.Div:
    return html.Div(
        dcc.Link(label, href=path, className="sidebar-link"),
        className="mb-2",
    )


def make_card(title: str, value: str, body: Optional[str] = None) -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(title),
            dbc.CardBody(
                [
                    html.H4(value, className="card-title"),
                    html.P(body, className="card-text") if body else None,
                ]
            ),
        ],
        className="mb-3",
    )


def layout_sidebar(initial_symbol: Optional[str]) -> html.Div:
    symbol_value = initial_symbol or (discover_symbols()[0] if discover_symbols() else None)
    return html.Div(
        [
            html.H3("Stock Pipeline", className="mb-3"),
            html.P("Select a symbol, configure controls, and launch actions.", className="text-muted"),
            html.Label("Symbol", className="fw-bold"),
            dcc.Dropdown(
                id="symbol-dropdown",
                options=[{"label": sym, "value": sym} for sym in discover_symbols()],
                value=symbol_value,
                clearable=False,
            ),
            html.Label("Frequency", className="mt-3 fw-bold"),
            dcc.RadioItems(
                id="frequency-radio",
                options=[
                    {"label": "Daily", "value": "daily"},
                    {"label": "Weekly", "value": "weekly"},
                ],
                value="daily",
                labelStyle={"display": "block"},
            ),
            html.Label("Target Gain Threshold", className="mt-3 fw-bold"),
            dcc.Slider(
                id="threshold-slider",
                min=0.005,
                max=0.1,
                step=0.005,
                marks={0.02: "2%", 0.05: "5%", 0.1: "10%"},
                value=0.02,
            ),
            html.Div(id="threshold-display", className="text-muted small"),
            html.Label("Forecast Window (bars)", className="mt-3 fw-bold"),
            dcc.RangeSlider(
                id="window-slider",
                min=1,
                max=10,
                step=1,
                value=[1, 5],
                marks={1: "1", 5: "5", 10: "10"},
                allowCross=False,
            ),
            html.Div(id="window-display", className="text-muted small"),
            html.Button("Run Forecast", id="run-forecast-btn", className="btn btn-primary w-100 mt-3"),
            html.Div(id="forecast-status", className="text-muted small mt-1"),
            html.Button("Run Backtest (Test Period)", id="run-backtest-btn", className="btn btn-secondary w-100 mt-3"),
            html.Div(id="backtest-status", className="text-muted small mt-1"),
            html.Button("Run Diagnostics", id="run-diagnostics-btn", className="btn btn-info w-100 mt-3"),
            html.Div(id="diagnostics-status", className="text-muted small mt-1"),
            html.Button(
                "Refresh Data & Features",
                id="refresh-data-btn",
                className="btn btn-outline-dark w-100 mt-3",
            ),
            html.Div(id="refresh-status", className="text-muted small mt-1"),
            html.Hr(),
            html.H5("Navigate", className="mt-3"),
            render_nav_link("/overview", "Overview"),
            render_nav_link("/forecast", "Forecast / Signals"),
            render_nav_link("/backtest", "Backtest & Performance"),
            render_nav_link("/diagnostics", "Model Diagnostics"),
            render_nav_link("/settings", "Settings / Info"),
        ],
        style={
            "padding": "1.5rem",
            "backgroundColor": "#f8f9fa",
            "height": "100vh",
            "overflow": "auto",
            "width": "320px",
            "flex": "0 0 320px",
        },
    )


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "Stock Pipeline Console"
server = app.server


app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dcc.Store(id="forecast-store"),
        dcc.Store(id="backtest-store"),
        dcc.Store(id="diagnostics-store"),
        dcc.Store(id="data-refresh-token"),
        layout_sidebar(discover_symbols()[0] if discover_symbols() else None),
        html.Div(
            dcc.Loading(
                id="page-loading",
                type="default",
                children=html.Div(id="page-content", style={"padding": "2rem"}),
            ),
            style={"flex": "1", "overflow": "auto", "height": "100vh"},
        ),
    ],
    style={"display": "flex", "height": "100vh", "fontFamily": "Segoe UI, Arial, sans-serif"},
)


def render_overview_page(
    symbol: str,
    frequency: str,
    threshold: float,
    window_tuple: Tuple[int, int],
    price_df: pd.DataFrame,
    status: Dict[str, object],
) -> html.Div:
    summary = price_summary(price_df)
    last_price = format_currency(summary["last_price"])
    last_date = summary["last_date"].strftime("%Y-%m-%d") if summary["last_date"] is not None else "—"
    last_return = format_percent(summary["return_30d"])
    status_cards = [
        make_card("Last Close", last_price, f"Date: {last_date}"),
        make_card("30-Day Return", last_return),
        make_card("Target Gain", format_percent(threshold), f"Window: {window_tuple[0]}–{window_tuple[1]} bars"),
    ]
    pipeline_cards = [
        make_card(
            "Regression Models",
            "Available" if status["regression_available"] else "Missing",
            f"Last trained: {status['regression_last_trained']}",
        ),
        make_card(
            "Classification Model",
            "Available" if status["classification_available"] else "Missing",
            f"Last trained: {status['classification_last_trained']}",
        ),
        make_card(
            "Time Split",
            "Configured" if status["split_available"] else "Missing",
            json.dumps(status["split"], indent=2) if status["split"] else "Run regression training to create splits.",
        ),
    ]
    overview_chart = dcc.Graph(figure=build_input_data_chart(price_df), id="input-data-chart")
    explanation = dcc.Markdown(
        """
        **What the model predicts**

        - Regression models forecast forward returns across configured horizons (1d, 3d, 5d, 1w, etc.).
        - The classifier converts those returns into the probability of reaching the target gain inside the selected window.
        - BUY signals indicate the stored probability threshold has been exceeded. These signals remain *hypothetical* and are **not financial advice**.
        """,
        className="mt-3",
    )
    return html.Div(
        [
            html.H2(f"Overview: {symbol.upper()} ({frequency})"),
            dbc.Row([dbc.Col(card) for card in status_cards]),
            html.H3("Pipeline Status", className="mt-4"),
            dbc.Row([dbc.Col(card) for card in pipeline_cards]),
            html.H3("Recent Input Data", className="mt-4"),
            overview_chart,
            explanation,
        ]
    )


def build_forecast_table_rows(
    latest_row: pd.Series,
    frequency: str,
    last_price: Optional[float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    steps_map = get_horizon_steps(frequency)
    for horizon, _ in steps_map.items():
        col = f"pred_{horizon}"
        value = latest_row.get(col)
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        predicted_return = float(value)
        predicted_price = last_price * (1 + predicted_return) if last_price is not None else None
        rows.append(
            {
                "key": horizon,
                "label": horizon.replace("target_", "").upper(),
                "pred_return": predicted_return,
                "pred_price": predicted_price,
            }
        )
    return rows


def build_forecast_chart(
    price_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    forecast_rows: List[Dict[str, object]],
    frequency: str,
) -> go.Figure:
    fig = go.Figure()
    if not price_df.empty:
        fig.add_trace(
            go.Scatter(
                x=price_df["date"],
                y=price_df["close"],
                mode="lines",
                name="Close",
            )
        )
    if not forecast_df.empty:
        merged = pd.merge(price_df, forecast_df[["date", "prob_buy", "decision"]], on="date", how="left")
        buy_mask = merged["decision"] == 1
        fig.add_trace(
            go.Scatter(
                x=merged.loc[buy_mask, "date"],
                y=merged.loc[buy_mask, "close"],
                mode="markers",
                marker=dict(color="green", size=10, symbol="triangle-up"),
                name="Buy signals",
                hovertemplate="Date=%{x}<br>Price=%{y:.2f}",
            )
        )
    if forecast_rows and not price_df.empty:
        base_date = forecast_df["date"].iloc[-1]
        base_price = price_df["close"].iloc[-1]
        steps_map = get_horizon_steps(frequency)
        future_dates = []
        future_prices = []
        for row in forecast_rows:
            steps = steps_map.get(row["key"], 1)
            days = steps if frequency == "daily" else steps * 7
            future_dates.append(base_date + pd.Timedelta(days=days))
            projected = row["pred_price"] if row["pred_price"] is not None else base_price
            future_prices.append(projected)
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_prices,
                mode="lines+markers",
                name="Projected price",
                line=dict(dash="dot"),
            )
        )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        title="Recent Prices & Signals",
        xaxis_title="Date",
        yaxis_title="Price",
    )
    return fig


def render_forecast_page(
    symbol: str,
    frequency: str,
    threshold: float,
    window_tuple: Tuple[int, int],
    price_df: pd.DataFrame,
    forecast_data: Optional[Dict[str, object]],
) -> html.Div:
    summary = price_summary(price_df)
    if not forecast_data or forecast_data.get("symbol") != symbol or forecast_data.get("frequency") != frequency:
        return html.Div(
            [
                html.H2("Forecast / Signals"),
                html.P("Run a forecast from the sidebar after selecting a symbol and settings."),
            ]
        )
    records = forecast_data.get("records", [])
    df = pd.DataFrame(records)
    if df.empty:
        return html.Div([html.H2("Forecast / Signals"), html.P("No forecast rows returned from the pipeline.")])
    df["date"] = pd.to_datetime(df["date"])
    latest_row = df.iloc[-1]
    rows = build_forecast_table_rows(latest_row, frequency, summary["last_price"])
    table = dbc.Table(
        [
            html.Thead(
                html.Tr([html.Th("Horizon"), html.Th("Predicted Return"), html.Th("Projected Price")])
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(row["label"]),
                            html.Td(format_percent(row["pred_return"])),
                            html.Td(format_currency(row["pred_price"])),
                        ]
                    )
                    for row in rows
                ]
            ),
        ],
        bordered=True,
        hover=True,
        size="sm",
    )
    probability = format_percent(_maybe_float(latest_row.get("prob_buy")))
    decision = "BUY" if latest_row.get("decision") == 1 else "NO BUY"
    prob_threshold = format_percent(forecast_data.get("prob_threshold"))
    chart = dcc.Graph(figure=build_forecast_chart(price_df, df, rows, frequency), id="forecast-graph")
    summary_cards = dbc.Row(
        [
            dbc.Col(make_card("Probability", probability, f"Threshold: {prob_threshold}"), md=4),
            dbc.Col(make_card("Decision", decision, f"Generated at {forecast_data.get('generated_at', '—')}"), md=4),
            dbc.Col(
                make_card(
                    "Target Window",
                    f"{window_tuple[0]}–{window_tuple[1]} bars",
                    f"Gain threshold: {format_percent(threshold)}",
                ),
                md=4,
            ),
        ]
    )
    return html.Div(
        [
            html.H2(f"Forecast & Signals — {symbol.upper()} ({frequency})"),
            summary_cards,
            html.H3("Per-Horizon Forecasts", className="mt-4"),
            table,
            chart,
        ]
    )


def build_backtest_chart(equity_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if equity_df.empty:
        return fig
    fig.add_trace(go.Scatter(x=equity_df["date"], y=equity_df["strategy_equity"], mode="lines", name="Strategy"))
    fig.add_trace(
        go.Scatter(
            x=equity_df["date"],
            y=equity_df["benchmark_equity"],
            mode="lines",
            name="Buy & Hold",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        title="Equity Curves (Test Period)",
        xaxis_title="Date",
        yaxis_title="Equity",
    )
    return fig


def extract_trades(equity_df: pd.DataFrame) -> List[Dict[str, object]]:
    trades: List[Dict[str, object]] = []
    in_position = False
    entry_date = None
    entry_price = None
    for _, row in equity_df.iterrows():
        position = row.get("position", 0)
        date = row["date"]
        price = row.get("price")
        if not in_position and position >= 1:
            in_position = True
            entry_date = date
            entry_price = price
        elif in_position and position < 1:
            exit_price = price
            ret = (exit_price / entry_price - 1) if entry_price else None
            trades.append(
                {
                    "entry": entry_date,
                    "exit": date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return": ret,
                }
            )
            in_position = False
    return trades


def render_backtest_page(
    symbol: str,
    frequency: str,
    backtest_data: Optional[Dict[str, object]],
) -> html.Div:
    if not backtest_data or backtest_data.get("symbol") != symbol or backtest_data.get("frequency") != frequency:
        return html.Div(
            [
                html.H2("Backtest & Performance"),
                html.P("Run a backtest from the sidebar to populate this page."),
            ]
        )
    metrics = backtest_data.get("metrics", {})
    benchmark = backtest_data.get("benchmark", {})
    equity = pd.DataFrame(backtest_data.get("equity_curve", []))
    if not equity.empty:
        equity["date"] = pd.to_datetime(equity["date"])
    cards = dbc.Row(
        [
            dbc.Col(make_card("Strategy Total Return", format_percent(metrics.get("total_return"))), md=4),
            dbc.Col(make_card("Strategy Sharpe", format_number(metrics.get("sharpe"))), md=4),
            dbc.Col(make_card("Win Rate", format_percent(metrics.get("win_rate"))), md=4),
            dbc.Col(make_card("Benchmark Return", format_percent(benchmark.get("benchmark_total_return"))), md=4),
            dbc.Col(make_card("Trades", str(metrics.get("trades", "—"))), md=4),
            dbc.Col(make_card("Max Drawdown", format_percent(metrics.get("max_drawdown"))), md=4),
        ]
    )
    trades = extract_trades(equity)[:10] if not equity.empty else []
    trades_table = (
        dbc.Table(
            [
                html.Thead(
                    html.Tr([html.Th("Entry"), html.Th("Exit"), html.Th("Entry Price"), html.Th("Exit Price"), html.Th("Return")])
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(trade["entry"]),
                                html.Td(trade["exit"]),
                                html.Td(format_currency(trade["entry_price"])),
                                html.Td(format_currency(trade["exit_price"])),
                                html.Td(format_percent(trade["return"])),
                            ]
                        )
                        for trade in trades
                    ]
                ),
            ],
            bordered=True,
            size="sm",
        )
        if trades
        else html.Div("No completed trades captured in the equity curve.")
    )
    figure = dcc.Graph(figure=build_backtest_chart(equity))
    return html.Div(
        [
            html.H2(f"Backtest Results — {symbol.upper()} ({frequency})"),
            cards,
            figure,
            html.H3("Recent Trades", className="mt-4"),
            trades_table,
        ]
    )


def render_diagnostics_page() -> html.Div:
    return html.Div(
        [
            html.H2("Model Diagnostics"),
            html.P("Use the sidebar button to compute diagnostics. Results populate below when available."),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Regression Horizon"),
                            dcc.Dropdown(id="diagnostics-horizon-dropdown", options=[], value=None, clearable=False),
                        ],
                        md=4,
                    ),
                ]
            ),
            html.Div(id="diagnostics-classification-summary", className="mt-3"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="diagnostics-residuals-chart"), md=6),
                    dbc.Col(dcc.Graph(id="diagnostics-regression-feature-chart"), md=6),
                ],
                className="mt-3",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="diagnostics-cross-horizon-chart"), md=6),
                    dbc.Col(dcc.Graph(id="diagnostics-classification-feature-chart"), md=6),
                ],
                className="mt-3",
            ),
            html.Div(id="diagnostics-messages", className="text-muted mt-3"),
        ]
    )


def render_settings_page() -> html.Div:
    return html.Div(
        [
            html.H2("Settings & Info"),
            html.Ul(
                [
                    html.Li(f"Database: {DATA_DB_PATH}"),
                    html.Li(f"Models directory: {MODELS_DIR}"),
                    html.Li("Use scripts/orchestrator.py to run full retraining pipelines."),
                    html.Li("Dash UI focuses on inference, backtesting, and diagnostics."),
                ]
            ),
        ]
    )


def _serialize_dataframe(df: pd.DataFrame, limit: Optional[int] = None) -> List[Dict[str, object]]:
    if df is None or df.empty:
        return []
    subset = df.tail(limit) if limit else df
    subset = subset.copy()
    if "date" in subset.columns:
        subset["date"] = subset["date"].apply(lambda v: v.isoformat() if hasattr(v, "isoformat") else str(v))
    return subset.replace({np.nan: None}).to_dict("records")


def _normalize_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)):
            normalized[key] = float(value)
        else:
            normalized[key] = value
    return normalized


def _serialize_diagnostics(bundle) -> Dict[str, object]:
    regression = {}
    for horizon, result in (bundle.regression or {}).items():
        regression[horizon] = {
            "metrics": result.metrics,
            "residuals": _serialize_dataframe(result.residuals, MAX_RESIDUAL_POINTS),
            "feature_importance": _serialize_dataframe(result.feature_importance, MAX_FEATURE_ROWS),
        }
    classification = {
        "metrics": bundle.classification.metrics if bundle.classification else {},
        "feature_importance": _serialize_dataframe(
            bundle.classification.feature_importance if bundle.classification else pd.DataFrame(),
            MAX_FEATURE_ROWS,
        ),
    }
    cross = {
        "consistency": bundle.cross_model.consistency_metrics if bundle.cross_model else {},
        "horizon_summary": _serialize_dataframe(
            bundle.cross_model.horizon_summary if bundle.cross_model else pd.DataFrame()
        ),
    }
    return {
        "regression": regression,
        "classification": classification,
        "cross": cross,
        "generated_at": datetime.utcnow().isoformat(),
    }


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("symbol-dropdown", "value"),
    Input("frequency-radio", "value"),
    Input("threshold-slider", "value"),
    Input("window-slider", "value"),
    State("forecast-store", "data"),
    State("backtest-store", "data"),
    State("data-refresh-token", "data"),
)
def render_page(
    pathname: Optional[str],
    symbol: Optional[str],
    frequency: str,
    threshold: float,
    window_range: Sequence[int],
    forecast_data,
    backtest_data,
    refresh_token,
):
    symbol = symbol or (discover_symbols()[0] if discover_symbols() else "AAPL")
    window_tuple = (int(window_range[0]), int(window_range[1]))
    cache_key = str(refresh_token or CACHE_SENTINEL)
    price_df = load_price_history_cached(symbol, frequency, cache_key=cache_key)
    status = load_pipeline_status_cached(symbol, frequency, threshold, window_tuple[0], window_tuple[1], cache_key=cache_key)
    pathname = pathname or "/overview"
    if pathname == "/overview":
        return render_overview_page(symbol, frequency, threshold, window_tuple, price_df, status)
    if pathname == "/forecast":
        return render_forecast_page(symbol, frequency, threshold, window_tuple, price_df, forecast_data)
    if pathname == "/backtest":
        return render_backtest_page(symbol, frequency, backtest_data)
    if pathname == "/diagnostics":
        return render_diagnostics_page()
    if pathname == "/settings":
        return render_settings_page()
    return render_overview_page(symbol, frequency, threshold, window_tuple, price_df, status)


@app.callback(
    Output("threshold-display", "children"),
    Input("threshold-slider", "value"),
)
def update_threshold_display(value: float) -> str:
    return f"{value*100:.1f}% target gain"


@app.callback(
    Output("window-display", "children"),
    Input("window-slider", "value"),
)
def update_window_display(value: Sequence[int]) -> str:
    start, end = value
    return f"{start} → {end} bars"


@app.callback(
    Output("symbol-dropdown", "options"),
    Input("data-refresh-token", "data"),
)
def update_symbol_options(_: Optional[str]):
    return [{"label": sym, "value": sym} for sym in discover_symbols()]


@app.callback(
    Output("symbol-dropdown", "value"),
    Input("symbol-dropdown", "options"),
    State("symbol-dropdown", "value"),
)
def ensure_symbol_value(options, value):
    if not options:
        return value
    labels = [opt["value"] for opt in options]
    if value in labels:
        return value
    return labels[0]


@app.callback(
    Output("forecast-store", "data"),
    Output("forecast-status", "children"),
    Input("run-forecast-btn", "n_clicks"),
    State("symbol-dropdown", "value"),
    State("frequency-radio", "value"),
    State("threshold-slider", "value"),
    State("window-slider", "value"),
    prevent_initial_call=True,
)
def trigger_forecast(n_clicks, symbol, frequency, threshold, window_range):
    if not n_clicks:
        raise PreventUpdate
    window_tuple = (int(window_range[0]), int(window_range[1]))
    try:
        inference_df = run_classification_inference(
            symbol,
            frequency=frequency,
            threshold=threshold,
            window=window_tuple,
        )
        identifier = build_classification_identifier(symbol, frequency, threshold, window_tuple)
        _, _, prob_threshold = load_classifier_artifacts(identifier)
    except FileNotFoundError:
        message = html.Span(
            "Models missing for this configuration. Train models via CLI first.",
            className="text-danger",
        )
        return no_update, message
    except Exception as exc:
        LOGGER.exception("Forecast failed: %s", exc)
        return no_update, html.Span(f"Forecast failed: {exc}", className="text-danger")
    data = {
        "symbol": symbol,
        "frequency": frequency,
        "threshold": threshold,
        "window": window_tuple,
        "prob_threshold": prob_threshold,
        "generated_at": datetime.utcnow().isoformat(),
        "records": _serialize_dataframe(inference_df),
    }
    return data, html.Span("Forecast complete.", className="text-success")


@app.callback(
    Output("backtest-store", "data"),
    Output("backtest-status", "children"),
    Input("run-backtest-btn", "n_clicks"),
    State("symbol-dropdown", "value"),
    State("frequency-radio", "value"),
    State("threshold-slider", "value"),
    State("window-slider", "value"),
    prevent_initial_call=True,
)
def trigger_backtest(n_clicks, symbol, frequency, threshold, window_range):
    if not n_clicks:
        raise PreventUpdate
    window_tuple = (int(window_range[0]), int(window_range[1]))
    try:
        result = run_backtest(symbol, frequency=frequency, threshold=threshold, window=window_tuple)
    except Exception as exc:
        LOGGER.exception("Backtest failed: %s", exc)
        return no_update, html.Span(f"Backtest failed: {exc}", className="text-danger")
    if result is None:
        return no_update, html.Span("Backtest could not run (missing data or models).", className="text-warning")
    equity_records = []
    if not result.equity_curve.empty:
        frame = result.equity_curve.copy()
        frame["date"] = frame["date"].apply(lambda v: v.isoformat() if hasattr(v, "isoformat") else str(v))
        equity_records = frame.replace({np.nan: None}).to_dict("records")
    data = {
        "symbol": symbol,
        "frequency": frequency,
        "threshold": threshold,
        "window": window_tuple,
        "generated_at": datetime.utcnow().isoformat(),
        "metrics": _normalize_metrics(result.metrics),
        "benchmark": _normalize_metrics(result.benchmark),
        "equity_curve": equity_records,
    }
    return data, html.Span("Backtest complete.", className="text-success")


@app.callback(
    Output("diagnostics-store", "data"),
    Output("diagnostics-status", "children"),
    Input("run-diagnostics-btn", "n_clicks"),
    State("symbol-dropdown", "value"),
    State("frequency-radio", "value"),
    State("threshold-slider", "value"),
    State("window-slider", "value"),
    prevent_initial_call=True,
)
def trigger_diagnostics(n_clicks, symbol, frequency, threshold, window_range):
    if not n_clicks:
        raise PreventUpdate
    window_tuple = (int(window_range[0]), int(window_range[1]))
    try:
        bundle = run_diagnostics(
            symbol,
            frequency=frequency,
            classification_threshold=threshold,
            window=window_tuple,
        )
    except Exception as exc:
        LOGGER.exception("Diagnostics failed: %s", exc)
        return no_update, html.Span(f"Diagnostics failed: {exc}", className="text-danger")
    if bundle is None:
        return no_update, html.Span("Diagnostics unavailable (missing data).", className="text-warning")
    data = _serialize_diagnostics(bundle)
    data.update({"symbol": symbol, "frequency": frequency})
    return data, html.Span("Diagnostics complete.", className="text-success")


@app.callback(
    Output("diagnostics-horizon-dropdown", "options"),
    Output("diagnostics-horizon-dropdown", "value"),
    Input("diagnostics-store", "data"),
)
def populate_diagnostics_horizons(data):
    if not data:
        return [], None
    regression = data.get("regression", {})
    if not regression:
        return [], None
    ordered = [name for name in ordered_target_names() if name in regression]
    remaining = [name for name in regression if name not in ordered]
    horizons = ordered + sorted(remaining)
    options = [{"label": name.replace("target_", "").upper(), "value": name} for name in horizons]
    value = options[0]["value"] if options else None
    return options, value


@app.callback(
    Output("diagnostics-residuals-chart", "figure"),
    Output("diagnostics-regression-feature-chart", "figure"),
    Output("diagnostics-classification-summary", "children"),
    Output("diagnostics-cross-horizon-chart", "figure"),
    Output("diagnostics-classification-feature-chart", "figure"),
    Output("diagnostics-messages", "children"),
    Input("diagnostics-horizon-dropdown", "value"),
    State("diagnostics-store", "data"),
)
def update_diagnostics_visuals(horizon, data):
    empty_fig = go.Figure()
    if not data:
        return empty_fig, empty_fig, "No diagnostics available.", empty_fig, empty_fig, ""
    regression = data.get("regression", {})
    if not regression:
        return empty_fig, empty_fig, "Regression diagnostics missing.", empty_fig, empty_fig, ""
    selected = regression.get(horizon) if horizon else next(iter(regression.values()))
    residuals = pd.DataFrame(selected.get("residuals", []))
    feat_imp = pd.DataFrame(selected.get("feature_importance", []))
    residual_fig = go.Figure()
    if not residuals.empty and "date" in residuals and "residual" in residuals:
        residual_fig.add_trace(go.Scatter(x=residuals["date"], y=residuals["residual"], mode="lines", name="Residuals"))
    residual_fig.update_layout(title="Residuals Over Time", margin=dict(l=10, r=10, t=40, b=10))
    regression_feat_fig = go.Figure()
    if not feat_imp.empty:
        y_values = feat_imp["abs_coeff"] if "abs_coeff" in feat_imp else feat_imp["coefficient"]
        regression_feat_fig.add_trace(go.Bar(x=feat_imp["feature"], y=y_values))
    regression_feat_fig.update_layout(title="Regression Feature Importance", margin=dict(l=10, r=10, t=40, b=10))

    classification = data.get("classification", {})
    metrics = classification.get("metrics", {})
    class_summary = html.Div(
        [
            html.H4("Classification Metrics"),
            html.Ul([html.Li(f"{key}: {value}") for key, value in metrics.items()]) if metrics else html.P("No metrics."),
        ]
    )

    cross = data.get("cross", {})
    horizon_df = pd.DataFrame(cross.get("horizon_summary", []))
    cross_fig = go.Figure()
    if not horizon_df.empty and "horizon" in horizon_df:
        other_columns = [col for col in horizon_df.columns if col != "horizon"]
        for col in other_columns:
            cross_fig.add_trace(go.Bar(x=horizon_df["horizon"], y=horizon_df[col], name=col))
    cross_fig.update_layout(barmode="group", title="Horizon Comparison", margin=dict(l=10, r=10, t=40, b=10))

    class_feat = pd.DataFrame(classification.get("feature_importance", []))
    class_feat_fig = go.Figure()
    if not class_feat.empty:
        values = class_feat["abs_coeff"] if "abs_coeff" in class_feat else class_feat.iloc[:, 1]
        class_feat_fig.add_trace(go.Bar(x=class_feat["feature"], y=values))
    class_feat_fig.update_layout(title="Classification Feature Importance", margin=dict(l=10, r=10, t=40, b=10))

    message = f"Diagnostics generated at {data.get('generated_at', '—')}"
    return residual_fig, regression_feat_fig, class_summary, cross_fig, class_feat_fig, message


@app.callback(
    Output("data-refresh-token", "data"),
    Output("refresh-status", "children"),
    Input("refresh-data-btn", "n_clicks"),
    State("symbol-dropdown", "value"),
    prevent_initial_call=True,
)
def refresh_data_and_features(n_clicks, symbol):
    if not n_clicks:
        raise PreventUpdate
    try:
        summary = update_data(symbol)
        generate_features(symbol, frequency="daily")
        generate_features(symbol, frequency="weekly")
    except Exception as exc:
        LOGGER.exception("Refresh failed: %s", exc)
        return no_update, html.Span(f"Refresh failed: {exc}", className="text-danger")
    load_price_history_cached.cache_clear()
    load_pipeline_status_cached.cache_clear()
    discover_symbols.cache_clear()
    token = datetime.utcnow().isoformat()
    message = f"Data refreshed: {summary.daily_rows} daily rows, {summary.weekly_rows} weekly rows."
    return token, html.Span(message, className="text-success")


if __name__ == "__main__":
    app.run(debug=True)
