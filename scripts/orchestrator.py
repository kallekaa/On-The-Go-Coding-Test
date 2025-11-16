"""Automation and orchestration entry point for the stock prediction workflow."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.data_pipeline import update_data
from scripts.feature_engineering import generate_features
from scripts.regression_training import train_regression_models
from scripts.classification_pipeline import train_buy_classifier
from scripts.backtesting import run_backtest
from scripts.diagnostics import run_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
DEFAULT_LOG_PATH = PROJECT_ROOT / "automation.log"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("orchestrator")


# ============================================================
# Configuration
# ============================================================


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    """Load configuration from JSON if it exists."""
    if path and path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def merge_configs(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge CLI overrides into base config."""
    merged = base.copy()
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


@dataclass
class TaskContext:
    symbol: str
    frequency: str
    config: Dict[str, Any]


# ============================================================
# Orchestrator functions per step
# ============================================================


def run_ingest_data(context: TaskContext) -> None:
    params = context.config.get("ingestion", {})
    start_date = params.get("start_date", "2022-01-01")
    end_date = params.get("end_date")
    LOGGER.info("Ingesting data for %s (%s) from %s to %s", context.symbol, context.frequency, start_date, end_date)
    update_data(context.symbol, start_date=start_date, end_date=end_date)


def run_feature_generation(context: TaskContext) -> None:
    params = context.config.get("features", {})
    poly_degree = params.get("poly_degree", 2)
    LOGGER.info("Generating %s features for %s", context.frequency, context.symbol)
    generate_features(
        context.symbol,
        frequency=context.frequency,
        start_date=params.get("start_date", "2022-01-01"),
        train_end_date=params.get("end_date"),
        poly_degree=poly_degree,
    )


def run_regression_training_task(context: TaskContext) -> None:
    params = context.config.get("regression", {})
    LOGGER.info("Training %s regression models for %s", context.frequency, context.symbol)
    train_regression_models(
        context.symbol,
        frequency=context.frequency,
        model_type=params.get("model_type", "ridge"),
        alpha_grid=params.get("alpha_grid"),
        train_ratio=params.get("train_ratio", 0.7),
        val_ratio=params.get("val_ratio", 0.15),
        test_ratio=params.get("test_ratio", 0.15),
        walk_forward=params.get("walk_forward"),
    )


def run_classification_training_task(context: TaskContext) -> None:
    params = context.config.get("classification", {})
    LOGGER.info("Training classifier for %s (%s)", context.symbol, context.frequency)
    train_buy_classifier(
        context.symbol,
        frequency=context.frequency,
        threshold=params.get("threshold", 0.02),
        window=tuple(params.get("window", (1, 5))),
        penalties=params.get("penalties", ("l2", "l1")),
        c_values=params.get("c_values", (0.01, 0.1, 1.0, 10.0)),
        train_ratio=params.get("train_ratio", 0.6),
        validation_ratio=params.get("validation_ratio", 0.2),
        test_ratio=params.get("test_ratio", 0.2),
        probability_strategy=params.get("probability_strategy", "f1"),
        cv_metric=params.get("cv_metric", "f1"),
        class_weight=params.get("class_weight"),
    )


def run_backtest_task(context: TaskContext) -> None:
    params = context.config.get("backtest", {})
    LOGGER.info("Running backtest for %s (%s)", context.symbol, context.frequency)
    run_backtest(
        context.symbol,
        frequency=context.frequency,
        threshold=params.get("threshold", 0.02),
        window=tuple(params.get("window", (1, 5))),
    )


def run_diagnostics_task(context: TaskContext) -> None:
    params = context.config.get("diagnostics", {})
    LOGGER.info("Running diagnostics for %s (%s)", context.symbol, context.frequency)
    run_diagnostics(
        context.symbol,
        frequency=context.frequency,
        regression_model_type=params.get("regression_model_type", "ridge"),
        classification_threshold=params.get("threshold", 0.02),
    )


TASK_HANDLERS = {
    "ingest_data": run_ingest_data,
    "generate_features": run_feature_generation,
    "train_regression_models": run_regression_training_task,
    "train_classification_model": run_classification_training_task,
    "run_backtest": run_backtest_task,
    "run_diagnostics": run_diagnostics_task,
}


# ============================================================
# Full pipeline sequencing
# ============================================================


def run_full_sequence(context: TaskContext) -> None:
    sequence = context.config.get(
        "full_run_sequence",
        [
            "ingest_data",
            "generate_features",
            "train_regression_models",
            "train_classification_model",
            "run_backtest",
            "run_diagnostics",
        ],
    )
    enabled_flags = context.config.get("full_run_flags", {})

    for task_name in sequence:
        if enabled_flags.get(task_name, True):
            handler = TASK_HANDLERS.get(task_name)
            if handler:
                LOGGER.info("Starting task: %s", task_name)
                try:
                    handler(context)
                except Exception as exc:
                    LOGGER.exception("Task %s failed: %s", task_name, exc)
                    if context.config.get("stop_on_failure", True):
                        raise
            else:
                LOGGER.warning("No handler for task %s", task_name)


# ============================================================
# CLI and dispatch
# ============================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock pipeline orchestrator")
    parser.add_argument("--task", required=True, choices=list(TASK_HANDLERS) + ["full_run"])
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--frequency", default="daily")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--window", type=int, nargs=2)
    parser.add_argument("--model-type")
    parser.add_argument("--stop-on-failure", type=bool)
    return parser.parse_args(argv)


def setup_logging(log_path: Path) -> None:
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_file)
    base_config = load_config(args.config)
    overrides = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "threshold": args.threshold,
        "window": tuple(args.window) if args.window else None,
        "regression_model_type": args.model_type,
        "stop_on_failure": args.stop_on_failure,
    }
    config = merge_configs(base_config, {k: v for k, v in overrides.items() if v is not None})
    context = TaskContext(symbol=args.symbol, frequency=args.frequency, config=config)

    if args.task == "full_run":
        run_full_sequence(context)
    else:
        handler = TASK_HANDLERS[args.task]
        handler(context)


if __name__ == "__main__":
    main()
