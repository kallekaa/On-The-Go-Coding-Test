"""Model diagnostics for regression and classification pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.regression_training import (
    build_model_path as build_regression_path,
    load_feature_data,
)
from scripts.classification_pipeline import (
    DEFAULT_TECH_FEATURES,
    build_classification_identifier,
    build_classification_features,
    compute_regression_predictions,
    load_classifier_artifacts,
)
from scripts.regression_training import get_connection
from scripts.time_splits import load_time_split
from scripts.targets import ordered_target_names

REGRESSION_HORIZONS = ordered_target_names()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_DIR = PROJECT_ROOT / "diagnostics"
PLOTS_DIR = DIAGNOSTICS_DIR / "plots"
TABLES_DIR = DIAGNOSTICS_DIR / "tables"
REPORTS_DIR = DIAGNOSTICS_DIR / "reports"

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger("diagnostics")


# ============================================================
# Helper data structures
# ============================================================


@dataclass
class RegressionDiagnosticResult:
    metrics: Dict[str, float]
    residuals: pd.DataFrame
    feature_importance: pd.DataFrame
    plots: List[Path]


@dataclass
class ClassificationDiagnosticResult:
    metrics: Dict[str, float]
    calibration: Dict[str, float]
    feature_importance: pd.DataFrame
    error_tables: Dict[str, pd.DataFrame]
    plots: List[Path]


@dataclass
class CrossModelDiagnosticResult:
    horizon_summary: pd.DataFrame
    consistency_metrics: Dict[str, float]
    drift_reports: Dict[str, pd.DataFrame]


@dataclass
class DiagnosticBundle:
    regression: Dict[str, RegressionDiagnosticResult]
    classification: ClassificationDiagnosticResult
    cross_model: CrossModelDiagnosticResult
    reports: Dict[str, Path]


# ============================================================
# Utility functions
# ============================================================


def ensure_dirs(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dirs(path.parent)
    df.to_csv(path, index=False)


def save_json(data: Dict, path: Path) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def plot_and_save(plot_func, path: Path) -> Path:
    ensure_dirs(path.parent)
    plt.figure()
    plot_func()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def rolling_rmse(series_true: pd.Series, series_pred: pd.Series, window: int = 30) -> pd.Series:
    residual_sq = (series_true - series_pred) ** 2
    return residual_sq.rolling(window=window).mean().apply(np.sqrt)


def prepare_feature_matrix(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=df.index)
    matrix = df[columns].replace([np.inf, -np.inf], np.nan)
    matrix = matrix.ffill().bfill().fillna(0.0)
    return matrix


# ============================================================
# Regression diagnostics
# ============================================================


def regression_summary_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    residuals = y_pred - y_true
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    bias = residuals.mean()
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "bias": float(bias),
    }


def residual_analysis_plots(
    y_true: pd.Series,
    y_pred: pd.Series,
    metadata: pd.DataFrame,
    *,
    plot_prefix: Path,
    window: int = 30,
) -> List[Path]:
    residuals = y_pred - y_true

    plots = []
    plots.append(
        plot_and_save(
            lambda: sns.scatterplot(x=y_pred, y=residuals),
            plot_prefix.with_name(plot_prefix.stem + "_resid_vs_pred.png"),
        )
    )
    plots.append(
        plot_and_save(
            lambda: sns.scatterplot(x=y_true, y=residuals),
            plot_prefix.with_name(plot_prefix.stem + "_resid_vs_actual.png"),
        )
    )
    plots.append(
        plot_and_save(
            lambda: plt.plot(metadata["date"], residuals),
            plot_prefix.with_name(plot_prefix.stem + "_resid_over_time.png"),
        )
    )
    rolling = rolling_rmse(y_true, y_pred, window=window)
    plots.append(
        plot_and_save(
            lambda: plt.plot(metadata["date"], rolling),
            plot_prefix.with_name(plot_prefix.stem + "_rolling_rmse.png"),
        )
    )
    return plots


def extract_linear_feature_importance(
    model_path: Path,
    feature_cols: Sequence[str],
    categories: Dict[str, str],
    scaler: Optional[StandardScaler] = None,
) -> pd.DataFrame:
    model = load(model_path)
    coef = getattr(model, "coef_", None)
    if coef is None:
        return pd.DataFrame()
    coef = np.asarray(coef).reshape(-1)
    if scaler is not None and getattr(scaler, "scale_", None) is not None:
        normalized = coef * scaler.scale_
    else:
        normalized = coef
    data = []
    for feature, raw, norm in zip(feature_cols, coef, normalized):
        category = categories.get(feature, "other")
        data.append({"feature": feature, "raw": raw, "normalized": norm, "category": category, "abs_norm": abs(norm)})
    df = pd.DataFrame(data).sort_values("abs_norm", ascending=False)
    return df


def plot_feature_importance(df: pd.DataFrame, plot_path: Path, top_n: int = 20) -> Path:
    top = df.head(top_n)
    plot_path = plot_path.with_suffix(".png")
    return plot_and_save(lambda: sns.barplot(data=top, x="normalized", y="feature"), plot_path)


def regression_prediction_distribution_plots(
    y_true: pd.Series,
    y_pred: pd.Series,
    plot_prefix: Path,
) -> List[Path]:
    plots = []
    plots.append(
        plot_and_save(
            lambda: sns.histplot(y_pred, kde=True, color="blue"),
            plot_prefix.with_name(plot_prefix.stem + "_pred_hist.png"),
        )
    )
    plots.append(
        plot_and_save(
            lambda: sns.histplot(y_true, kde=True, color="green"),
            plot_prefix.with_name(plot_prefix.stem + "_actual_hist.png"),
        )
    )
    plots.append(
        plot_and_save(
            lambda: sns.scatterplot(x=y_true, y=y_pred),
            plot_prefix.with_name(plot_prefix.stem + "_pred_vs_actual.png"),
        )
    )
    plots.append(
        plot_and_save(
            lambda: sns.scatterplot(x=np.sort(y_true), y=np.sort(y_pred)),
            plot_prefix.with_name(plot_prefix.stem + "_qq_plot.png"),
        )
    )
    return plots


def run_regression_diagnostics(
    symbol: str,
    frequency: str,
    horizon: str,
    dataset: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    preds: pd.Series,
    *,
    category_map: Dict[str, str],
    scaler: Optional[StandardScaler],
    model_type: str,
    subset: Optional[pd.DataFrame] = None,
) -> RegressionDiagnosticResult:
    analysis_df = subset if subset is not None else dataset
    subset_preds = preds.loc[analysis_df.index]
    valid_mask = analysis_df[target_col].notna() & subset_preds.notna()
    valid_df = analysis_df.loc[valid_mask]
    subset_preds = subset_preds.loc[valid_mask]
    metadata = valid_df[["date", "symbol"]]
    metrics = regression_summary_metrics(valid_df[target_col], subset_preds)
    plots_prefix = PLOTS_DIR / symbol / frequency / f"{horizon}"
    residual_plots = residual_analysis_plots(valid_df[target_col], subset_preds, metadata, plot_prefix=plots_prefix)
    distribution_plots = regression_prediction_distribution_plots(valid_df[target_col], subset_preds, plots_prefix)
    model_path = build_regression_path(symbol, frequency, model_type, horizon)
    feature_importance = extract_linear_feature_importance(
        model_path, feature_cols, categories=category_map, scaler=scaler
    )
    if not feature_importance.empty:
        plot_feature_importance(feature_importance, plots_prefix.with_name(plots_prefix.stem + "_feature_importance"))
    residuals_df = pd.DataFrame(
        {
            "date": valid_df["date"],
            "actual": valid_df[target_col],
            "prediction": subset_preds,
            "residual": subset_preds - valid_df[target_col],
        }
    )
    save_dataframe(residuals_df, TABLES_DIR / symbol / frequency / f"{horizon}_residuals.csv")
    save_dataframe(feature_importance, TABLES_DIR / symbol / frequency / f"{horizon}_feature_importance.csv")
    return RegressionDiagnosticResult(
        metrics=metrics,
        residuals=residuals_df,
        feature_importance=feature_importance,
        plots=residual_plots + distribution_plots,
    )


# ============================================================
# Classification diagnostics
# ============================================================


def classification_summary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
    }
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    metrics["pr_auc"] = auc(recall, precision)
    cm = confusion_matrix(y_true, preds)
    metrics["confusion_matrix"] = cm.tolist()
    return {k: (float(v) if isinstance(v, np.generic) else v) for k, v in metrics.items()}


def probability_diagnostics(y_true: np.ndarray, y_prob: np.ndarray, metadata: pd.DataFrame, plot_prefix: Path):
    plots = []
    plots.append(
        plot_and_save(
            lambda: CalibrationDisplay.from_predictions(y_true, y_prob),
            plot_prefix.with_name(plot_prefix.stem + "_calibration.png"),
        )
    )
    plots.append(
        plot_and_save(
            lambda: sns.histplot(y_prob, bins=20),
            plot_prefix.with_name(plot_prefix.stem + "_prob_hist.png"),
        )
    )
    plots.append(
        plot_and_save(
            lambda: plt.plot(metadata["date"], y_prob),
            plot_prefix.with_name(plot_prefix.stem + "_prob_over_time.png"),
        )
    )
    brier = mean_squared_error(y_true, y_prob)
    return plots, brier


def classification_feature_importance(model, feature_cols: Sequence[str]) -> pd.DataFrame:
    coef = getattr(model, "coef_", None)
    if coef is None:
        return pd.DataFrame()
    coef = coef.reshape(-1)
    df = pd.DataFrame({"feature": feature_cols, "coefficient": coef})
    df["abs_coeff"] = df["coefficient"].abs()
    return df.sort_values("abs_coeff", ascending=False)


def classification_error_analysis(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, pd.DataFrame]:
    preds = (y_prob >= threshold).astype(int)
    df = df.copy()
    df["actual"] = y_true
    df["prob"] = y_prob
    df["prediction"] = preds
    errors = {}
    false_positives = df[(df["prediction"] == 1) & (df["actual"] == 0)]
    false_negatives = df[(df["prediction"] == 0) & (df["actual"] == 1)]
    errors["false_positives"] = false_positives
    errors["false_negatives"] = false_negatives
    by_volatility = df.groupby(pd.cut(df["volatility_std_20d"], bins=5))["prediction"].mean().reset_index()
    errors["volatility_regimes"] = by_volatility
    by_weekday = (
        df.assign(weekday=df["date"].dt.day_name())
        .groupby("weekday")[["prediction", "actual"]]
        .mean()
        .reset_index()
    )
    errors["weekday_effect"] = by_weekday
    return errors


def run_classification_diagnostics(
    symbol: str,
    frequency: str,
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    feature_cols: Sequence[str],
    threshold: float,
    model,
) -> ClassificationDiagnosticResult:
    metrics = classification_summary_metrics(y_true, y_prob, threshold)
    plots_prefix = PLOTS_DIR / symbol / frequency / "classification"
    prob_plots, brier = probability_diagnostics(y_true, y_prob, df, plots_prefix)
    metrics["brier"] = float(brier)
    feature_imp = classification_feature_importance(model, feature_cols)
    error_tables = classification_error_analysis(df, y_true, y_prob, threshold)
    for name, table in error_tables.items():
        save_dataframe(table, TABLES_DIR / symbol / frequency / f"classification_{name}.csv")
    save_dataframe(feature_imp, TABLES_DIR / symbol / frequency / "classification_feature_importance.csv")
    return ClassificationDiagnosticResult(
        metrics=metrics,
        calibration={"brier": float(brier)},
        feature_importance=feature_imp,
        error_tables=error_tables,
        plots=prob_plots,
    )


# ============================================================
# Cross-model diagnostics
# ============================================================


def horizon_comparison(regression_results: Dict[str, RegressionDiagnosticResult]) -> pd.DataFrame:
    rows = []
    for horizon, result in regression_results.items():
        row = {"horizon": horizon}
        row.update(result.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def consistency_diagnostics(regression_preds: Dict[str, pd.Series]) -> Dict[str, float]:
    horizons = list(regression_preds)
    correlations = []
    for i in range(len(horizons)):
        for j in range(i + 1, len(horizons)):
            corr = regression_preds[horizons[i]].corr(regression_preds[horizons[j]])
            correlations.append(corr)
    return {"average_horizon_corr": float(np.nanmean(correlations)) if correlations else np.nan}


def drift_detection(df: pd.DataFrame, feature_cols: Sequence[str]) -> Dict[str, pd.DataFrame]:
    splits = np.array_split(df, 4)
    drift_tables = {}
    for idx in range(len(splits) - 1):
        current = splits[idx][feature_cols].mean()
        next_split = splits[idx + 1][feature_cols].mean()
        delta = (next_split - current).abs().sort_values(ascending=False)
        drift_tables[f"drift_segment_{idx}_{idx+1}"] = pd.DataFrame({"feature": delta.index, "mean_shift": delta.values})
    return drift_tables


def run_cross_model_diagnostics(
    regression_results: Dict[str, RegressionDiagnosticResult],
    regression_preds: Dict[str, pd.Series],
    combined_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> CrossModelDiagnosticResult:
    horizon_summary = horizon_comparison(regression_results)
    consistency_metrics = consistency_diagnostics(regression_preds)
    drift_reports = drift_detection(combined_df, feature_cols)
    for name, table in drift_reports.items():
        save_dataframe(table, TABLES_DIR / f"{name}.csv")
    save_dataframe(horizon_summary, TABLES_DIR / "horizon_summary.csv")
    save_json(consistency_metrics, REPORTS_DIR / "consistency_metrics.json")
    return CrossModelDiagnosticResult(
        horizon_summary=horizon_summary,
        consistency_metrics=consistency_metrics,
        drift_reports=drift_reports,
    )


# ============================================================
# Orchestration
# ============================================================


def run_diagnostics(
    symbol: str,
    *,
    frequency: str = "daily",
    regression_model_type: str = "ridge",
    classification_threshold: float = 0.02,
    window: Tuple[int, int] = (1, 5),
    db_path: Path | str = PROJECT_ROOT / "data" / "stocks.db",
) -> Optional[DiagnosticBundle]:
    ensure_dirs(DIAGNOSTICS_DIR, PLOTS_DIR, TABLES_DIR, REPORTS_DIR)
    conn = get_connection(db_path)
    try:
        dataset = load_feature_data(symbol, frequency, conn)
    finally:
        conn.close()
    if dataset.data.empty:
        LOGGER.warning("No data available for diagnostics.")
        return None

    split_definition = load_time_split(symbol, frequency)
    analysis_df = (
        split_definition.select(dataset.data, "test")
        if split_definition is not None
        else dataset.data
    )
    regression_results = {}
    regression_preds = {}
    category_map = {col: ("ma" if "ma_" in col else "other") for col in dataset.feature_cols}
    for horizon in [col for col in dataset.target_cols if col.startswith("target_")]:
        model_path = build_regression_path(symbol, frequency, regression_model_type, horizon)
        if not model_path.exists():
            continue
        model = load(model_path)
        feature_matrix = prepare_feature_matrix(dataset.data, dataset.feature_cols)
        preds = model.predict(feature_matrix)
        regression_preds[horizon] = pd.Series(preds, index=dataset.data.index)
        regression_results[horizon] = run_regression_diagnostics(
            symbol,
            frequency,
            horizon,
            dataset.data,
            dataset.feature_cols,
            horizon,
            regression_preds[horizon],
            category_map=category_map,
            scaler=None,
            model_type=regression_model_type,
            subset=analysis_df,
        )

    identifier = build_classification_identifier(symbol, frequency, classification_threshold, window)
    try:
        classifier, scaler, threshold = load_classifier_artifacts(identifier)
    except FileNotFoundError:
        LOGGER.warning("Classification artifacts missing.")
        classifier_result = ClassificationDiagnosticResult({}, {}, pd.DataFrame(), {}, [])
    else:
        regression_output_map = {
            horizon: regression_preds[horizon]
            for horizon in REGRESSION_HORIZONS
            if horizon in regression_preds
        }
        feature_matrix, feature_cols = build_classification_features(
            dataset.data, regression_output_map, selected_features=DEFAULT_TECH_FEATURES
        )
        numeric_df = feature_matrix[feature_cols]
        probs = classifier.predict_proba(scaler.transform(numeric_df))[:, 1]
        prob_series = pd.Series(probs, index=dataset.data.index)
        labels_series = (
            dataset.data["label_buy"].fillna(0).astype(int)
            if "label_buy" in dataset.data
            else pd.Series(np.zeros(len(probs), dtype=int), index=dataset.data.index)
        )
        analysis_probs = prob_series.loc[analysis_df.index].values
        analysis_labels = labels_series.loc[analysis_df.index].values
        classifier_result = run_classification_diagnostics(
            symbol,
            frequency,
            analysis_df,
            analysis_labels,
            analysis_probs,
            feature_cols,
            threshold,
            classifier,
        )

    cross_result = run_cross_model_diagnostics(
        regression_results, regression_preds, dataset.data, dataset.feature_cols
    )
    summary_report = REPORTS_DIR / symbol / frequency / "diagnostics_summary.txt"
    ensure_dirs(summary_report.parent)
    with summary_report.open("w", encoding="utf-8") as fh:
        fh.write("Regression Metrics:\n")
        for horizon, result in regression_results.items():
            fh.write(f"{horizon}: {result.metrics}\n")
        fh.write("\nClassification Metrics:\n")
        fh.write(json.dumps(classifier_result.metrics, indent=2))
        fh.write("\nCross-Model Insights:\n")
        fh.write(json.dumps(cross_result.consistency_metrics, indent=2))
    LOGGER.info("Diagnostics completed.")
    return DiagnosticBundle(
        regression=regression_results,
        classification=classifier_result,
        cross_model=cross_result,
        reports={"summary": summary_report},
    )


if __name__ == "__main__":
    run_diagnostics("AAPL", frequency="daily")
