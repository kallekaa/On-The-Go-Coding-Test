"""Utilities for writing diagnostic JSON outputs and maintaining an index."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_ROOT = PROJECT_ROOT / "diagnostics"
INDEX_PATH = DIAGNOSTICS_ROOT / "index.json"


def _timestamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _load_index() -> Dict[str, Any]:
    if INDEX_PATH.exists():
        with INDEX_PATH.open("r", encoding="utf-8") as fh:
            try:
                return json.load(fh)
            except json.JSONDecodeError:
                return {"generated_at": None, "entries": []}
    return {"generated_at": None, "entries": []}


def _save_index(index: Dict[str, Any]) -> None:
    index["generated_at"] = _timestamp()
    _write_json(INDEX_PATH, index)


def _is_path_like(value: Any) -> bool:
    return isinstance(value, (Path, str))


def _normalize_artifacts(artifacts: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for artifact in artifacts or []:
        entry = dict(artifact)
        path_value = entry.get("path")
        if _is_path_like(path_value):
            entry["path"] = _relative_path(Path(path_value))
        normalized.append(entry)
    return normalized


def _register_entry(entry: Dict[str, Any]) -> None:
    index = _load_index()
    entries = index.get("entries", [])
    entries = [item for item in entries if item.get("id") != entry["id"]]
    entries.append(entry)
    entries.sort(key=lambda item: (item.get("symbol", ""), item.get("frequency", ""), item.get("stage", "")))
    index["entries"] = entries
    _save_index(index)


def save_regression_summary(
    *,
    symbol: str,
    frequency: str,
    model_type: str,
    horizon: str,
    metrics: Dict[str, Any],
    best_alpha: float,
    feature_importance: List[Dict[str, Any]],
    error_series: List[Dict[str, Any]],
    artifacts: Optional[List[Dict[str, Any]]] = None,
    insight_flags: Optional[List[str]] = None,
) -> Path:
    identifier = f"{symbol.upper()}_{frequency}_{model_type}_{horizon}"
    diag_id = f"regression::{identifier}"
    path = DIAGNOSTICS_ROOT / "regression" / f"{identifier}.json"
    summary = {
        "id": diag_id,
        "stage": "regression",
        "symbol": symbol.upper(),
        "frequency": frequency,
        "model_type": model_type,
        "horizon": horizon,
        "best_alpha": best_alpha,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "error_series": error_series,
        "insight_flags": insight_flags or [],
        "artifacts": _normalize_artifacts(artifacts),
        "generated_at": _timestamp(),
    }
    _write_json(path, summary)
    _register_entry(
        {
            "id": diag_id,
            "stage": "regression",
            "symbol": symbol.upper(),
            "frequency": frequency,
            "horizon": horizon,
            "path": _relative_path(path),
            "updated_at": summary["generated_at"],
            "metadata": {
                "model_type": model_type,
                "best_alpha": best_alpha,
            },
        }
    )
    return path


def save_classification_summary(
    *,
    symbol: str,
    frequency: str,
    identifier: str,
    metrics: Dict[str, Any],
    threshold: float,
    confusion_matrix: List[List[int]],
    curves: Dict[str, Any],
    artifacts: Optional[List[Dict[str, Any]]] = None,
    insight_flags: Optional[List[str]] = None,
) -> Path:
    diag_id = f"classification::{identifier}"
    path = DIAGNOSTICS_ROOT / "classification" / f"{identifier}.json"
    summary = {
        "id": diag_id,
        "stage": "classification",
        "symbol": symbol.upper(),
        "frequency": frequency,
        "identifier": identifier,
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix,
        "curves": curves,
        "insight_flags": insight_flags or [],
        "artifacts": _normalize_artifacts(artifacts),
        "generated_at": _timestamp(),
    }
    _write_json(path, summary)
    _register_entry(
        {
            "id": diag_id,
            "stage": "classification",
            "symbol": symbol.upper(),
            "frequency": frequency,
            "identifier": identifier,
            "path": _relative_path(path),
            "updated_at": summary["generated_at"],
            "metadata": {
                "threshold": threshold,
                "metrics": metrics,
            },
        }
    )
    return path


def save_backtest_summary(
    *,
    symbol: str,
    frequency: str,
    identifier: str,
    metrics: Dict[str, Any],
    benchmark: Dict[str, Any],
    equity_curve: List[Dict[str, Any]],
    insight_flags: Optional[List[str]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    diag_id = f"backtest::{identifier}"
    path = DIAGNOSTICS_ROOT / "backtesting" / f"{identifier}.json"
    summary = {
        "id": diag_id,
        "stage": "backtest",
        "symbol": symbol.upper(),
        "frequency": frequency,
        "identifier": identifier,
        "metrics": metrics,
        "benchmark": benchmark,
        "equity_curve": equity_curve,
        "insight_flags": insight_flags or [],
        "artifacts": _normalize_artifacts(artifacts),
        "generated_at": _timestamp(),
    }
    _write_json(path, summary)
    _register_entry(
        {
            "id": diag_id,
            "stage": "backtest",
            "symbol": symbol.upper(),
            "frequency": frequency,
            "identifier": identifier,
            "path": _relative_path(path),
            "updated_at": summary["generated_at"],
            "metadata": {
                "metrics": metrics,
            },
        }
    )
    return path

