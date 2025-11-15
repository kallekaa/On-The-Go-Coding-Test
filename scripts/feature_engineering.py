"""Feature engineering script to transform raw data into model-ready features."""

from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def build_features(raw_filename: str, output_filename: str) -> Path:
    """Placeholder for transforming raw data into engineered features."""
    source = RAW_DATA_DIR / raw_filename
    destination = PROCESSED_DATA_DIR / output_filename
    print(f"Generating features from {source} -> {destination}")
    return destination


def main() -> None:
    """Entry point for feature engineering pipeline."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    features_path = build_features("dataset.csv", "features.parquet")
    print(f"Features saved to {features_path}")


if __name__ == "__main__":
    main()
