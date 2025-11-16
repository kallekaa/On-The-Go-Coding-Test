"""Data ingestion script for collecting and storing raw datasets."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")


def download_data(source: str) -> Path:
    """Placeholder for downloading data from a remote source."""
    destination = RAW_DATA_DIR / "dataset.csv"
    message = f"Simulating download from {source} to {destination}"
    logger.info(message)
    print(message)
    return destination


def main() -> None:
    """Entry point for the data ingestion workflow."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = download_data("https://example.com/dataset")
    message = f"Data saved to {dataset_path}"
    logger.info(message)
    print(message)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
