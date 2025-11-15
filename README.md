# On-The-Go Coding Test

This repository contains a simple data science/ML project skeleton with clear separation between data, models, and scripts.

## Project Structure

- `data/`: Raw and processed datasets.
  - `raw/`
  - `processed/`
- `models/`: Serialized model artifacts and checkpoints.
- `scripts/`: Modular Python scripts for each pipeline stage.
  - `data_ingestion.py`
  - `feature_engineering.py`
  - `training_validation.py`
  - `inference.py`

Each script provides a `main()` function to run the respective stage independently.
