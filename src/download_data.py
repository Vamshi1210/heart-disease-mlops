"""
Download the UCI Heart Disease dataset (Cleveland, 303 patients)
and save it as a CSV file in data/raw/.

Source: https://archive.ics.uci.edu/dataset/45/heart+disease (CC BY 4.0)
Run:    python src/download_data.py
"""
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

UCI_DATASET_ID = 45
RAW_PATH = Path("data/raw/heart_disease_raw.csv")


def download_heart_data() -> pd.DataFrame:
    """Fetch the dataset from UCI and return features + target in one table."""
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    features = dataset.data.features   # 13 patient measurements
    target = dataset.data.targets      # 'num': 0 = no disease, 1-4 = disease
    return pd.concat([features, target], axis=1)


def main() -> None:
    df = download_heart_data()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    print(f"Saved {df.shape[0]} rows x {df.shape[1]} columns to {RAW_PATH}")
    missing = df.isna().sum()
    print("Columns with missing values:")
    print(missing[missing > 0])


if __name__ == "__main__":
    main()
