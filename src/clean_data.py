"""
Clean the raw heart disease data and save it to data/processed/.

Row-by-row fixes only:
  1. Give columns readable names.
  2. Remove exact duplicate rows.
  3. Turn the diagnosis (0-4) into a yes/no target: 0 = no disease, 1 = disease.
  4. Check every value is in a medically sensible range.

Missing values (num_major_vessels, thalassemia) are NOT filled here.
They are filled later inside the model pipeline, using training data only.

Run: python src/clean_data.py
"""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/heart_disease_raw.csv")
CLEAN_PATH = Path("data/processed/heart_disease_clean.csv")
TARGET = "heart_disease"

RENAME_MAP = {
    "cp": "chest_pain_type",
    "trestbps": "resting_bp",
    "chol": "cholesterol",
    "fbs": "fasting_blood_sugar",
    "restecg": "resting_ecg",
    "thalach": "max_heart_rate",
    "exang": "exercise_angina",
    "oldpeak": "st_depression",
    "slope": "st_slope",
    "ca": "num_major_vessels",
    "thal": "thalassemia",
    "num": "diagnosis",
}

# Allowed (min, max) for number columns
NUMERIC_RANGES = {
    "age": (18, 100),
    "resting_bp": (60, 250),
    "cholesterol": (80, 700),
    "max_heart_rate": (50, 250),
    "st_depression": (0.0, 10.0),
}

# Allowed codes for category columns (from the UCI dataset page)
CATEGORY_CODES = {
    "sex": {0, 1},
    "chest_pain_type": {1, 2, 3, 4},
    "fasting_blood_sugar": {0, 1},
    "resting_ecg": {0, 1, 2},
    "exercise_angina": {0, 1},
    "st_slope": {1, 2, 3},
    "num_major_vessels": {0, 1, 2, 3},
    "thalassemia": {3, 6, 7},
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw data."""
    df = df.rename(columns=RENAME_MAP).drop_duplicates().reset_index(drop=True)
    df[TARGET] = (df["diagnosis"] > 0).astype(int)
    return df.drop(columns=["diagnosis"])


def validate(df: pd.DataFrame) -> None:
    """Raise an error listing every value outside its allowed range or codes.
    Missing values (NaN) are allowed; the model pipeline handles them."""
    problems = []
    for col, (low, high) in NUMERIC_RANGES.items():
        bad = df[col].notna() & ~df[col].between(low, high)
        if bad.any():
            problems.append(f"{col}: {bad.sum()} value(s) outside {low}-{high}")
    for col, codes in CATEGORY_CODES.items():
        bad = df[col].notna() & ~df[col].isin(codes)
        if bad.any():
            problems.append(f"{col}: {bad.sum()} value(s) not in {sorted(codes)}")
    if problems:
        raise ValueError("Data validation failed:\n" + "\n".join(problems))


def main() -> None:
    raw = pd.read_csv(RAW_PATH)
    df = clean(raw)
    validate(df)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Raw rows: {len(raw)} | Clean rows: {len(df)} | Duplicates removed: {len(raw) - len(df)}")
    print(f"Target balance: {df[TARGET].value_counts().to_dict()}")
    print(f"Saved to {CLEAN_PATH}")


if __name__ == "__main__":
    main()