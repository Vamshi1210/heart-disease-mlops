"""Tests for data cleaning and validation (src/clean_data.py)."""
import numpy as np
import pandas as pd
import pytest

from src.clean_data import TARGET, clean, validate


def raw_rows():
    """Three patients in the raw UCI format; the last row duplicates the first."""
    row_a = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
             "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
             "ca": 0.0, "thal": 6.0, "num": 0}
    row_b = {**row_a, "age": 67, "cp": 4, "ca": np.nan, "num": 3}
    return pd.DataFrame([row_a, row_b, row_a])


def test_clean_renames_columns_and_drops_diagnosis():
    df = clean(raw_rows())
    assert "resting_bp" in df.columns and "trestbps" not in df.columns
    assert "diagnosis" not in df.columns


def test_clean_removes_duplicate_rows():
    assert len(clean(raw_rows())) == 2


def test_clean_makes_binary_target():
    # diagnosis 0 -> 0 (no disease); diagnosis 3 -> 1 (disease)
    assert clean(raw_rows())[TARGET].tolist() == [0, 1]


def test_validate_accepts_good_data_with_missing_values():
    validate(clean(raw_rows()))  # NaN in num_major_vessels is allowed; no error expected


@pytest.mark.parametrize("column, bad_value", [
    ("age", 500),
    ("cholesterol", 5),
    ("thalassemia", 5),
    ("chest_pain_type", 9),
])
def test_validate_rejects_impossible_values(column, bad_value):
    df = clean(raw_rows())
    df.loc[0, column] = bad_value
    with pytest.raises(ValueError, match=column):
        validate(df)
