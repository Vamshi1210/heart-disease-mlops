"""
Feature lists and preprocessing, shared by training, tests and the API.

Keeping them in one place guarantees the API prepares a new patient's data
exactly the way training did.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "heart_disease"

# Amounts and counts: fill gaps with the median, then scale to a common range.
# num_major_vessels is a count (0-3) whose disease rate rises steadily (see EDA),
# so it is treated as a number.
NUMERIC = ["age", "resting_bp", "cholesterol", "max_heart_rate",
           "st_depression", "num_major_vessels"]

# Already 0/1: only fill gaps.
BINARY = ["sex", "fasting_blood_sugar", "exercise_angina"]

# Codes with no meaningful order (thalassemia 3/6/7 are labels, and for st_slope
# "flat" has more disease than "down"): fill gaps, then one-hot encode.
NOMINAL = ["chest_pain_type", "resting_ecg", "st_slope", "thalassemia"]

FEATURES = NUMERIC + BINARY + NOMINAL


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted preprocessing step. It learns medians and scales
    from training data only, so no information leaks from the test set."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("binary", SimpleImputer(strategy="most_frequent"), BINARY),
        ("nominal", nominal, NOMINAL),
    ])