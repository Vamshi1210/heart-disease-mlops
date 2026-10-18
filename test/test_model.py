"""Tests for preprocessing (src/features.py) and metrics (src/train.py)."""
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES, NOMINAL, TARGET, build_preprocessor
from src.train import compute_metrics


def patients():
    """Four patients in the clean format, with gaps in the two columns that have them."""
    return pd.DataFrame({
        "age": [63, 67, 41, 56], "resting_bp": [145, 160, 130, 120],
        "cholesterol": [233, 286, 204, 236], "max_heart_rate": [150, 108, 172, 178],
        "st_depression": [2.3, 1.5, 1.4, 0.8], "num_major_vessels": [0, 3, np.nan, 0],
        "sex": [1, 1, 0, 1], "fasting_blood_sugar": [1, 0, 0, 0],
        "exercise_angina": [0, 1, 0, 0], "chest_pain_type": [1, 4, 2, 2],
        "resting_ecg": [2, 2, 2, 0], "st_slope": [3, 2, 1, 1],
        "thalassemia": [6, 3, 3, np.nan],
    })


def test_feature_list_is_complete_and_excludes_target():
    assert len(FEATURES) == 13
    assert len(set(FEATURES)) == 13
    assert TARGET not in FEATURES


def test_preprocessor_fills_missing_values():
    out = build_preprocessor().fit_transform(patients()[FEATURES])
    assert not np.isnan(out).any()


def test_preprocessor_one_hot_encodes_codes():
    df = patients()[FEATURES]
    out = build_preprocessor().fit_transform(df)
    n_one_hot = sum(df[col].nunique() for col in NOMINAL)
    assert out.shape == (4, 9 + n_one_hot)  # 6 numeric + 3 binary + one-hot columns


def test_preprocessor_ignores_unseen_codes():
    pre = build_preprocessor().fit(patients()[FEATURES])
    new = patients()[FEATURES].head(1).assign(resting_ecg=1)  # code never seen in fit
    assert pre.transform(new).shape[0] == 1


def test_metrics_perfect_predictions():
    y = np.array([0, 1, 0, 1])
    scores = compute_metrics(y, np.array([0.1, 0.9, 0.2, 0.8]))
    assert all(v == pytest.approx(1.0) for v in scores.values())


def test_lower_threshold_raises_recall():
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.3, 0.2, 0.1])  # one sick patient only 30% likely
    assert compute_metrics(y, proba, threshold=0.5)["recall"] == 0.5
    assert compute_metrics(y, proba, threshold=0.25)["recall"] == 1.0
