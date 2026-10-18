"""Tests for the prediction API (api/main.py), using the saved model in models/."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

SAMPLE = json.loads((Path(__file__).parent.parent / "api" / "sample_request.json").read_text())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" runs startup, which loads the model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["disease_probability"] <= 1
    # confidence = probability of whichever class was predicted
    p = body["disease_probability"]
    expected = p if body["prediction"] == 1 else 1 - p
    assert body["confidence"] == pytest.approx(expected, abs=1e-3)
    assert body["model_version"]


def test_predict_works_with_unknown_optional_fields(client):
    patient = {k: v for k, v in SAMPLE.items() if k not in ("num_major_vessels", "thalassemia")}
    assert client.post("/predict", json=patient).status_code == 200


@pytest.mark.parametrize("change", [
    {"age": -5},                # impossible age
    {"cholesterol": 5000},      # out of range
    {"thalassemia": 5},         # not a valid code
    {"chest_pain_type": "high"},  # wrong type
    {"favourite_colour": "red"},  # unknown field
])
def test_predict_rejects_bad_input(client, change):
    assert client.post("/predict", json={**SAMPLE, **change}).status_code == 422


def test_predict_rejects_missing_required_field(client):
    patient = {k: v for k, v in SAMPLE.items() if k != "age"}
    assert client.post("/predict", json=patient).status_code == 422
