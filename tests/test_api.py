import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import json

# Mock everything before importing app
mock_model = MagicMock()
mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
feature_names = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH"]

with patch("joblib.load", return_value=mock_model), \
     patch("builtins.open", MagicMock()), \
     patch("json.load", return_value=feature_names):
    from src.app import app

from fastapi.testclient import TestClient
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_response():
    r = client.post("/predict", json={"features": {"AMT_INCOME_TOTAL": 202500}})
    assert r.status_code == 200
    data = r.json()
    assert "default_probability" in data
    assert data["risk_label"] in ["HIGH RISK", "LOW RISK"]


def test_predict_empty_features():
    r = client.post("/predict", json={"features": {}})
    assert r.status_code == 200


def test_features_endpoint():
    r = client.get("/features")
    assert r.status_code == 200
    assert "features" in r.json()