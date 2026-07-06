from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.abspath("."))

# Mock the model and feature names so tests don't need actual files
import unittest.mock as mock
import numpy as np

mock_model = mock.MagicMock()
mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

with mock.patch("joblib.load", return_value=mock_model), \
     mock.patch("builtins.open", mock.mock_open(read_data='["AMT_INCOME_TOTAL","AMT_CREDIT","DAYS_BIRTH"]')), \
     mock.patch("json.load", return_value=["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH"]):
    from src.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_high_risk():
    r = client.post("/predict", json={"features": {"AMT_INCOME_TOTAL": 202500}})
    assert r.status_code == 200
    data = r.json()
    assert "default_probability" in data
    assert data["risk_label"] in ["HIGH RISK", "LOW RISK"]
    assert 0.0 <= data["default_probability"] <= 1.0

def test_predict_empty_features():
    r = client.post("/predict", json={"features": {}})
    assert r.status_code == 200

def test_features_endpoint():
    r = client.get("/features")
    assert r.status_code == 200
    assert "features" in r.json()