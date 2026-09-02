# ============================================================
# FastAPI Automated Tests
# Flight Price Prediction MLOps Project
# ============================================================

import os
import requests


# ------------------------------------------------------------
# FastAPI URL
# ------------------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)


# ------------------------------------------------------------
# Test 1 — Root Endpoint
# ------------------------------------------------------------

def test_root_endpoint():

    response = requests.get(
        f"{API_URL}/",
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == (
        "Flight Price Prediction API is running"
    )

    assert data["model"] == "flight-price-xgboost"

    assert data["model_version"] == "1"


# ------------------------------------------------------------
# Test 2 — Health Endpoint
# ------------------------------------------------------------

def test_health_endpoint():

    response = requests.get(
        f"{API_URL}/health",
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert data["model_status"] == "Ready"

    assert data["model_name"] == (
        "flight-price-xgboost"
    )

    assert data["model_version"] == "1"


# ------------------------------------------------------------
# Test 3 — Features Endpoint
# ------------------------------------------------------------

def test_features_endpoint():

    response = requests.get(
        f"{API_URL}/features",
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == (
        "flight-price-xgboost"
    )

    assert data["model_version"] == "1"

    assert data["number_of_features"] == 26

    assert len(data["features"]) == 26


# ------------------------------------------------------------
# Test 4 — Prediction Endpoint
# ------------------------------------------------------------

def test_prediction_endpoint():

    payload = {
        "features": {
            "time": 2.0,
            "distance": 500.0,

            "flightType_economic": 1,
            "flightType_firstClass": 0,
            "flightType_premium": 0,

            "agency_CloudFy": 1,
            "agency_FlyingDrops": 0,
            "agency_Rainbow": 0,

            "from_Aracaju (SE)": 0,
            "from_Brasilia (DF)": 0,
            "from_Campo Grande (MS)": 0,
            "from_Florianopolis (SC)": 1,
            "from_Natal (RN)": 0,
            "from_Recife (PE)": 0,
            "from_Rio de Janeiro (RJ)": 0,
            "from_Salvador (BH)": 0,
            "from_Sao Paulo (SP)": 0,

            "to_Aracaju (SE)": 0,
            "to_Brasilia (DF)": 0,
            "to_Campo Grande (MS)": 0,
            "to_Florianopolis (SC)": 0,
            "to_Natal (RN)": 0,
            "to_Recife (PE)": 0,
            "to_Rio de Janeiro (RJ)": 0,
            "to_Salvador (BH)": 0,
            "to_Sao Paulo (SP)": 1
        }
    }

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=30
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_price" in data

    assert isinstance(
        data["predicted_price"],
        float
    )

    assert data["predicted_price"] >= 0


# ------------------------------------------------------------
# Test 5 — Invalid Prediction Request
# ------------------------------------------------------------

def test_invalid_prediction_request():

    payload = {
        "features": {}
    }

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=30
    )

    assert response.status_code == 400