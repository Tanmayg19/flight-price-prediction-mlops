import os

import requests


# ============================================================
# TEST CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

EXPECTED_MODEL_VERSION = os.getenv(
    "EXPECTED_MODEL_VERSION",
    "4"
)


# ============================================================
# EXPECTED MODEL FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "time",
    "distance",
    "flightType_economic",
    "flightType_firstClass",
    "flightType_premium",
    "agency_CloudFy",
    "agency_FlyingDrops",
    "agency_Rainbow",
    "from_Aracaju (SE)",
    "from_Brasilia (DF)",
    "from_Campo Grande (MS)",
    "from_Florianopolis (SC)",
    "from_Natal (RN)",
    "from_Recife (PE)",
    "from_Rio de Janeiro (RJ)",
    "from_Salvador (BH)",
    "from_Sao Paulo (SP)",
    "to_Aracaju (SE)",
    "to_Brasilia (DF)",
    "to_Campo Grande (MS)",
    "to_Florianopolis (SC)",
    "to_Natal (RN)",
    "to_Recife (PE)",
    "to_Rio de Janeiro (RJ)",
    "to_Salvador (BH)",
    "to_Sao Paulo (SP)"
]


# ============================================================
# VALID PREDICTION PAYLOAD
# ============================================================

VALID_PAYLOAD = {
    "features": {
        "time": 1.5,
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
        "from_Florianopolis (SC)": 0,
        "from_Natal (RN)": 0,
        "from_Recife (PE)": 0,
        "from_Rio de Janeiro (RJ)": 0,
        "from_Salvador (BH)": 0,
        "from_Sao Paulo (SP)": 1,
        "to_Aracaju (SE)": 0,
        "to_Brasilia (DF)": 0,
        "to_Campo Grande (MS)": 0,
        "to_Florianopolis (SC)": 0,
        "to_Natal (RN)": 0,
        "to_Recife (PE)": 0,
        "to_Rio de Janeiro (RJ)": 1,
        "to_Salvador (BH)": 0,
        "to_Sao Paulo (SP)": 0
    }
}


# ============================================================
# TEST 1 - ROOT ENDPOINT
# ============================================================

def test_root_endpoint():

    response = requests.get(
        f"{API_URL}/",
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["message"]
        == "Flight Price Prediction API is running"
    )

    assert (
        data["model"]
        == "flight-price-xgboost"
    )

    assert (
        data["model_version"]
        == EXPECTED_MODEL_VERSION
    )

    assert (
        data["model_status"]
        == "Ready"
    )


# ============================================================
# TEST 2 - HEALTH ENDPOINT
# ============================================================

def test_health_endpoint():

    response = requests.get(
        f"{API_URL}/health",
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["status"]
        == "healthy"
    )

    assert (
        data["model_status"]
        == "Ready"
    )

    assert (
        data["model_name"]
        == "flight-price-xgboost"
    )

    assert (
        data["model_version"]
        == EXPECTED_MODEL_VERSION
    )


# ============================================================
# TEST 3 - FEATURES ENDPOINT
# ============================================================

def test_features_endpoint():

    response = requests.get(
        f"{API_URL}/features",
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["model_name"]
        == "flight-price-xgboost"
    )

    assert (
        data["model_version"]
        == EXPECTED_MODEL_VERSION
    )

    assert (
        data["number_of_features"]
        == 26
    )

    assert (
        data["features"]
        == EXPECTED_FEATURES
    )


# ============================================================
# TEST 4 - PREDICTION ENDPOINT
# ============================================================

def test_prediction_endpoint():

    response = requests.post(
        f"{API_URL}/predict",
        json=VALID_PAYLOAD,
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        "predicted_price"
        in data
    )

    assert isinstance(
        data["predicted_price"],
        (int, float)
    )

    assert (
        data["predicted_price"]
        > 0
    )


# ============================================================
# TEST 5 - INVALID PREDICTION REQUEST
# ============================================================

def test_invalid_prediction_request():

    invalid_payload = {
        "invalid_field": {}
    }

    response = requests.post(
        f"{API_URL}/predict",
        json=invalid_payload,
        timeout=10
    )

    assert (
        response.status_code
        == 422
    )