import os

import requests


# ============================================================
# API Configuration
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
# Expected Flight Model Features
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
# Sample Prediction Payload
# ============================================================

SAMPLE_FEATURES = {
    "time": 1.5,
    "distance": 600.0,
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


# ============================================================
# Existing Flight API Tests
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

    assert (
        data["recommendation_service"]
        == "Ready"
    )


def test_health_endpoint():

    response = requests.get(
        f"{API_URL}/health",
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

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

    assert (
        data["recommendation_service"]
        == "Ready"
    )


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


def test_prediction_endpoint():

    response = requests.post(
        f"{API_URL}/predict",
        json={
            "features": SAMPLE_FEATURES
        },
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


def test_invalid_prediction_request():

    response = requests.post(
        f"{API_URL}/predict",
        json={},
        timeout=10
    )

    assert response.status_code == 422


# ============================================================
# Recommendation API Tests
# ============================================================

def test_personalized_recommendation():

    response = requests.get(
        (
            f"{API_URL}"
            "/recommendations/1104"
            "?top_n=3"
        ),
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == 1104

    assert (
        data["requested_top_n"]
        == 3
    )

    assert (
        data["returned_recommendations"]
        >= 1
    )

    assert (
        data["returned_recommendations"]
        <= 3
    )

    recommendations = (
        data["recommendations"]
    )

    assert len(
        recommendations
    ) == data[
        "returned_recommendations"
    ]

    for recommendation in recommendations:

        assert (
            recommendation[
                "recommendation_type"
            ]
            == "Personalized SVD"
        )

        assert (
            recommendation["userCode"]
            == 1104
        )

        assert (
            "name"
            in recommendation
        )

        assert (
            "place"
            in recommendation
        )

        assert (
            "average_price"
            in recommendation
        )


def test_cold_start_recommendation():

    response = requests.get(
        (
            f"{API_URL}"
            "/recommendations/99999"
            "?top_n=3"
        ),
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == 99999

    assert (
        data["requested_top_n"]
        == 3
    )

    assert (
        data["returned_recommendations"]
        == 3
    )

    recommendations = (
        data["recommendations"]
    )

    assert len(
        recommendations
    ) == 3

    expected_hotels = [
        "Hotel CB",
        "Hotel K",
        "Hotel AF"
    ]

    returned_hotels = [
        recommendation["name"]
        for recommendation
        in recommendations
    ]

    assert (
        returned_hotels
        == expected_hotels
    )

    for recommendation in recommendations:

        assert (
            recommendation[
                "recommendation_type"
            ]
            == "Popularity Fallback"
        )

        assert (
            recommendation["userCode"]
            == 99999
        )

        assert (
            recommendation[
                "recommendation_score"
            ]
            is None
        )


def test_recommendation_top_n():

    response = requests.get(
        (
            f"{API_URL}"
            "/recommendations/99999"
            "?top_n=2"
        ),
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["requested_top_n"]
        == 2
    )

    assert (
        data["returned_recommendations"]
        == 2
    )

    assert (
        len(data["recommendations"])
        == 2
    )


def test_invalid_recommendation_top_n_zero():

    response = requests.get(
        (
            f"{API_URL}"
            "/recommendations/99999"
            "?top_n=0"
        ),
        timeout=10
    )

    assert response.status_code == 422


def test_invalid_recommendation_top_n_too_large():

    response = requests.get(
        (
            f"{API_URL}"
            "/recommendations/99999"
            "?top_n=10"
        ),
        timeout=10
    )

    assert response.status_code == 422