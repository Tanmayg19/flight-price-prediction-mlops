import os

import mlflow
import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from api.recommendation_service import get_hotel_recommendations


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Travel Analytics API",
    description=(
        "REST API for flight price prediction and "
        "personalized hotel recommendations."
    ),
    version="2.0.0"
)


# ============================================================
# MLflow Configuration
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "flight-price-xgboost"
)

MODEL_VERSION = os.getenv(
    "MLFLOW_MODEL_VERSION",
    "4"
)

MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"


# ============================================================
# Load Flight Price Model
# ============================================================

try:
    model = mlflow.xgboost.load_model(MODEL_URI)
    model_status = "Ready"

except Exception as e:
    model = None
    model_status = f"Failed: {str(e)}"


# ============================================================
# Request Models
# ============================================================

class PredictionRequest(BaseModel):
    features: dict


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Flight Price Prediction API is running",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_status": model_status,
        "recommendation_service": "Ready"
    }


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health():

    if model is None:

        return {
            "status": "unhealthy",
            "model_status": model_status,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "recommendation_service": "Ready"
        }

    return {
        "status": "healthy",
        "model_status": model_status,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "recommendation_service": "Ready"
    }


# ============================================================
# Flight Model Features
# ============================================================

@app.get("/features")
def get_features():

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Flight price model is not loaded."
        )

    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "number_of_features": len(
            model.feature_names_in_
        ),
        "features": list(
            model.feature_names_in_
        )
    }


# ============================================================
# Flight Price Prediction
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Flight price model is not loaded."
        )

    try:

        input_data = pd.DataFrame(
            [request.features]
        )

        prediction = model.predict(
            input_data
        )

        return {
            "predicted_price": float(
                prediction[0]
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ============================================================
# Hotel Recommendations
# ============================================================

@app.get("/recommendations/{user_id}")
def recommendations(
    user_id: int,
    top_n: int = Query(
        default=3,
        ge=1,
        le=9
    )
):

    try:

        recommendation_df = (
            get_hotel_recommendations(
                user_id=user_id,
                top_n=top_n
            )
        )

        # ----------------------------------------------------
        # Convert NaN values to Python None
        #
        # This is especially important for popularity fallback
        # recommendations because recommendation_score is NaN.
        #
        # JSON does not support NaN, but it supports null.
        # Python None becomes JSON null.
        # ----------------------------------------------------

        recommendation_df = (
            recommendation_df
            .astype(object)
            .where(
                pd.notnull(recommendation_df),
                None
            )
        )

        recommendation_records = (
            recommendation_df
            .to_dict(
                orient="records"
            )
        )

        return {
            "user_id": user_id,
            "requested_top_n": top_n,
            "returned_recommendations": len(
                recommendation_records
            ),
            "recommendations": recommendation_records
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Recommendation service failed: "
                f"{str(e)}"
            )
        )