import os
from pathlib import Path

import joblib
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
        "REST API for flight price prediction, "
        "personalized hotel recommendations, and "
        "gender classification."
    ),
    version="3.0.0"
)


# ============================================================
# Project Paths
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent


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

    model = mlflow.xgboost.load_model(
        MODEL_URI
    )

    model_status = "Ready"

except Exception as e:

    model = None

    model_status = (
        f"Failed: {str(e)}"
    )


# ============================================================
# Gender Classification Model Path
# ============================================================

# Local project:
# project/models/gender_classification/gender_classifier.pkl
#
# Docker:
# /app/models/gender_classification/gender_classifier.pkl

docker_gender_model_path = (
    Path("/app")
    / "models"
    / "gender_classification"
    / "gender_classifier.pkl"
)

local_gender_model_path = (
    PROJECT_ROOT
    / "models"
    / "gender_classification"
    / "gender_classifier.pkl"
)


if docker_gender_model_path.exists():

    GENDER_MODEL_PATH = (
        docker_gender_model_path
    )

else:

    GENDER_MODEL_PATH = (
        local_gender_model_path
    )


# ============================================================
# Load Gender Classification Model
# ============================================================

try:

    gender_model = joblib.load(
        GENDER_MODEL_PATH
    )

    gender_model_status = "Ready"

except Exception as e:

    gender_model = None

    gender_model_status = (
        f"Failed: {str(e)}"
    )


# ============================================================
# Request Models
# ============================================================

class PredictionRequest(BaseModel):
    features: dict


class GenderPredictionRequest(BaseModel):
    company: str
    age: int


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Travel Analytics API is running",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_status": model_status,
        "recommendation_service": "Ready",
        "gender_model_status": gender_model_status
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
            "recommendation_service": "Ready",
            "gender_model_status": gender_model_status
        }

    return {
        "status": "healthy",
        "model_status": model_status,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "recommendation_service": "Ready",
        "gender_model_status": gender_model_status
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
def predict(
    request: PredictionRequest
):

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
# Gender Classification
# ============================================================

@app.post("/gender/predict")
def predict_gender(
    request: GenderPredictionRequest
):

    if gender_model is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Gender classification model "
                "is not loaded."
            )
        )

    try:

        # ----------------------------------------------------
        # Validate age
        # ----------------------------------------------------

        if request.age < 21 or request.age > 65:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Age must be between 21 and 65, "
                    "which is the range represented "
                    "in the training dataset."
                )
            )


        # ----------------------------------------------------
        # Create raw model input
        #
        # The saved pipeline performs:
        # - OneHotEncoding for company
        # - StandardScaling for age
        # - Random Forest prediction
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [
                {
                    "company": request.company,
                    "age": request.age
                }
            ]
        )


        # ----------------------------------------------------
        # Generate prediction
        # ----------------------------------------------------

        prediction = gender_model.predict(
            input_data
        )

        predicted_gender = str(
            prediction[0]
        )


        # ----------------------------------------------------
        # Return prediction
        # ----------------------------------------------------

        return {
            "predicted_gender": predicted_gender,
            "company": request.company,
            "age": request.age,
            "model": "Tuned Random Forest",
            "test_accuracy": 0.5222,
            "macro_f1": 0.5220
        }


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                "Gender prediction failed: "
                f"{str(e)}"
            )
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
                pd.notnull(
                    recommendation_df
                ),
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
            "recommendations": (
                recommendation_records
            )
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