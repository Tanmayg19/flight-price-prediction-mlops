import os

import mlflow
import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Flight Price Prediction API",
    description=(
        "REST API for predicting flight prices using "
        "the MLflow registered XGBoost model."
    ),
    version="1.0.0"
)


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


# ============================================================
# REGISTERED MODEL CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "flight-price-xgboost"
)

MODEL_VERSION = os.getenv(
    "MLFLOW_MODEL_VERSION",
    "4"
)

MODEL_URI = (
    f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)


# ============================================================
# LOAD REGISTERED MODEL
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
# REQUEST SCHEMA
# ============================================================

class PredictionRequest(BaseModel):

    features: dict


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": (
            "Flight Price Prediction API is running"
        ),
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_status": model_status
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    if model is None:

        return {
            "status": "unhealthy",
            "model_status": model_status,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION
        }

    return {
        "status": "healthy",
        "model_status": model_status,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION
    }


# ============================================================
# MODEL FEATURES
# ============================================================

@app.get("/features")
def get_features():

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Model is not loaded."
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
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(
    request: PredictionRequest
):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Model is not loaded."
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