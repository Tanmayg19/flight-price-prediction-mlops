from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import os


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="Flight Price Prediction API",
    description="REST API for predicting flight prices using the MLflow registered XGBoost model.",
    version="1.0.0"
)


# --------------------------------------------------
# MLflow Configuration
# --------------------------------------------------

# Docker Compose will provide:
# MLFLOW_TRACKING_URI=http://mlflow:5000
#
# When running locally, it will fall back to:
# http://localhost:5000

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

# --------------------------------------------------
# Registered Model Configuration
# --------------------------------------------------

MODEL_NAME = "flight-price-xgboost"

MODEL_VERSION = "1"

MODEL_URI = (
    f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)


# --------------------------------------------------
# Load Registered Model
# --------------------------------------------------

try:

    model = mlflow.xgboost.load_model(MODEL_URI)

    model_status = "Ready"

except Exception as e:

    model = None
    model_status = f"Failed: {str(e)}"


# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class PredictionRequest(BaseModel):

    features: dict


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Flight Price Prediction API is running",
        "model": "flight-price-xgboost",
        "model_version": "1",
        "model_status": model_status
    }


# --------------------------------------------------
# Model Status
# --------------------------------------------------

@app.get("/health")
def health():

    if model is None:

        return {
            "status": "unhealthy",
            "model_status": model_status
        }

    return {
        "status": "healthy",
        "model_status": "Ready",
        "model_name": "flight-price-xgboost",
        "model_version": "1"
    }


# --------------------------------------------------
# Model Features
# --------------------------------------------------

@app.get("/features")
def get_features():

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded."
        )

    return {
        "model_name": "flight-price-xgboost",
        "model_version": "1",
        "number_of_features": len(model.feature_names_in_),
        "features": list(model.feature_names_in_)
    }

# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Model is not loaded."
        )

    try:

        # Convert incoming features into DataFrame
        input_data = pd.DataFrame([request.features])

        # Make prediction
        prediction = model.predict(input_data)

        return {
            "predicted_price": float(prediction[0])
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )