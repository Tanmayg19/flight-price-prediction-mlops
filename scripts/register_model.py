import os
import time

import joblib
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient


# ============================================================
# CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "flight-price-prediction-ci"
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/xgb_flight_price_model.pkl"
)

MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "flight-price-xgboost"
)


# ============================================================
# CONNECT TO MLFLOW
# ============================================================

print("Connecting to MLflow...")

print(
    f"Tracking URI: {MLFLOW_TRACKING_URI}"
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

client = MlflowClient()


# ============================================================
# WAIT FOR MLFLOW SERVER
# ============================================================

for attempt in range(30):

    try:

        client.search_experiments()

        print(
            "MLflow is ready."
        )

        break

    except Exception:

        print(
            f"Waiting for MLflow... "
            f"attempt {attempt + 1}/30"
        )

        time.sleep(2)

else:

    raise RuntimeError(
        "MLflow server did not become available."
    )


# ============================================================
# GET OR CREATE EXPERIMENT
# ============================================================

experiment = client.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:

    print(
        f"Experiment '{EXPERIMENT_NAME}' "
        "does not exist."
    )

    print(
        "Creating MLflow experiment..."
    )

    experiment_id = (
        client.create_experiment(
            EXPERIMENT_NAME
        )
    )

    experiment = client.get_experiment(
        experiment_id
    )

    print(
        "Experiment created successfully."
    )

else:

    experiment_id = (
        experiment.experiment_id
    )

    print(
        "Using existing MLflow experiment."
    )


# ============================================================
# DISPLAY EXPERIMENT INFORMATION
# ============================================================

print(
    f"Experiment: {EXPERIMENT_NAME}"
)

print(
    f"Experiment ID: {experiment_id}"
)

print(
    f"Artifact location: "
    f"{experiment.artifact_location}"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print(
    f"Loading model from: {MODEL_PATH}"
)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )


model = joblib.load(
    MODEL_PATH
)

print(
    "XGBoost model loaded successfully."
)


# ============================================================
# LOG AND REGISTER MODEL
# ============================================================

print(
    "Logging and registering model in MLflow..."
)

with mlflow.start_run(
    experiment_id=experiment_id,
    run_name="model-registration"
):

    model_info = mlflow.xgboost.log_model(
        model,
        name="model",
        registered_model_name=MODEL_NAME,
        await_registration_for=300
    )

    run_id = (
        mlflow.active_run().info.run_id
    )


# ============================================================
# VERIFY REGISTERED MODEL INFORMATION
# ============================================================

registered_version = (
    model_info.registered_model_version
)

if registered_version is None:

    raise RuntimeError(
        "Model was logged but was not "
        "registered in MLflow Model Registry."
    )


print(
    "Model logged successfully."
)

print(
    f"Run ID: {run_id}"
)

print(
    f"Model URI: {model_info.model_uri}"
)

print(
    f"Registered model: {MODEL_NAME}"
)

print(
    f"Registered model version: "
    f"{registered_version}"
)


# ============================================================
# WAIT FOR REGISTERED MODEL TO BECOME READY
# ============================================================

for attempt in range(30):

    try:

        version = client.get_model_version(
            name=MODEL_NAME,
            version=str(
                registered_version
            )
        )

        if version.status == "READY":

            print(
                f"Model version "
                f"{registered_version} is READY."
            )

            break

        print(
            f"Model version status: "
            f"{version.status}"
        )

    except Exception:

        print(
            f"Waiting for registered model... "
            f"attempt {attempt + 1}/30"
        )

    time.sleep(2)

else:

    raise RuntimeError(
        "Registered model did not become READY."
    )


# ============================================================
# FINAL CONFIRMATION
# ============================================================

print(
    "=============================================="
)

print(
    "MLflow model registration "
    "completed successfully."
)

print(
    f"Experiment: {EXPERIMENT_NAME}"
)

print(
    f"Model: {MODEL_NAME}"
)

print(
    f"Version: {registered_version}"
)

print(
    f"Model URI: "
    f"models:/{MODEL_NAME}/{registered_version}"
)

print(
    "=============================================="
)