import os
import time
import joblib
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

MODEL_PATH = "models/xgb_flight_price_model.pkl"
MODEL_NAME = "flight-price-xgboost"


# --------------------------------------------------
# Connect to MLflow
# --------------------------------------------------

print("Connecting to MLflow...")

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

client = MlflowClient()


# --------------------------------------------------
# Wait for MLflow
# --------------------------------------------------

for attempt in range(30):

    try:

        client.search_experiments()

        print("MLflow is ready.")

        break

    except Exception as e:

        print(
            f"Waiting for MLflow... "
            f"attempt {attempt + 1}/30"
        )

        time.sleep(2)

else:

    raise RuntimeError(
        "MLflow server did not become available."
    )


# --------------------------------------------------
# Create experiment if it does not exist
# --------------------------------------------------

experiment_name = "flight-price-prediction-ci"

experiment = client.get_experiment_by_name(
    experiment_name
)

if experiment is None:

    experiment_id = client.create_experiment(
        experiment_name
    )

else:

    experiment_id = experiment.experiment_id


print(
    f"Using experiment: {experiment_name}"
)

print(
    f"Experiment ID: {experiment_id}"
)


# --------------------------------------------------
# Load trained XGBoost model
# --------------------------------------------------

print(
    f"Loading model from: {MODEL_PATH}"
)

model = joblib.load(
    MODEL_PATH
)

print(
    "XGBoost model loaded successfully."
)


# --------------------------------------------------
# Log and register model in MLflow
# --------------------------------------------------

print(
    "Logging model to MLflow..."
)

with mlflow.start_run(
    experiment_id=experiment_id,
    run_name="ci-model-registration"
):

    model_info = mlflow.xgboost.log_model(
        model,
        name="model",
        registered_model_name=MODEL_NAME,
        await_registration_for=300
    )

    run_id = mlflow.active_run().info.run_id


# --------------------------------------------------
# Display model information
# --------------------------------------------------

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
    f"Model ID: {model_info.model_id}"
)

print(
    f"Registered model: {MODEL_NAME}"
)

print(
    f"Registered model version: "
    f"{model_info.registered_model_version}"
)


# --------------------------------------------------
# Verify registered model
# --------------------------------------------------

registered_version = (
    model_info.registered_model_version
)

if registered_version is None:

    raise RuntimeError(
        "Model was logged but was not registered "
        "in the MLflow Model Registry."
    )


print(
    "Model registration verified successfully."
)


# --------------------------------------------------
# Verify model version exists
# --------------------------------------------------

for attempt in range(30):

    try:

        version = client.get_model_version(
            name=MODEL_NAME,
            version=str(registered_version)
        )

        if version.status == "READY":

            print(
                f"Model version {registered_version} "
                f"is READY."
            )

            break

        print(
            f"Model version status: "
            f"{version.status}"
        )

    except Exception as e:

        print(
            f"Waiting for registered model... "
            f"attempt {attempt + 1}/30"
        )

    time.sleep(2)

else:

    raise RuntimeError(
        "Registered model did not become READY."
    )


# --------------------------------------------------
# Final confirmation
# --------------------------------------------------

print(
    "=============================================="
)

print(
    "MLflow model registration completed successfully."
)

print(
    f"Model: {MODEL_NAME}"
)

print(
    f"Version: {registered_version}"
)

print(
    f"Model URI: models:/{MODEL_NAME}/{registered_version}"
)

print(
    "=============================================="
)