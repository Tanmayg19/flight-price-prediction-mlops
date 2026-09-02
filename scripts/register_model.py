import os
import time
import joblib
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

MODEL_PATH = "models/xgb_flight_price_model.pkl"
MODEL_NAME = "flight-price-xgboost"


print("Connecting to MLflow...")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

# Wait until MLflow is available
for attempt in range(30):
    try:
        client.search_experiments()
        print("MLflow is ready.")
        break
    except Exception:
        print(f"Waiting for MLflow... attempt {attempt + 1}/30")
        time.sleep(2)
else:
    raise RuntimeError("MLflow server did not become available.")


# Create experiment if it does not exist
experiment_name = "flight-price-prediction-ci"

experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = client.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id


# Load trained XGBoost model
print(f"Loading model from: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


# Log model to MLflow
with mlflow.start_run(
    experiment_id=experiment_id,
    run_name="ci-model-registration"
):
    mlflow.xgboost.log_model(
        model,
        name="model"
    )

    run_id = mlflow.active_run().info.run_id

    print(f"Model logged successfully.")
    print(f"Run ID: {run_id}")


# Register model
model_uri = f"runs:/{run_id}/model"

registered = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print(
    f"Registered model: {MODEL_NAME}, "
    f"version: {registered.version}"
)

print("Model registration completed successfully.")