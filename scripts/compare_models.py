import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


# ============================================================
# Paths
# ============================================================

PROJECT_PATH = "/opt/airflow/project"

FEATURE_DATA_PATH = os.path.join(
    PROJECT_PATH,
    "data",
    "processed",
    "flights_feature_engineered.csv",
)

FEATURE_COLUMNS_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "feature_columns.pkl",
)

PRODUCTION_MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "xgb_flight_price_model.pkl",
)

CANDIDATE_MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "xgb_flight_price_candidate.pkl",
)


# ============================================================
# Configuration
# ============================================================

TARGET_COLUMN = "price"
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# Helper functions
# ============================================================

def load_feature_columns():
    print("\nLoading feature columns...")

    if not os.path.exists(FEATURE_COLUMNS_PATH):
        raise FileNotFoundError(
            f"Feature columns file not found: {FEATURE_COLUMNS_PATH}"
        )

    with open(FEATURE_COLUMNS_PATH, "rb") as file:
        feature_columns = pickle.load(file)

    print(f"Loaded {len(feature_columns)} feature columns.")

    return feature_columns


def load_dataset(feature_columns):
    print("\nLoading feature-engineered dataset...")

    if not os.path.exists(FEATURE_DATA_PATH):
        raise FileNotFoundError(
            f"Feature-engineered data not found: {FEATURE_DATA_PATH}"
        )

    df = pd.read_csv(FEATURE_DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    expected_columns = feature_columns + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    unexpected_columns = [
        column
        for column in df.columns
        if column not in expected_columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    if unexpected_columns:
        raise ValueError(
            f"Unexpected columns found: {unexpected_columns}"
        )

    if df.isnull().any().any():
        raise ValueError(
            "Null values found in feature-engineered dataset."
        )

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y


def create_test_split(X, y):
    print("\nCreating train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")

    return X_train, X_test, y_train, y_test


def load_model(model_path, model_name):
    print(f"\nLoading {model_name}...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_name} not found: {model_path}"
        )

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    print(f"{model_name} loaded successfully.")

    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
    }


def print_metrics(model_name, metrics):
    print(f"\n{model_name}")
    print("-" * 40)
    print(f"R2:   {metrics['r2']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")


# ============================================================
# Main comparison pipeline
# ============================================================

def compare_models():

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    feature_columns = load_feature_columns()

    X, y = load_dataset(feature_columns)

    _, X_test, _, y_test = create_test_split(
        X,
        y,
    )

    production_model = load_model(
        PRODUCTION_MODEL_PATH,
        "Existing Production Model",
    )

    candidate_model = load_model(
        CANDIDATE_MODEL_PATH,
        "New Candidate Model",
    )

    print("\nEvaluating both models on the SAME test set...")

    production_metrics = evaluate_model(
        production_model,
        X_test,
        y_test,
    )

    candidate_metrics = evaluate_model(
        candidate_model,
        X_test,
        y_test,
    )

    print("\n" + "=" * 60)

    print_metrics(
        "EXISTING PRODUCTION MODEL",
        production_metrics,
    )

    print_metrics(
        "NEW CANDIDATE MODEL",
        candidate_metrics,
    )

    print("\n" + "=" * 60)

    print("COMPARISON")
    print("-" * 40)

    r2_difference = (
        candidate_metrics["r2"]
        - production_metrics["r2"]
    )

    rmse_difference = (
        production_metrics["rmse"]
        - candidate_metrics["rmse"]
    )

    mae_difference = (
        production_metrics["mae"]
        - candidate_metrics["mae"]
    )

    print(f"R2 improvement:   {r2_difference:.6f}")
    print(f"RMSE reduction:   {rmse_difference:.6f}")
    print(f"MAE reduction:    {mae_difference:.6f}")

    print("=" * 60)


if __name__ == "__main__":
    compare_models()