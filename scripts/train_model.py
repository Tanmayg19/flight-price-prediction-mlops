import json
import os
import pickle

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_PATH = "/opt/airflow/project"

PROCESSED_DATA_PATH = os.path.join(
    PROJECT_PATH,
    "data",
    "processed",
)

MODELS_PATH = os.path.join(
    PROJECT_PATH,
    "models",
)

FEATURE_DATA_FILE = os.path.join(
    PROCESSED_DATA_PATH,
    "flights_feature_engineered.csv",
)

FEATURE_COLUMNS_FILE = os.path.join(
    MODELS_PATH,
    "feature_columns.pkl",
)

BEST_PARAMS_FILE = os.path.join(
    MODELS_PATH,
    "best_hyperparameters.json",
)

CANDIDATE_MODEL_FILE = os.path.join(
    MODELS_PATH,
    "xgb_flight_price_candidate.pkl",
)

CANDIDATE_METRICS_FILE = os.path.join(
    MODELS_PATH,
    "candidate_model_metrics.json",
)


# =========================================================
# TRAINING CONFIGURATION
# =========================================================

TARGET_COLUMN = "price"

TEST_SIZE = 0.20

RANDOM_STATE = 42


# =========================================================
# LOAD FEATURE SCHEMA
# =========================================================

def load_feature_columns():
    """
    Load the exact feature schema used by the
    existing flight-price model.
    """

    if not os.path.exists(FEATURE_COLUMNS_FILE):
        raise FileNotFoundError(
            f"Feature schema not found: "
            f"{FEATURE_COLUMNS_FILE}"
        )

    with open(
        FEATURE_COLUMNS_FILE,
        "rb",
    ) as file:
        feature_columns = pickle.load(file)

    feature_columns = list(feature_columns)

    if len(feature_columns) == 0:
        raise ValueError(
            "feature_columns.pkl contains no features."
        )

    print(
        f"Loaded {len(feature_columns)} model features."
    )

    return feature_columns


# =========================================================
# LOAD BEST HYPERPARAMETERS
# =========================================================

def load_best_hyperparameters():
    """
    Load the previously tuned XGBoost
    hyperparameters.
    """

    if not os.path.exists(BEST_PARAMS_FILE):
        raise FileNotFoundError(
            f"Best hyperparameter file not found: "
            f"{BEST_PARAMS_FILE}"
        )

    with open(
        BEST_PARAMS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        best_params = json.load(file)

    if not isinstance(best_params, dict):
        raise ValueError(
            "best_hyperparameters.json must "
            "contain a JSON object."
        )

    if not best_params:
        raise ValueError(
            "best_hyperparameters.json is empty."
        )

    print(
        "Loaded tuned XGBoost hyperparameters:"
    )

    for key, value in best_params.items():
        print(
            f"  {key}: {value}"
        )

    return best_params


# =========================================================
# LOAD TRAINING DATA
# =========================================================

def load_training_data(
    feature_columns,
):
    """
    Load and validate the model-ready dataset.
    """

    if not os.path.exists(FEATURE_DATA_FILE):
        raise FileNotFoundError(
            f"Feature-engineered dataset not found: "
            f"{FEATURE_DATA_FILE}"
        )

    print(
        f"Loading training data: "
        f"{FEATURE_DATA_FILE}"
    )

    df = pd.read_csv(
        FEATURE_DATA_FILE
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    if df.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            f"not found."
        )

    missing_features = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Training dataset is missing "
            f"features: {missing_features}"
        )

    expected_columns = (
        feature_columns
        + [TARGET_COLUMN]
    )

    unexpected_columns = [
        column
        for column in df.columns
        if column not in expected_columns
    ]

    if unexpected_columns:
        raise ValueError(
            f"Unexpected columns found in "
            f"training dataset: "
            f"{unexpected_columns}"
        )

    X = df[
        feature_columns
    ].copy()

    y = df[
        TARGET_COLUMN
    ].copy()

    if X.isnull().sum().sum() > 0:
        raise ValueError(
            "Training features contain null values."
        )

    if y.isnull().sum() > 0:
        raise ValueError(
            "Target contains null values."
        )

    if list(X.columns) != feature_columns:
        raise ValueError(
            "Training feature order does not "
            "match feature_columns.pkl."
        )

    print(
        "PASS: Training dataset validation succeeded."
    )

    print(
        f"Training rows available: {len(X)}"
    )

    print(
        f"Number of model features: {X.shape[1]}"
    )

    return X, y


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    X_train,
    y_train,
    best_params,
):
    """
    Train XGBoost using the previously tuned
    hyperparameters.
    """

    print(
        "Creating XGBoost candidate model..."
    )

    # Copy so we do not modify the dictionary
    # loaded from the JSON file.
    model_params = best_params.copy()

    # Reproducibility
    model_params["random_state"] = RANDOM_STATE

    # Regression objective
    model_params.setdefault(
        "objective",
        "reg:squarederror",
    )

    # Keep Airflow/Docker resource use controlled.
    model_params["n_jobs"] = 2

    model = XGBRegressor(
        **model_params
    )

    print(
        "Starting XGBoost training..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "XGBoost training completed successfully."
    )

    return model


# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate the candidate model using
    R2, RMSE and MAE.
    """

    print(
        "Evaluating candidate model..."
    )

    predictions = model.predict(
        X_test
    )

    r2 = float(
        r2_score(
            y_test,
            predictions,
        )
    )

    rmse = float(
        mean_squared_error(
            y_test,
            predictions,
        ) ** 0.5
    )

    mae = float(
        mean_absolute_error(
            y_test,
            predictions,
        )
    )

    metrics = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }

    print(
        "Candidate model metrics:"
    )

    print(
        f"R2   : {r2:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"MAE  : {mae:.4f}"
    )

    return metrics


# =========================================================
# SAVE MODEL AND METRICS
# =========================================================

def save_training_artifacts(
    model,
    metrics,
):
    """
    Save the candidate model and its evaluation
    metrics.

    We intentionally do NOT overwrite the current
    production model at this stage.
    """

    os.makedirs(
        MODELS_PATH,
        exist_ok=True,
    )

    temporary_model_file = (
        CANDIDATE_MODEL_FILE + ".tmp"
    )

    with open(
        temporary_model_file,
        "wb",
    ) as file:
        pickle.dump(
            model,
            file,
        )

    os.replace(
        temporary_model_file,
        CANDIDATE_MODEL_FILE,
    )

    print(
        f"Candidate model saved: "
        f"{CANDIDATE_MODEL_FILE}"
    )

    temporary_metrics_file = (
        CANDIDATE_METRICS_FILE + ".tmp"
    )

    with open(
        temporary_metrics_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    os.replace(
        temporary_metrics_file,
        CANDIDATE_METRICS_FILE,
    )

    print(
        f"Candidate metrics saved: "
        f"{CANDIDATE_METRICS_FILE}"
    )


# =========================================================
# MAIN TRAINING PIPELINE
# =========================================================

def run_training_pipeline():
    """
    Execute the complete automated model-training
    workflow.
    """

    print(
        "=============================================="
    )

    print(
        "TRAVEL ANALYTICS XGBOOST TRAINING"
    )

    print(
        "=============================================="
    )

    # -----------------------------------------------------
    # Load feature schema
    # -----------------------------------------------------

    feature_columns = (
        load_feature_columns()
    )

    # -----------------------------------------------------
    # Load tuned parameters
    # -----------------------------------------------------

    best_params = (
        load_best_hyperparameters()
    )

    # -----------------------------------------------------
    # Load model-ready data
    # -----------------------------------------------------

    X, y = load_training_data(
        feature_columns
    )

    # -----------------------------------------------------
    # Train / test split
    # -----------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(
        "Train/test split completed."
    )

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows: {len(X_test)}"
    )

    # -----------------------------------------------------
    # Train candidate model
    # -----------------------------------------------------

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        best_params=best_params,
    )

    # -----------------------------------------------------
    # Evaluate candidate model
    # -----------------------------------------------------

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    # -----------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------

    save_training_artifacts(
        model=model,
        metrics=metrics,
    )

    print(
        "=============================================="
    )

    print(
        "MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY"
    )

    print(
        "=============================================="
    )


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":
    run_training_pipeline()