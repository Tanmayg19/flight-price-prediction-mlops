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


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_PATH = "/opt/airflow/project"

DATA_PATH = os.path.join(
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

EVALUATION_REPORT_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "model_evaluation_report.json",
)

TARGET_COLUMN = "price"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with open(path, "rb") as file:
        return pickle.load(file)


def calculate_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)

    r2 = r2_score(
        y_test,
        predictions,
    )

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


# ---------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------

def run_model_evaluation():

    print("=" * 60)
    print("MODEL EVALUATION AND PERFORMANCE GATE")
    print("=" * 60)

    # -----------------------------------------------------
    # Load feature schema
    # -----------------------------------------------------

    feature_columns = load_pickle(
        FEATURE_COLUMNS_PATH
    )

    print(
        f"\nLoaded {len(feature_columns)} "
        "expected model features."
    )

    # -----------------------------------------------------
    # Load feature-engineered data
    # -----------------------------------------------------

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Feature-engineered data not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(
        f"Dataset shape: {df.shape}"
    )

    expected_columns = (
        list(feature_columns)
        + [TARGET_COLUMN]
    )

    if list(df.columns) != expected_columns:
        raise ValueError(
            "Feature-engineered dataset schema "
            "does not match the expected model schema."
        )

    if df.isnull().any().any():
        raise ValueError(
            "Null values detected in evaluation dataset."
        )

    # -----------------------------------------------------
    # Prepare X and y
    # -----------------------------------------------------

    X = df[feature_columns]

    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Evaluation rows: {len(X_test)}"
    )

    # -----------------------------------------------------
    # Load models
    # -----------------------------------------------------

    print(
        "\nLoading current production model..."
    )

    production_model = load_pickle(
        PRODUCTION_MODEL_PATH
    )

    print(
        "Loading candidate model..."
    )

    candidate_model = load_pickle(
        CANDIDATE_MODEL_PATH
    )

    # -----------------------------------------------------
    # Evaluate production model
    # -----------------------------------------------------

    print(
        "\nEvaluating current production model..."
    )

    production_metrics = calculate_metrics(
        production_model,
        X_test,
        y_test,
    )

    # -----------------------------------------------------
    # Evaluate candidate model
    # -----------------------------------------------------

    print(
        "Evaluating candidate model..."
    )

    candidate_metrics = calculate_metrics(
        candidate_model,
        X_test,
        y_test,
    )

    # -----------------------------------------------------
    # Performance gate
    # -----------------------------------------------------

    candidate_passed = (
        candidate_metrics["r2"]
        > production_metrics["r2"]
        and
        candidate_metrics["rmse"]
        < production_metrics["rmse"]
        and
        candidate_metrics["mae"]
        < production_metrics["mae"]
    )

    decision = (
        "PROMOTE"
        if candidate_passed
        else "REJECT"
    )

    # -----------------------------------------------------
    # Calculate improvements
    # -----------------------------------------------------

    r2_improvement = (
        candidate_metrics["r2"]
        - production_metrics["r2"]
    )

    rmse_reduction = (
        production_metrics["rmse"]
        - candidate_metrics["rmse"]
    )

    mae_reduction = (
        production_metrics["mae"]
        - candidate_metrics["mae"]
    )

    # -----------------------------------------------------
    # Create evaluation report
    # -----------------------------------------------------

    evaluation_report = {
        "production_model": production_metrics,
        "candidate_model": candidate_metrics,
        "improvement": {
            "r2_difference": float(
                r2_improvement
            ),
            "rmse_reduction": float(
                rmse_reduction
            ),
            "mae_reduction": float(
                mae_reduction
            ),
        },
        "performance_gate": {
            "candidate_passed": candidate_passed,
            "decision": decision,
            "rule": (
                "Candidate must have higher R2, "
                "lower RMSE, and lower MAE "
                "than the production model "
                "on the same holdout dataset."
            ),
        },
        "evaluation_config": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "evaluation_rows": len(X_test),
        },
    }

    # -----------------------------------------------------
    # Save report atomically
    # -----------------------------------------------------

    temp_path = (
        EVALUATION_REPORT_PATH
        + ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation_report,
            file,
            indent=4,
        )

    os.replace(
        temp_path,
        EVALUATION_REPORT_PATH,
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("PRODUCTION MODEL")
    print("=" * 60)

    print(
        f"R2   : {production_metrics['r2']:.6f}"
    )
    print(
        f"RMSE : {production_metrics['rmse']:.6f}"
    )
    print(
        f"MAE  : {production_metrics['mae']:.6f}"
    )

    print("\n" + "=" * 60)
    print("CANDIDATE MODEL")
    print("=" * 60)

    print(
        f"R2   : {candidate_metrics['r2']:.6f}"
    )
    print(
        f"RMSE : {candidate_metrics['rmse']:.6f}"
    )
    print(
        f"MAE  : {candidate_metrics['mae']:.6f}"
    )

    print("\n" + "=" * 60)
    print("PERFORMANCE GATE")
    print("=" * 60)

    print(
        f"R2 improvement   : "
        f"{r2_improvement:.6f}"
    )

    print(
        f"RMSE reduction   : "
        f"{rmse_reduction:.6f}"
    )

    print(
        f"MAE reduction    : "
        f"{mae_reduction:.6f}"
    )

    print(
        f"\nDecision: {decision}"
    )

    print(
        "\nEvaluation report saved to:"
    )
    print(
        EVALUATION_REPORT_PATH
    )

    if not candidate_passed:
        raise RuntimeError(
            "Candidate model failed the performance gate. "
            "Model promotion is not allowed."
        )

    print(
        "\nMODEL EVALUATION COMPLETED SUCCESSFULLY. "
        "Candidate passed the performance gate."
    )
    


if __name__ == "__main__":
    run_model_evaluation()