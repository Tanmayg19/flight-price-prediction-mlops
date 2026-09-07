from datetime import datetime
import os
import shutil
import subprocess

import pandas as pd

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator


# =========================================================
# Configuration
# =========================================================

PROJECT_PATH = "/opt/airflow/project"

RAW_DATA_PATH = os.path.join(
    PROJECT_PATH,
    "data",
    "raw",
)

PROCESSED_DATA_PATH = os.path.join(
    PROJECT_PATH,
    "data",
    "processed",
)

FEATURE_ENGINEERING_SCRIPT = os.path.join(
    PROJECT_PATH,
    "scripts",
    "feature_engineering.py",
)

TRAIN_MODEL_SCRIPT = os.path.join(
    PROJECT_PATH,
    "scripts",
    "train_model.py",
)

EVALUATE_MODEL_SCRIPT = os.path.join(
    PROJECT_PATH,
    "scripts",
    "evaluate_candidate.py",
)


USERS_COLUMNS = [
    "code",
    "company",
    "name",
    "gender",
    "age",
]

FLIGHTS_COLUMNS = [
    "travelCode",
    "userCode",
    "from",
    "to",
    "flightType",
    "price",
    "time",
    "distance",
    "agency",
    "date",
]

HOTELS_COLUMNS = [
    "travelCode",
    "userCode",
    "name",
    "place",
    "days",
    "price",
    "total",
    "date",
]


# =========================================================
# Helper functions
# =========================================================

def get_raw_file_path(filename):
    return os.path.join(
        RAW_DATA_PATH,
        filename,
    )


def get_processed_file_path(filename):
    return os.path.join(
        PROCESSED_DATA_PATH,
        filename,
    )


def basic_dataframe_validation(
    df,
    expected_columns,
    dataset_name,
):
    if df.empty:
        raise ValueError(
            f"{dataset_name} dataset is empty."
        )

    if list(df.columns) != expected_columns:
        raise ValueError(
            f"{dataset_name} schema mismatch.\n"
            f"Expected: {expected_columns}\n"
            f"Actual: {list(df.columns)}"
        )

    if df.isnull().any().any():
        null_counts = (
            df.isnull()
            .sum()
        )

        null_counts = null_counts[
            null_counts > 0
        ]

        raise ValueError(
            f"{dataset_name} contains null values:\n"
            f"{null_counts}"
        )

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"{dataset_name} contains "
            f"{duplicate_count} duplicate rows."
        )


# =========================================================
# Task functions
# =========================================================

def check_source_files():
    print("=" * 60)
    print("CHECKING SOURCE FILES")
    print("=" * 60)

    required_files = [
        "users.csv",
        "flights.csv",
        "hotels.csv",
    ]

    for filename in required_files:
        path = get_raw_file_path(
            filename
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required source file not found: {path}"
            )

        print(
            f"Found: {path}"
        )

    print(
        "\nALL REQUIRED SOURCE FILES EXIST."
    )


def validate_users_data():
    print("=" * 60)
    print("VALIDATING USERS DATA")
    print("=" * 60)

    path = get_raw_file_path(
        "users.csv"
    )

    df = pd.read_csv(path)

    print(
        f"Users shape: {df.shape}"
    )

    basic_dataframe_validation(
        df,
        USERS_COLUMNS,
        "Users",
    )

    if not df["code"].is_unique:
        raise ValueError(
            "Users.code must contain unique values."
        )

    invalid_age = ~df["age"].between(
        18,
        100,
    )

    if invalid_age.any():
        raise ValueError(
            "Users contains age values "
            "outside the allowed range 18-100."
        )

    print(
        "USERS DATA VALIDATION PASSED."
    )


def validate_flights_data():
    print("=" * 60)
    print("VALIDATING FLIGHTS DATA")
    print("=" * 60)

    path = get_raw_file_path(
        "flights.csv"
    )

    df = pd.read_csv(path)

    print(
        f"Flights shape: {df.shape}"
    )

    basic_dataframe_validation(
        df,
        FLIGHTS_COLUMNS,
        "Flights",
    )

    for column in [
        "price",
        "time",
        "distance",
    ]:
        if (df[column] <= 0).any():
            raise ValueError(
                f"Flights.{column} must "
                "contain only positive values."
            )

    allowed_flight_types = {
        "economic",
        "premium",
        "firstClass",
    }

    invalid_flight_types = (
        set(df["flightType"].unique())
        - allowed_flight_types
    )

    if invalid_flight_types:
        raise ValueError(
            "Unexpected flight types found: "
            f"{invalid_flight_types}"
        )

    same_origin_destination = (
        df["from"] == df["to"]
    )

    if same_origin_destination.any():
        raise ValueError(
            "Flights contains rows where "
            "origin and destination are identical."
        )

    print(
        "FLIGHTS DATA VALIDATION PASSED."
    )


def validate_hotels_data():
    print("=" * 60)
    print("VALIDATING HOTELS DATA")
    print("=" * 60)

    path = get_raw_file_path(
        "hotels.csv"
    )

    df = pd.read_csv(path)

    print(
        f"Hotels shape: {df.shape}"
    )

    basic_dataframe_validation(
        df,
        HOTELS_COLUMNS,
        "Hotels",
    )

    for column in [
        "days",
        "price",
        "total",
    ]:
        if (df[column] <= 0).any():
            raise ValueError(
                f"Hotels.{column} must "
                "contain only positive values."
            )

    print(
        "HOTELS DATA VALIDATION PASSED."
    )


def cross_table_validation():
    print("=" * 60)
    print("RUNNING CROSS-TABLE VALIDATION")
    print("=" * 60)

    users = pd.read_csv(
        get_raw_file_path(
            "users.csv"
        )
    )

    flights = pd.read_csv(
        get_raw_file_path(
            "flights.csv"
        )
    )

    hotels = pd.read_csv(
        get_raw_file_path(
            "hotels.csv"
        )
    )

    user_codes = set(
        users["code"]
    )

    invalid_flight_users = (
        set(flights["userCode"])
        - user_codes
    )

    if invalid_flight_users:
        raise ValueError(
            "Flights contains userCode values "
            "that do not exist in users.code."
        )

    invalid_hotel_users = (
        set(hotels["userCode"])
        - user_codes
    )

    if invalid_hotel_users:
        raise ValueError(
            "Hotels contains userCode values "
            "that do not exist in users.code."
        )

    flight_travel_codes = set(
        flights["travelCode"]
    )

    invalid_hotel_travel_codes = (
        set(hotels["travelCode"])
        - flight_travel_codes
    )

    if invalid_hotel_travel_codes:
        raise ValueError(
            "Hotels contains travelCode values "
            "that do not exist in flights.travelCode."
        )

    print(
        "CROSS-TABLE VALIDATION PASSED."
    )


def save_validated_data():
    print("=" * 60)
    print("SAVING VALIDATED DATA")
    print("=" * 60)

    os.makedirs(
        PROCESSED_DATA_PATH,
        exist_ok=True,
    )

    filenames = [
        "users.csv",
        "flights.csv",
        "hotels.csv",
    ]

    for filename in filenames:
        source = get_raw_file_path(
            filename
        )

        destination = (
            get_processed_file_path(
                filename
            )
        )

        shutil.copy2(
            source,
            destination,
        )

        print(
            f"Saved: {destination}"
        )

    print(
        "\nVALIDATED DATA SAVED SUCCESSFULLY."
    )


def run_feature_engineering_task():
    print("=" * 60)
    print("RUNNING FEATURE ENGINEERING")
    print("=" * 60)

    if not os.path.exists(
        FEATURE_ENGINEERING_SCRIPT
    ):
        raise FileNotFoundError(
            "Feature engineering script "
            "not found: "
            f"{FEATURE_ENGINEERING_SCRIPT}"
        )

    subprocess.run(
        [
            "python",
            FEATURE_ENGINEERING_SCRIPT,
        ],
        check=True,
    )

    print(
        "\nFEATURE ENGINEERING "
        "COMPLETED SUCCESSFULLY."
    )


def run_model_training_task():
    print("=" * 60)
    print("RUNNING MODEL TRAINING")
    print("=" * 60)

    if not os.path.exists(
        TRAIN_MODEL_SCRIPT
    ):
        raise FileNotFoundError(
            "Training script not found: "
            f"{TRAIN_MODEL_SCRIPT}"
        )

    subprocess.run(
        [
            "python",
            TRAIN_MODEL_SCRIPT,
        ],
        check=True,
    )

    print(
        "\nMODEL TRAINING "
        "COMPLETED SUCCESSFULLY."
    )


def run_model_evaluation_task():
    print("=" * 60)
    print("RUNNING MODEL EVALUATION")
    print("=" * 60)

    if not os.path.exists(
        EVALUATE_MODEL_SCRIPT
    ):
        raise FileNotFoundError(
            "Evaluation script not found: "
            f"{EVALUATE_MODEL_SCRIPT}"
        )

    subprocess.run(
        [
            "python",
            EVALUATE_MODEL_SCRIPT,
        ],
        check=True,
    )

    print(
        "\nMODEL EVALUATION "
        "COMPLETED SUCCESSFULLY."
    )


def pipeline_complete():
    print("=" * 60)
    print("TRAVEL MLOPS PIPELINE COMPLETED")
    print("=" * 60)

    print(
        "Data validation, feature engineering, "
        "model training and model evaluation "
        "completed successfully."
    )


# =========================================================
# DAG definition
# =========================================================

with DAG(
    dag_id="travel_data_ingestion_validation",
    description=(
        "Travel data ingestion, validation, "
        "feature engineering, XGBoost training "
        "and model evaluation."
    ),
    start_date=datetime(
        2026,
        1,
        1,
    ),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=[
        "travel",
        "ingestion",
        "validation",
        "data-quality",
        "feature-engineering",
        "model-training",
        "model-evaluation",
        "xgboost",
        "mlops",
    ],
) as dag:

    check_files_task = PythonOperator(
        task_id="check_source_files",
        python_callable=check_source_files,
    )

    validate_users_task = PythonOperator(
        task_id="validate_users_data",
        python_callable=validate_users_data,
    )

    validate_flights_task = PythonOperator(
        task_id="validate_flights_data",
        python_callable=validate_flights_data,
    )

    validate_hotels_task = PythonOperator(
        task_id="validate_hotels_data",
        python_callable=validate_hotels_data,
    )

    cross_table_validation_task = (
        PythonOperator(
            task_id="cross_table_validation",
            python_callable=cross_table_validation,
        )
    )

    save_validated_data_task = (
        PythonOperator(
            task_id="save_validated_data",
            python_callable=save_validated_data,
        )
    )

    feature_engineering_task = (
        PythonOperator(
            task_id="feature_engineering",
            python_callable=(
                run_feature_engineering_task
            ),
        )
    )

    model_training_task = PythonOperator(
        task_id="model_training",
        python_callable=run_model_training_task,
    )

    model_evaluation_task = (
        PythonOperator(
            task_id="model_evaluation",
            python_callable=(
                run_model_evaluation_task
            ),
        )
    )

    pipeline_complete_task = PythonOperator(
        task_id="pipeline_complete",
        python_callable=pipeline_complete,
    )


# =========================================================
# Dependencies
# =========================================================

check_files_task >> [
    validate_users_task,
    validate_flights_task,
    validate_hotels_task,
]

[
    validate_users_task,
    validate_flights_task,
    validate_hotels_task,
] >> cross_table_validation_task

cross_table_validation_task >> (
    save_validated_data_task
)

save_validated_data_task >> (
    feature_engineering_task
)

feature_engineering_task >> (
    model_training_task
)

model_training_task >> (
    model_evaluation_task
)

model_evaluation_task >> (
    pipeline_complete_task
)