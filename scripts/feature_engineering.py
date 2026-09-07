import os
import pickle
import pandas as pd


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

INPUT_FILE = os.path.join(
    PROCESSED_DATA_PATH,
    "flights.csv",
)

OUTPUT_FILE = os.path.join(
    PROCESSED_DATA_PATH,
    "flights_feature_engineered.csv",
)

FEATURE_COLUMNS_FILE = os.path.join(
    MODELS_PATH,
    "feature_columns.pkl",
)


# =========================================================
# REQUIRED SOURCE COLUMNS
# =========================================================

REQUIRED_COLUMNS = [
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


# =========================================================
# LOAD EXPECTED MODEL FEATURES
# =========================================================

def load_expected_features():
    """
    Load the feature-column schema used by the
    existing XGBoost model.
    """

    if not os.path.exists(FEATURE_COLUMNS_FILE):
        raise FileNotFoundError(
            f"Feature-column file not found: "
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
        f"Loaded {len(feature_columns)} "
        f"expected model features."
    )

    print(
        "Expected model features:"
    )

    for feature in feature_columns:
        print(
            f"  - {feature}"
        )

    return feature_columns


# =========================================================
# VALIDATE SOURCE DATA
# =========================================================

def validate_source_dataframe(df):
    """
    Validate that processed flights.csv contains
    all fields required for feature engineering.
    """

    print(
        "Validating feature-engineering source data..."
    )

    if df.empty:
        raise ValueError(
            "Processed flights dataset is empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing from flights.csv: "
            f"{missing_columns}"
        )

    if df["price"].isnull().any():
        raise ValueError(
            "Target column 'price' contains null values."
        )

    if df["time"].isnull().any():
        raise ValueError(
            "Feature column 'time' contains null values."
        )

    if df["distance"].isnull().any():
        raise ValueError(
            "Feature column 'distance' contains null values."
        )

    categorical_columns = [
        "flightType",
        "agency",
        "from",
        "to",
    ]

    for column in categorical_columns:

        if df[column].isnull().any():
            raise ValueError(
                f"Categorical column '{column}' "
                f"contains null values."
            )

    print(
        "PASS: Feature-engineering source data is valid."
    )


# =========================================================
# CREATE FEATURES
# =========================================================

def create_features(df):
    """
    Convert raw flight attributes into the model-ready
    feature representation.
    """

    print(
        "Starting feature engineering..."
    )

    # -----------------------------------------------------
    # Numerical features
    # -----------------------------------------------------

    numeric_features = df[
        [
            "time",
            "distance",
        ]
    ].copy()

    print(
        "Created numerical features:"
    )

    print(
        list(numeric_features.columns)
    )

    # -----------------------------------------------------
    # Categorical features
    # -----------------------------------------------------

    categorical_features = df[
        [
            "flightType",
            "agency",
            "from",
            "to",
        ]
    ].copy()

    encoded_features = pd.get_dummies(
        categorical_features,
        columns=[
            "flightType",
            "agency",
            "from",
            "to",
        ],
        dtype=int,
    )

    print(
        f"Created {encoded_features.shape[1]} "
        f"one-hot encoded columns."
    )

    # -----------------------------------------------------
    # Combine numerical + categorical features
    # -----------------------------------------------------

    X = pd.concat(
        [
            numeric_features,
            encoded_features,
        ],
        axis=1,
    )

    print(
        f"Feature matrix before schema alignment: "
        f"{X.shape}"
    )

    return X


# =========================================================
# ALIGN FEATURES WITH MODEL SCHEMA
# =========================================================

def align_feature_schema(
    X,
    expected_features,
):
    """
    Force generated features to match the exact
    training feature schema and column order.
    """

    unexpected_features = [
        column
        for column in X.columns
        if column not in expected_features
    ]

    if unexpected_features:
        print(
            "WARNING: Ignoring features that are not "
            "part of the trained model schema:"
        )

        for feature in unexpected_features:
            print(
                f"  - {feature}"
            )

    missing_features = [
        column
        for column in expected_features
        if column not in X.columns
    ]

    if missing_features:
        print(
            "Features absent from current data will "
            "be created with value 0:"
        )

        for feature in missing_features:
            print(
                f"  - {feature}"
            )

    X = X.reindex(
        columns=expected_features,
        fill_value=0,
    )

    if list(X.columns) != expected_features:
        raise ValueError(
            "Final feature schema does not match "
            "the expected model schema."
        )

    print(
        "PASS: Feature schema matches the model."
    )

    return X


# =========================================================
# VALIDATE FINAL FEATURE DATA
# =========================================================

def validate_feature_dataframe(
    X,
    y,
    expected_features,
):
    """
    Validate generated model-ready data.
    """

    if X.empty:
        raise ValueError(
            "Feature-engineered dataframe is empty."
        )

    if len(X) != len(y):
        raise ValueError(
            "Feature rows and target rows do not match."
        )

    if X.isnull().sum().sum() > 0:
        raise ValueError(
            "Feature-engineered data contains null values."
        )

    if len(X.columns) != len(expected_features):
        raise ValueError(
            f"Expected {len(expected_features)} model "
            f"features but generated {len(X.columns)}."
        )

    if list(X.columns) != expected_features:
        raise ValueError(
            "Feature column order does not match "
            "feature_columns.pkl."
        )

    print(
        "PASS: Final feature-engineered dataset "
        "passed validation."
    )

    print(
        f"Rows: {X.shape[0]}"
    )

    print(
        f"Model features: {X.shape[1]}"
    )


# =========================================================
# MAIN FEATURE ENGINEERING PIPELINE
# =========================================================

def run_feature_engineering():
    """
    Execute the complete feature-engineering workflow.
    """

    print(
        "=============================================="
    )

    print(
        "TRAVEL ANALYTICS FEATURE ENGINEERING"
    )

    print(
        "=============================================="
    )

    # -----------------------------------------------------
    # Check input
    # -----------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Processed flights file not found: "
            f"{INPUT_FILE}"
        )

    # -----------------------------------------------------
    # Read validated flights
    # -----------------------------------------------------

    flights = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Loaded processed flights dataset: "
        f"{flights.shape}"
    )

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    validate_source_dataframe(
        flights
    )

    # -----------------------------------------------------
    # Load expected model feature schema
    # -----------------------------------------------------

    expected_features = (
        load_expected_features()
    )

    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------

    y = flights[
        "price"
    ].copy()

    # -----------------------------------------------------
    # Build model features
    # -----------------------------------------------------

    X = create_features(
        flights
    )

    # -----------------------------------------------------
    # Match existing model schema
    # -----------------------------------------------------

    X = align_feature_schema(
        X=X,
        expected_features=expected_features,
    )

    # -----------------------------------------------------
    # Validate output
    # -----------------------------------------------------

    validate_feature_dataframe(
        X=X,
        y=y,
        expected_features=expected_features,
    )

    # -----------------------------------------------------
    # Add target for training dataset
    # -----------------------------------------------------

    feature_engineered_df = (
        X.copy()
    )

    feature_engineered_df[
        "price"
    ] = y.values

    # -----------------------------------------------------
    # Write safely
    # -----------------------------------------------------

    os.makedirs(
        PROCESSED_DATA_PATH,
        exist_ok=True,
    )

    temporary_output = (
        OUTPUT_FILE + ".tmp"
    )

    feature_engineered_df.to_csv(
        temporary_output,
        index=False,
    )

    os.replace(
        temporary_output,
        OUTPUT_FILE,
    )

    print(
        "Feature-engineered dataset saved successfully:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        f"Final dataset shape: "
        f"{feature_engineered_df.shape}"
    )

    print(
        f"Model feature count: "
        f"{len(expected_features)}"
    )

    print(
        "Target column: price"
    )

    print(
        "Feature engineering completed successfully."
    )


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":
    run_feature_engineering()