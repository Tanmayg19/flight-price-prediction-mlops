# ============================================================
# ✈️ FLIGHT PRICE PREDICTION APPLICATION
# ============================================================
# Streamlit + REST API + MLflow + XGBoost
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import streamlit as st
import mlflow
import mlflow.xgboost
import pandas as pd
import joblib
import requests
import os

from pathlib import Path


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_FEATURES_PATH = (
    BASE_DIR / "models" / "feature_columns.pkl"
)


# --------------------------------------------------
# FastAPI Configuration
# --------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

# ============================================================
# 5. LOAD FEATURE COLUMNS
# ============================================================

@st.cache_resource
def load_feature_columns():

    feature_columns = joblib.load(
        MODEL_FEATURES_PATH
    )

    return feature_columns


try:

    feature_columns = load_feature_columns()

except Exception as e:

    st.error(
        "Unable to load model feature columns."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 6. APPLICATION TITLE
# ============================================================

st.title("✈️ Flight Price Predictor")

st.markdown(
    """
    ### ML-powered flight price estimation

    Enter your flight details below and use our
    **XGBoost machine learning model** to estimate
    the flight price.

    The prediction request is sent through a
    **REST API**, while the model is managed using
    **MLflow Model Registry**.
    """
)


# ============================================================
# 7. CHECK FASTAPI CONNECTION
# ============================================================

def check_api_health():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        if response.status_code == 200:

            return response.json()

        return None

    except requests.exceptions.RequestException:

        return None


health_data = check_api_health()


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("✈️ Flight Price Predictor")

    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Prediction",
            "📊 Model Insights",
            "ℹ️ About"
        ]
    )

    st.markdown("---")

    # REST API status

    if health_data:

        if health_data.get("status") == "healthy":

            st.success(
                "🟢 REST API Connected"
            )

            st.caption(
                f"Model: {health_data.get('model_name')}"
            )

            st.caption(
                f"Version: {health_data.get('model_version')}"
            )

        else:

            st.error(
                "🔴 API Model Unhealthy"
            )

    else:

        st.error(
            "🔴 REST API Offline"
        )

    st.markdown("---")

    st.caption(
        "Streamlit + FastAPI + MLflow + XGBoost"
    )


# ============================================================
# ============================================================
# PAGE 1 — PREDICTION
# ============================================================

if page == "🏠 Prediction":

    st.header("✈️ Flight Price Prediction")

    st.markdown(
        """
        Enter the details of your flight below and our
        **XGBoost machine learning model** will estimate
        the flight price.
        """
    )

    # --------------------------------------------------------
    # CHECK FASTAPI CONNECTION
    # --------------------------------------------------------

    try:

        health_response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        if health_response.status_code == 200:

            health_data = health_response.json()

            if health_data.get("status") == "healthy":

                st.success(
                    f"✅ REST API Connected | "
                    f"Model: {health_data.get('model_name')} | "
                    f"Version: {health_data.get('model_version')}"
                )

            else:

                st.error(
                    "❌ FastAPI is running, "
                    "but the model is not healthy."
                )

        else:

            st.error(
                "❌ FastAPI is not responding correctly."
            )

    except requests.exceptions.RequestException:

        st.error(
            "❌ Cannot connect to FastAPI.\n\n"
            "Please make sure the API is running with:\n\n"
            "`uvicorn api.app:app --reload --port 8000`"
        )

    # ========================================================
    # FLIGHT INPUT FORM
    # ========================================================

    st.subheader("📝 Enter Flight Details")

    with st.form("flight_prediction_form"):

        # ----------------------------------------------------
        # TRAVEL DATE
        # ----------------------------------------------------

        travel_date = st.date_input(
            "📅 Travel Date"
        )

        st.caption(
            "Travel date is collected for the user interface. "
            "Date-related features were excluded during feature "
            "engineering because EDA showed minimal impact on price."
        )

        # ----------------------------------------------------
        # TWO-COLUMN INPUT LAYOUT
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        # ====================================================
        # LEFT COLUMN
        # ====================================================

        with col1:

            # ------------------------------------------------
            # FLIGHT TYPE
            # ------------------------------------------------

            flight_type = st.selectbox(
                "✈️ Flight Type",
                [
                    "First Class",
                    "Economic",
                    "Premium"
                ]
            )

            # ------------------------------------------------
            # AIRLINE AGENCY
            # ------------------------------------------------

            agency = st.selectbox(
                "🏢 Airline Agency",
                [
                    "FlyingDrops",
                    "CloudFy",
                    "Rainbow"
                ]
            )

            # ------------------------------------------------
            # ORIGIN
            # ------------------------------------------------

            origin = st.selectbox(
                "📍 Origin",
                [
                    "Florianopolis (SC)",
                    "Sao Paulo (SP)",
                    "Brasilia (DF)",
                    "Campo Grande (MS)",
                    "Natal (RN)",
                    "Salvador (BH)",
                    "Aracaju (SE)",
                    "Recife (PE)",
                    "Rio de Janeiro (RJ)"
                ]
            )

        # ====================================================
        # RIGHT COLUMN
        # ====================================================

        with col2:

            # ------------------------------------------------
            # DESTINATION
            # ------------------------------------------------

            destination = st.selectbox(
                "🎯 Destination",
                [
                    "Sao Paulo (SP)",
                    "Salvador (BH)",
                    "Brasilia (DF)",
                    "Aracaju (SE)",
                    "Florianopolis (SC)",
                    "Rio de Janeiro (RJ)",
                    "Campo Grande (MS)",
                    "Natal (RN)",
                    "Recife (PE)"
                ]
            )

            # ------------------------------------------------
            # DISTANCE
            # ------------------------------------------------

            distance = st.number_input(
                "📏 Distance",
                min_value=0.0,
                value=500.0,
                step=10.0
            )

            # ------------------------------------------------
            # FLIGHT TIME
            # ------------------------------------------------

            travel_time = st.number_input(
                "⏱️ Flight Time",
                min_value=0.0,
                value=2.0,
                step=0.1
            )

        # ----------------------------------------------------
        # PREDICTION BUTTON
        # ----------------------------------------------------

        st.markdown("")

        predict_button = st.form_submit_button(
            "🔮 Predict Flight Price",
            use_container_width=True
        )

    # ========================================================
    # CREATE MODEL INPUT AND CALL FASTAPI
    # ========================================================

    if predict_button:

        try:

            # ------------------------------------------------
            # CREATE DATAFRAME WITH ALL 26 FEATURES
            # ------------------------------------------------

            input_data = pd.DataFrame(
                0,
                index=[0],
                columns=feature_columns
            )

            # ------------------------------------------------
            # NUMERICAL FEATURES
            # ------------------------------------------------

            input_data["time"] = float(travel_time)

            input_data["distance"] = float(distance)

            # ------------------------------------------------
            # FLIGHT TYPE
            # ------------------------------------------------

            flight_type_column = {

                "First Class":
                    "flightType_firstClass",

                "Economic":
                    "flightType_economic",

                "Premium":
                    "flightType_premium"
            }

            selected_flight_type = (
                flight_type_column[flight_type]
            )

            if selected_flight_type in input_data.columns:

                input_data[selected_flight_type] = 1

            # ------------------------------------------------
            # AIRLINE AGENCY
            # ------------------------------------------------

            agency_column = f"agency_{agency}"

            if agency_column in input_data.columns:

                input_data[agency_column] = 1

            # ------------------------------------------------
            # ORIGIN
            # ------------------------------------------------

            origin_column = f"from_{origin}"

            if origin_column in input_data.columns:

                input_data[origin_column] = 1

            # ------------------------------------------------
            # DESTINATION
            # ------------------------------------------------

            destination_column = f"to_{destination}"

            if destination_column in input_data.columns:

                input_data[destination_column] = 1

            # ------------------------------------------------
            # VERIFY FEATURE COUNT
            # ------------------------------------------------

            if len(input_data.columns) != 26:

                st.error(
                    f"❌ Expected 26 model features, "
                    f"but found {len(input_data.columns)}."
                )

                st.stop()

            # ------------------------------------------------
            # CONVERT MODEL INPUT TO DICTIONARY
            # ------------------------------------------------

            features_dict = (
                input_data.iloc[0]
                .astype(float)
                .to_dict()
            )

            # ------------------------------------------------
            # CREATE API PAYLOAD
            # ------------------------------------------------

            payload = {
                "features": features_dict
            }

            # ------------------------------------------------
            # SEND REQUEST TO FASTAPI
            # ------------------------------------------------

            with st.spinner(
                "🔄 Sending request to the prediction API..."
            ):

                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=30
                )

            # =================================================
            # PROCESS API RESPONSE
            # =================================================

            if response.status_code == 200:

                result = response.json()

                predicted_price = float(
                    result["predicted_price"]
                )

                # ------------------------------------------------
                # SUCCESS MESSAGE
                # ------------------------------------------------

                st.success(
                    "✅ Prediction generated successfully!"
                )

                st.markdown("---")

                # ------------------------------------------------
                # PREDICTION RESULT
                # ------------------------------------------------

                st.subheader(
                    "💰 Estimated Flight Price"
                )

                result_col1, result_col2, result_col3 = (
                    st.columns(3)
                )

                with result_col1:

                    st.metric(
                        "Predicted Price",
                        f"${predicted_price:,.2f}"
                    )

                with result_col2:

                    st.metric(
                        "Model",
                        "XGBoost"
                    )

                with result_col3:

                    st.metric(
                        "MLflow Version",
                        "1"
                    )

                # ------------------------------------------------
                # PREDICTION SUMMARY
                # ------------------------------------------------

                st.markdown("---")

                st.subheader(
                    "📋 Prediction Summary"
                )

                summary_col1, summary_col2 = (
                    st.columns(2)
                )

                with summary_col1:

                    st.write(
                        f"**Travel Date:** "
                        f"{travel_date.strftime('%d %B %Y')}"
                    )

                    st.write(
                        f"**Flight Type:** "
                        f"{flight_type}"
                    )

                    st.write(
                        f"**Airline Agency:** "
                        f"{agency}"
                    )

                with summary_col2:

                    st.write(
                        f"**Origin:** "
                        f"{origin}"
                    )

                    st.write(
                        f"**Destination:** "
                        f"{destination}"
                    )

                    st.write(
                        f"**Distance:** "
                        f"{distance:,.1f}"
                    )

                    st.write(
                        f"**Flight Time:** "
                        f"{travel_time:.1f}"
                    )

                # ------------------------------------------------
                # API INFORMATION
                # ------------------------------------------------

                with st.expander(
                    "🔗 REST API Response"
                ):

                    st.json(result)

                # ------------------------------------------------
                # MODEL INPUT
                # ------------------------------------------------

                with st.expander(
                    "🔍 View 26 Model Features"
                ):

                    st.dataframe(
                        input_data,
                        use_container_width=True
                    )

            else:

                # ------------------------------------------------
                # API ERROR
                # ------------------------------------------------

                st.error(
                    "❌ Prediction API returned an error."
                )

                st.code(
                    response.text
                )

        # =====================================================
        # CONNECTION ERROR
        # =====================================================

        except requests.exceptions.ConnectionError:

            st.error(
                """
                ❌ Could not connect to the FastAPI server.

                Please make sure FastAPI is running:

                `uvicorn api.app:app --reload --port 8000`
                """
            )

        # =====================================================
        # TIMEOUT ERROR
        # =====================================================

        except requests.exceptions.Timeout:

            st.error(
                "❌ The prediction API request timed out."
            )

        # =====================================================
        # OTHER ERRORS
        # =====================================================

        except Exception as e:

            st.error(
                "❌ Something went wrong while "
                "generating the prediction."
            )

            st.exception(e)

# ============================================================
# PAGE 2 — MODEL INSIGHTS
# ============================================================

elif page == "📊 Model Insights":

    st.header("📊 Model Insights")


    st.markdown(
        """
        The flight price prediction model was developed
        using **XGBoost** and is managed using
        **MLflow Model Registry**.

        The Streamlit application communicates with the
        trained model through a **FastAPI REST API**.
        """
    )


    # --------------------------------------------------------
    # GET MODEL HEALTH
    # --------------------------------------------------------

    try:

        health_response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )


        if health_response.status_code == 200:

            health_data = health_response.json()


        else:

            health_data = None


    except Exception:

        health_data = None


    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    st.subheader(
        "🟢 Model Status"
    )


    if health_data:

        status_col1, status_col2, status_col3 = (
            st.columns(3)
        )


        with status_col1:

            st.metric(
                "Status",
                health_data.get(
                    "model_status",
                    "Unknown"
                )
            )


        with status_col2:

            st.metric(
                "Model",
                health_data.get(
                    "model_name",
                    "Unknown"
                )
            )


        with status_col3:

            st.metric(
                "Version",
                health_data.get(
                    "model_version",
                    "Unknown"
                )
            )


    else:

        st.error(
            "Unable to retrieve model status from FastAPI."
        )


    st.markdown("---")


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.subheader(
        "📈 Model Performance"
    )


    metric1, metric2, metric3 = (
        st.columns(3)
    )


    with metric1:

        st.metric(
            "R² Score",
            "0.9784"
        )


    with metric2:

        st.metric(
            "RMSE",
            "62.30"
        )


    with metric3:

        st.metric(
            "MAE",
            "50.58"
        )


    st.markdown("---")


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.subheader(
        "🤖 Model Information"
    )


    info_col1, info_col2 = (
        st.columns(2)
    )


    with info_col1:

        st.write(
            "**Algorithm:** XGBoost Regressor"
        )

        st.write(
            "**Model Registry:** MLflow"
        )

        st.write(
            "**Registered Model:** "
            "`flight-price-xgboost`"
        )


    with info_col2:

        st.write(
            "**Model Version:** 1"
        )

        st.write(
            "**Number of Features:** "
            f"{len(feature_columns)}"
        )

        st.write(
            "**Model Status:** 🟢 READY"
        )


    st.markdown("---")


    # ========================================================
    # REST API INFORMATION
    # ========================================================

    st.subheader(
        "🌐 REST API"
    )


    api_col1, api_col2 = (
        st.columns(2)
    )


    with api_col1:

        st.write(
            "**API Base URL:**"
        )

        st.code(
            API_URL
        )


        st.write(
            "**Health Endpoint:**"
        )

        st.code(
            f"{API_URL}/health"
        )


    with api_col2:

        st.write(
            "**Prediction Endpoint:**"
        )

        st.code(
            f"{API_URL}/predict"
        )


        st.write(
            "**API Documentation:**"
        )

        st.code(
            f"{API_URL}/docs"
        )


    st.markdown("---")


    # ========================================================
    # MLFLOW INFORMATION
    # ========================================================

    st.subheader(
        "🔬 MLflow Integration"
    )


    st.markdown(
        """
        MLflow is responsible for:

        - Experiment tracking
        - Parameter logging
        - Metric logging
        - Model logging
        - Model versioning
        - Model Registry

        The registered model is:

        **`flight-price-xgboost` — Version 1**
        """
    )


    st.write(
        "**MLflow Model URI:**"
    )

    st.code(
        "models:/flight-price-xgboost/1"
    )


    st.markdown("---")


    # ========================================================
    # FEATURE INFORMATION
    # ========================================================

    st.subheader(
        "🧩 Model Features"
    )


    st.write(
        f"The model uses **{len(feature_columns)} features**."
    )


    with st.expander(
        "View all model features"
    ):

        for i, feature in enumerate(
            feature_columns,
            start=1
        ):

            st.write(
                f"{i}. `{feature}`"
            )


# ============================================================
# PAGE 3 — ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.header(
        "ℹ️ About This Project"
    )


    st.markdown(
        """
        ## ✈️ Flight Price Prediction

        This project is an end-to-end machine learning
        solution designed to predict flight prices using
        an **XGBoost regression model**.

        The project demonstrates the complete journey
        from data preparation and exploratory analysis
        to machine learning, experiment tracking,
        model registry, REST API and interactive
        application deployment.
        """
    )


    st.markdown("---")


    # ========================================================
    # TECHNOLOGY STACK
    # ========================================================

    st.subheader(
        "🛠️ Technology Stack"
    )


    tech_col1, tech_col2 = (
        st.columns(2)
    )


    with tech_col1:

        st.markdown(
            """
            **Data & Machine Learning**

            - Python
            - Pandas
            - NumPy
            - Scikit-learn
            - XGBoost
            """
        )


    with tech_col2:

        st.markdown(
            """
            **MLOps & Deployment**

            - MLflow
            - FastAPI
            - REST API
            - Streamlit
            - Docker
            """
        )


    st.markdown("---")


    # ========================================================
    # PROJECT ARCHITECTURE
    # ========================================================

    st.subheader(
        "🏗️ Project Architecture"
    )


    st.markdown(
        """
        **User**

        ↓

        **Streamlit Application**

        ↓

        **FastAPI REST API**

        ↓

        **MLflow Model Registry**

        ↓

        **XGBoost Model**

        ↓

        **Predicted Flight Price**
        """
    )


    st.markdown("---")


    # ========================================================
    # MLFLOW
    # ========================================================

    st.subheader(
        "🔬 MLflow Integration"
    )


    st.markdown(
        """
        MLflow is used for:

        - Experiment tracking
        - Model training runs
        - Parameter logging
        - Metric logging
        - Model artifact management
        - Model registration
        - Model versioning

        The registered model used by the REST API is:

        **`flight-price-xgboost` — Version 1**
        """
    )


    st.markdown("---")


    # ========================================================
    # REST API
    # ========================================================

    st.subheader(
        "🌐 REST API Integration"
    )


    st.markdown(
        """
        FastAPI provides a REST API layer between the
        Streamlit application and the machine learning model.

        This architecture separates:

        **Frontend → API → Model**

        making the machine learning model easier to
        test, maintain and deploy independently.
        """
    )


    st.markdown("---")


    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    st.subheader(
        "📊 Model Summary"
    )


    st.write(
        """
        The final tuned XGBoost model achieved an
        R² score of approximately **0.9784**, indicating
        that the model explains a very high proportion
        of the variation in flight prices.
        """
    )


    st.markdown("---")


    st.caption(
        "Flight Price Prediction | "
        "Machine Learning & MLOps Project"
    )