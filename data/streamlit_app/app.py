
# FLIGHT PRICE PREDICTION APPLICATION
# ============================================================
# Streamlit + REST API + MLflow + XGBoost
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import os
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Travel Analytics ML App",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. PROJECT PATHS
# ============================================================

# This application can run in two environments:
#
# Local:
# project/data/streamlit_app/app.py
# project/models/feature_columns.pkl
#
# Docker:
# /app/app.py
# /app/models/feature_columns.pkl

CURRENT_DIR = Path(__file__).resolve().parent

docker_model_path = (
    CURRENT_DIR
    / "models"
    / "feature_columns.pkl"
)

local_model_path = (
    CURRENT_DIR.parent.parent
    / "models"
    / "feature_columns.pkl"
)

if docker_model_path.exists():

    MODEL_FEATURES_PATH = docker_model_path

else:

    MODEL_FEATURES_PATH = local_model_path


# ============================================================
# 4. FASTAPI CONFIGURATION
# ============================================================

# Docker Compose:
# API_URL=http://api:8000
#
# Local:
# http://127.0.0.1:8000

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
# 6. FASTAPI HEALTH CHECK
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


# ------------------------------------------------------------
# Active model information
# ------------------------------------------------------------

if health_data:

    ACTIVE_MODEL_NAME = health_data.get(
        "model_name",
        "Unknown"
    )

    ACTIVE_MODEL_VERSION = health_data.get(
        "model_version",
        "Unknown"
    )

    ACTIVE_MODEL_STATUS = health_data.get(
        "model_status",
        "Unknown"
    )

else:

    ACTIVE_MODEL_NAME = "Unknown"
    ACTIVE_MODEL_VERSION = "Unknown"
    ACTIVE_MODEL_STATUS = "Unknown"


# ============================================================
# 7. APPLICATION TITLE
# ============================================================

st.title("🌍 Travel Analytics ML Application")

st.markdown(
    """
    ### ML-powered travel analytics

    Use the application to estimate **flight prices** with the
    XGBoost regression model or explore **personalized hotel
    recommendations** generated from historical booking behavior.

    Both features are served through the **FastAPI REST API**.
    The flight model is managed using **MLflow Model Registry**.
    """
)


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌍 Travel Analytics")

    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🔮 Prediction",
            "🏨 Hotel Recommendations",
            "👤 Gender Classification",
            "📊 Model Insights",
            "ℹ️ About"
        ]
    )

    st.markdown("---")

    # --------------------------------------------------------
    # REST API status
    # --------------------------------------------------------

    if health_data:

        if health_data.get("status") == "healthy":

            st.success(
                "🟢 REST API Connected"
            )

            st.caption(
                f"Model: {ACTIVE_MODEL_NAME}"
            )

            st.caption(
                f"Version: {ACTIVE_MODEL_VERSION}"
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
# PAGE 1 — PREDICTION
# ============================================================

if page == "🔮 Prediction":

    st.header(
        "✈️ Flight Price Prediction"
    )

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

            prediction_health_data = (
                health_response.json()
            )

            if (
                prediction_health_data.get("status")
                == "healthy"
            ):

                st.success(
                    f"✅ REST API Connected | "
                    f"Model: "
                    f"{prediction_health_data.get('model_name')} | "
                    f"Version: "
                    f"{prediction_health_data.get('model_version')}"
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
            """
            ❌ Cannot connect to FastAPI.

            Please make sure the API is running.
            """
        )


    # ========================================================
    # FLIGHT INPUT FORM
    # ========================================================

    st.subheader(
        "📝 Enter Flight Details"
    )

    with st.form(
        "flight_prediction_form"
    ):

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

        predict_button = (
            st.form_submit_button(
                "🔮 Predict Flight Price",
                use_container_width=True
            )
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

            input_data["time"] = float(
                travel_time
            )

            input_data["distance"] = float(
                distance
            )


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
                flight_type_column[
                    flight_type
                ]
            )

            if (
                selected_flight_type
                in input_data.columns
            ):

                input_data[
                    selected_flight_type
                ] = 1


            # ------------------------------------------------
            # AIRLINE AGENCY
            # ------------------------------------------------

            agency_column = (
                f"agency_{agency}"
            )

            if (
                agency_column
                in input_data.columns
            ):

                input_data[
                    agency_column
                ] = 1


            # ------------------------------------------------
            # ORIGIN
            # ------------------------------------------------

            origin_column = (
                f"from_{origin}"
            )

            if (
                origin_column
                in input_data.columns
            ):

                input_data[
                    origin_column
                ] = 1


            # ------------------------------------------------
            # DESTINATION
            # ------------------------------------------------

            destination_column = (
                f"to_{destination}"
            )

            if (
                destination_column
                in input_data.columns
            ):

                input_data[
                    destination_column
                ] = 1


            # ------------------------------------------------
            # VERIFY FEATURE COUNT
            # ------------------------------------------------

            if len(input_data.columns) != 26:

                st.error(
                    f"❌ Expected 26 model features, "
                    f"but found "
                    f"{len(input_data.columns)}."
                )

                st.stop()


            # ------------------------------------------------
            # CONVERT MODEL INPUT TO DICTIONARY
            # ------------------------------------------------

            features_dict = (
                input_data
                .iloc[0]
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
                "🔄 Sending request "
                "to the prediction API..."
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
                    result[
                        "predicted_price"
                    ]
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

                (
                    result_col1,
                    result_col2,
                    result_col3
                ) = st.columns(3)


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
                        ACTIVE_MODEL_VERSION
                    )


                # ------------------------------------------------
                # PREDICTION SUMMARY
                # ------------------------------------------------

                st.markdown("---")

                st.subheader(
                    "📋 Prediction Summary"
                )

                (
                    summary_col1,
                    summary_col2
                ) = st.columns(2)


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

                    st.json(
                        result
                    )


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

                st.error(
                    "❌ Prediction API "
                    "returned an error."
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

                Please make sure FastAPI is running.
                """
            )


        # =====================================================
        # TIMEOUT ERROR
        # =====================================================

        except requests.exceptions.Timeout:

            st.error(
                "❌ The prediction API "
                "request timed out."
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
# PAGE 2 — HOTEL RECOMMENDATIONS
# ============================================================

elif page == "🏨 Hotel Recommendations":

    st.header("🏨 Personalized Hotel Recommendations")

    st.markdown(
        """
        Discover hotels based on historical booking behavior.

        - **Known users** receive personalized rankings from the tuned SVD recommender.
        - **New or unknown users** receive a popularity-based fallback.
        - A known user may receive fewer recommendations than requested when only a small
          number of unseen hotels remain.

        Recommendations are ranked across the available hotel catalog. They are **not**
        presented as multiple hotel choices within one destination because this dataset
        contains one hotel for each represented destination.
        """
    )

    if not health_data:
        st.error("❌ REST API is offline. Start the API before requesting recommendations.")
    else:
        recommendation_status = health_data.get(
            "recommendation_service",
            "Unknown"
        )

        if recommendation_status == "Ready":
            st.success("✅ Hotel recommendation service is ready")
        else:
            st.warning(
                f"Recommendation service status: {recommendation_status}"
            )

    st.markdown("---")
    st.subheader("👤 Recommendation Request")

    with st.form("hotel_recommendation_form"):
        input_col1, input_col2 = st.columns(2)

        with input_col1:
            user_id = st.number_input(
                "User Code",
                min_value=0,
                value=1104,
                step=1,
                help=(
                    "Enter an existing user code for personalized recommendations. "
                    "An unknown code demonstrates the popularity fallback."
                )
            )

        with input_col2:
            top_n = st.selectbox(
                "Maximum Recommendations",
                options=[1, 2, 3],
                index=2
            )

        recommendation_button = st.form_submit_button(
            "🏨 Get Hotel Recommendations",
            use_container_width=True
        )

    if recommendation_button:
        try:
            with st.spinner("🔄 Getting hotel recommendations..."):
                response = requests.get(
                    f"{API_URL}/recommendations/{int(user_id)}",
                    params={"top_n": int(top_n)},
                    timeout=30
                )

            if response.status_code == 200:
                result = response.json()
                recommendations = result.get("recommendations", [])

                if recommendations:
                    recommendation_type = recommendations[0].get(
                        "recommendation_type",
                        "Unknown"
                    )

                    if recommendation_type == "Personalized SVD":
                        st.success(
                            "✅ Personalized recommendations generated from this "
                            "user's booking history."
                        )
                    else:
                        st.info(
                            "ℹ️ No usable personalized history was found for this "
                            "user, so popular hotels are shown as a fallback."
                        )

                    returned_count = result.get(
                        "returned_recommendations",
                        len(recommendations)
                    )

                    metric_col1, metric_col2, metric_col3 = st.columns(3)

                    with metric_col1:
                        st.metric("User Code", int(user_id))

                    with metric_col2:
                        st.metric("Method", recommendation_type)

                    with metric_col3:
                        st.metric("Hotels Returned", returned_count)

                    if returned_count < int(top_n):
                        st.info(
                            f"{returned_count} recommendation(s) returned instead of "
                            f"{int(top_n)} because this known user has fewer unseen "
                            "hotels available."
                        )

                    st.markdown("---")
                    st.subheader("⭐ Recommended Hotels")

                    for rank, hotel in enumerate(recommendations, start=1):
                        st.markdown(f"### {rank}. {hotel.get('name', 'Unknown Hotel')}")

                        hotel_col1, hotel_col2, hotel_col3 = st.columns(3)

                        with hotel_col1:
                            st.write(
                                f"**📍 Location:** {hotel.get('place', 'Unknown')}"
                            )
                            st.write(
                                f"**💰 Average Price:** "
                                f"${float(hotel.get('average_price', 0)):,.2f}"
                            )

                        with hotel_col2:
                            st.write(
                                f"**🛏️ Average Stay:** "
                                f"{float(hotel.get('average_stay_days', 0)):.2f} days"
                            )
                            st.write(
                                f"**📚 Historical Bookings:** "
                                f"{int(hotel.get('total_bookings', 0)):,}"
                            )

                        with hotel_col3:
                            score = hotel.get("recommendation_score")
                            if score is None:
                                st.write("**🎯 Recommendation Score:** N/A")
                            else:
                                st.write(
                                    f"**🎯 Recommendation Score:** {float(score):.4f}"
                                )
                            st.write(
                                f"**🤖 Method:** "
                                f"{hotel.get('recommendation_type', 'Unknown')}"
                            )

                        st.markdown("---")

                    with st.expander("🔗 REST API Response"):
                        st.json(result)

                else:
                    st.warning("No hotel recommendations were returned.")

            else:
                st.error("❌ Recommendation API returned an error.")
                st.code(response.text)

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Could not connect to the FastAPI server. "
                "Please make sure the API is running."
            )

        except requests.exceptions.Timeout:
            st.error("❌ The recommendation API request timed out.")

        except Exception as e:
            st.error(
                "❌ Something went wrong while generating hotel recommendations."
            )
            st.exception(e)

# ============================================================
# PAGE 3 — GENDER CLASSIFICATION
# ============================================================

elif page == "👤 Gender Classification":

    st.header(
        "👤 Gender Classification"
    )

    st.markdown(
        """
        This page demonstrates the **Gender Classification**
        component of the Travel Analytics project.

        The model uses the available non-identifying features
        **company** and **age** to predict the recorded gender
        label in the modeling dataset.
        """
    )

    # --------------------------------------------------------
    # MODEL LIMITATION NOTICE
    # --------------------------------------------------------

    st.warning(
        """
        ⚠️ **Experimental classification model**

        This classifier is included to demonstrate a complete
        machine-learning classification workflow.

        Its predictive performance is close to chance level,
        so predictions should **not** be interpreted as reliable
        gender inference.
        """
    )

    st.markdown("---")


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.subheader(
        "🤖 Model Information"
    )

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:

        st.metric(
            "Model",
            "Random Forest"
        )

    with info_col2:

        st.metric(
            "Test Accuracy",
            "52.22%"
        )

    with info_col3:

        st.metric(
            "Macro F1",
            "52.20%"
        )


    st.markdown("---")


    # ========================================================
    # GENDER CLASSIFICATION FORM
    # ========================================================

    st.subheader(
        "📝 Enter User Information"
    )

    with st.form(
        "gender_classification_form"
    ):

        input_col1, input_col2 = st.columns(2)


        # ----------------------------------------------------
        # COMPANY
        # ----------------------------------------------------

        with input_col1:

            company = st.selectbox(
                "🏢 Company",
                [
                    "4You",
                    "Acme Factory",
                    "Monsters CYA",
                    "Umbrella LTDA",
                    "Wonka Company"
                ]
            )


        # ----------------------------------------------------
        # AGE
        # ----------------------------------------------------

        with input_col2:

            age = st.number_input(
                "🎂 Age",
                min_value=21,
                max_value=65,
                value=35,
                step=1,
                help=(
                    "The training dataset contains users "
                    "between 21 and 65 years old."
                )
            )


        # ----------------------------------------------------
        # PREDICTION BUTTON
        # ----------------------------------------------------

        st.markdown("")

        gender_predict_button = (
            st.form_submit_button(
                "👤 Generate Classification",
                use_container_width=True
            )
        )


    # ========================================================
    # CALL FASTAPI
    # ========================================================

    if gender_predict_button:

        try:

            payload = {
                "company": company,
                "age": int(age)
            }


            with st.spinner(
                "🔄 Sending request to the classification API..."
            ):

                response = requests.post(
                    f"{API_URL}/gender/predict",
                    json=payload,
                    timeout=30
                )


            # =================================================
            # SUCCESSFUL RESPONSE
            # =================================================

            if response.status_code == 200:

                result = response.json()

                predicted_gender = (
                    result.get(
                        "predicted_gender",
                        "Unknown"
                    )
                )


                st.success(
                    "✅ Classification generated successfully!"
                )

                st.markdown("---")


                # --------------------------------------------
                # CLASSIFICATION RESULT
                # --------------------------------------------

                st.subheader(
                    "🎯 Classification Result"
                )

                result_col1, result_col2 = st.columns(2)


                with result_col1:

                    st.metric(
                        "Predicted Recorded Label",
                        predicted_gender.title()
                    )


                with result_col2:

                    st.metric(
                        "Model",
                        result.get(
                            "model",
                            "Tuned Random Forest"
                        )
                    )


                # --------------------------------------------
                # INPUT SUMMARY
                # --------------------------------------------

                st.markdown("---")

                st.subheader(
                    "📋 Input Summary"
                )

                summary_col1, summary_col2 = st.columns(2)


                with summary_col1:

                    st.write(
                        f"**Company:** {company}"
                    )


                with summary_col2:

                    st.write(
                        f"**Age:** {int(age)}"
                    )


                # --------------------------------------------
                # IMPORTANT INTERPRETATION
                # --------------------------------------------

                st.info(
                    """
                    **Interpretation:** This output represents the
                    model's predicted recorded label based only on
                    age and company.

                    Because the model achieved approximately 52%
                    test accuracy, this prediction should not be
                    treated as a reliable inference about an
                    individual's gender.
                    """
                )


                # --------------------------------------------
                # API RESPONSE
                # --------------------------------------------

                with st.expander(
                    "🔗 REST API Response"
                ):

                    st.json(
                        result
                    )


            # =================================================
            # API ERROR RESPONSE
            # =================================================

            else:

                st.error(
                    "❌ Gender classification API "
                    "returned an error."
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

                Please make sure FastAPI is running.
                """
            )


        # =====================================================
        # TIMEOUT ERROR
        # =====================================================

        except requests.exceptions.Timeout:

            st.error(
                "❌ The gender classification API "
                "request timed out."
            )


        # =====================================================
        # OTHER ERRORS
        # =====================================================

        except Exception as e:

            st.error(
                "❌ Something went wrong while "
                "generating the classification."
            )

            st.exception(e)


# ============================================================
# PAGE 4 — MODEL INSIGHTS
# ============================================================

elif page == "📊 Model Insights":

    st.header(
        "📊 Model Insights"
    )

    st.markdown(
        """

        ### ML-powered travel analytics

        This application brings together three machine-learning workflows:

        - **Flight Price Prediction** using a tuned XGBoost regression model
        - **Personalized Hotel Recommendations** using implicit-feedback Truncated SVD
          with a popularity fallback
        - **Experimental Gender Classification** using age and company to predict the
          recorded gender label in the modeling dataset

        The machine-learning services are exposed through the
        **FastAPI REST API**. The flight model is managed using
        **MLflow Model Registry**.
        """
    )


    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    st.subheader(
        "🟢 Model Status"
    )

    if health_data:

        (
            status_col1,
            status_col2,
            status_col3
        ) = st.columns(3)


        with status_col1:

            st.metric(
                "Status",
                ACTIVE_MODEL_STATUS
            )


        with status_col2:

            st.metric(
                "Model",
                ACTIVE_MODEL_NAME
            )


        with status_col3:

            st.metric(
                "Version",
                ACTIVE_MODEL_VERSION
            )

    else:

        st.error(
            "Unable to retrieve model "
            "status from FastAPI."
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


    st.caption(
        "Performance values shown above correspond "
        "to the final tuned model evaluation."
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
            f"**Model Version:** "
            f"{ACTIVE_MODEL_VERSION}"
        )

        st.write(
            "**Number of Features:** "
            f"{len(feature_columns)}"
        )

        st.write(
            f"**Model Status:** "
            f"{ACTIVE_MODEL_STATUS}"
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

        st.write(
            API_URL
        )

        st.write(
            "**Health Endpoint:**"
        )

        st.write(
            f"{API_URL}/health"
        )


    with api_col2:

        st.write(
            "**Prediction Endpoint:**"
        )

        st.write(
            f"{API_URL}/predict"
        )

        st.write(
            "**API Documentation:**"
        )

        st.write(
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
        f"""
        MLflow is responsible for:

        - Experiment tracking
        - Parameter logging
        - Metric logging
        - Model logging
        - Model versioning
        - Model Registry

        The registered model is:

        **`flight-price-xgboost` — Version {ACTIVE_MODEL_VERSION}**
        """
    )

    st.write(
        "**MLflow Model URI:**"
    )

    st.write(
        f"models:/flight-price-xgboost/"
        f"{ACTIVE_MODEL_VERSION}"
    )


    st.markdown("---")


    # ========================================================
    # FEATURE INFORMATION
    # ========================================================

    st.subheader(
        "🧩 Model Features"
    )

    st.write(
        f"The model uses "
        f"**{len(feature_columns)} features**."
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
# PAGE 4 — ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.header(
        "ℹ️ About This Project"
    )

    st.markdown(
        """
        ## 🌍 Travel Analytics MLOps

        - **Flight Price Prediction** using a tuned XGBoost regression model
        - **Personalized Hotel Recommendations** using an implicit-feedback
          Truncated SVD recommender with a popularity fallback
        - **Experimental Gender Classification** using a tuned Random Forest
          pipeline based on age and company

        The gender classifier is included as a classification-workflow
        demonstration. Its performance is close to chance level, so its
        predictions should not be interpreted as reliable gender inference.

        The project demonstrates the journey from data preparation and
        exploratory analysis through model development, experiment tracking,
        REST API integration, containerization, workflow orchestration,
        CI/CD configuration, and interactive application deployment.
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
            - Apache Airflow
            - Kubernetes manifests
            - Jenkins pipeline configuration
            - GitHub Actions
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

        **Machine Learning Services**

        **1. Flight Price Prediction**  
        MLflow Model Registry → XGBoost Model → Predicted Flight Price

        **2. Hotel Recommendation**  
        Truncated SVD → Personalized Hotel Recommendations  
        Popularity Model → Cold-start / fallback recommendations

        **3. Gender Classification**  
        Scikit-learn Pipeline → Tuned Random Forest → Predicted Recorded Label

        Docker Compose integrates the **MLflow**, **FastAPI**, and
        **Streamlit** services for local containerized deployment.
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
        f"""
        MLflow is used for:

        - Experiment tracking
        - Model training runs
        - Parameter logging
        - Metric logging
        - Model artifact management
        - Model registration
        - Model versioning

        The registered model used by the REST API is:

        **`flight-price-xgboost` — Version {ACTIVE_MODEL_VERSION}**
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
        Streamlit frontend and the machine learning services.

        This architecture separates:

        **Frontend → API → ML Services**

        FastAPI exposes:

        - `/predict` for flight-price prediction
        - `/recommendations/{user_id}` for hotel recommendations
        - `/gender/predict` for the experimental gender-classification workflow
        - `/health` for application and model-service health monitoring

        This keeps the Streamlit user interface separate from the
        underlying model and recommendation logic.
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
        "Travel Analytics | "
        "Machine Learning & MLOps Project"
    )