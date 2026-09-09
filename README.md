# Travel Analytics - End-to-End Machine Learning & MLOps Project

An end-to-end Machine Learning and MLOps portfolio project for **flight price prediction** and **personalized hotel recommendation**.

The project demonstrates a production-oriented machine learning lifecycle including data validation, exploratory data analysis, feature engineering, model training, recommendation modeling, experiment tracking, model registry, automated testing, REST API development, an interactive frontend, containerization, workflow orchestration, CI/CD configuration, and Kubernetes deployment definitions.

---

## Project Overview

This project contains two integrated machine learning use cases:

1. **Flight Price Prediction**
   - Predicts flight prices using an optimized **XGBoost regression model**
   - Uses travel time, distance, flight type, airline agency, origin, and destination
   - Served through FastAPI and managed with MLflow Model Registry

2. **Hotel Recommendation**
   - Recommends hotels using an **implicit-feedback TruncatedSVD recommender**
   - Uses historical user-hotel booking frequency transformed with `log1p(booking_count)`
   - Uses a **popularity-based fallback** for unknown users or users without usable personalized candidates

Users interact with a **Streamlit web application**, which communicates with a **FastAPI REST API**.

The project is containerized using **Docker Compose**, while **Apache Airflow** orchestrates the automated flight-model data and training workflow.

The repository also includes:

- GitHub Actions automated CI
- Jenkins declarative CI/CD pipeline configuration
- Kubernetes deployment and service manifests

---

## Project Objectives

The main objectives are:

- Explore and understand user, flight, and hotel travel data
- Perform exploratory data analysis
- Engineer reproducible machine learning features
- Train and optimize an XGBoost regression model
- Build an implicit-feedback hotel recommendation model
- Track experiments with MLflow
- Register and version the flight model using MLflow Model Registry
- Build REST endpoints using FastAPI
- Build an interactive frontend using Streamlit
- Containerize services using Docker
- Orchestrate services using Docker Compose
- Automate data validation, feature engineering, training, and evaluation using Apache Airflow
- Implement automated API and Docker testing with GitHub Actions
- Define a Jenkins CI/CD pipeline
- Provide Kubernetes deployment manifests
- Create a reproducible portfolio-ready MLOps architecture

---

## Architecture

```text
                              User
                                |
                                v
                       Streamlit Frontend
                          Port 8501
                                |
                                | REST Requests
                                v
                           FastAPI API
                          Port 8000
                    _________|__________
                   |                    |
                   v                    v
          Flight Price Service    Hotel Recommender
                   |                    |
                   v                    v
            MLflow Registry      TruncatedSVD Model
             Port 5000           + Popularity Fallback
                   |
                   v
           XGBoost Regressor
```

The supporting flight-model MLOps workflow is:

```text
Raw Data
   |
   v
Data Validation
   |
   v
Feature Engineering
   |
   v
Model Training
   |
   v
Candidate Model Evaluation
   |
   v
Performance Gate
   |
   v
MLflow / Model Management
   |
   v
FastAPI
   |
   v
Streamlit
   |
   v
Docker / Docker Compose

Supporting Automation:
- Apache Airflow
- GitHub Actions
- Jenkins Pipeline Definition
- Kubernetes Manifests
```

---

# 1. Flight Price Prediction Model

## Machine Learning Model

The flight prediction model is an **XGBoost Regressor**.

| Property | Value |
|---|---|
| Problem Type | Regression |
| Target | Flight Price |
| Algorithm | XGBoost Regressor |
| Registered Model | `flight-price-xgboost` |
| Active Local Registered Version | 4 |

### Original Tuned Model Performance

| Metric | Value |
|---|---:|
| R² Score | 0.9784 |
| RMSE | 62.30 |
| MAE | 50.58 |

An R² score of approximately 0.9784 indicates that the model explains a high proportion of variation in flight prices on its original evaluation setup.

> The original tuned-model metrics and the later Airflow production/candidate comparison use different evaluation setups and should not be compared directly.

---

## Automated Candidate Model Evaluation

The Airflow training pipeline generates a candidate model and evaluates it against the existing production model using the **same holdout dataset**.

Evaluation configuration:

```text
Test size:       20%
Random state:    42
Evaluation rows: 54,378
```

### Production vs Candidate

| Metric | Production | Candidate |
|---|---:|---:|
| R² | 0.991312 | 0.994826 |
| RMSE | 33.835476 | 26.111473 |
| MAE | 24.320074 | 20.872142 |

Improvement:

```text
R² difference:   +0.003514
RMSE reduction:   7.724003
MAE reduction:    3.447933
```

### Performance Gate

The candidate is accepted only when all three conditions are satisfied:

```text
Candidate R²   > Production R²
Candidate RMSE < Production RMSE
Candidate MAE  < Production MAE
```

Verified result:

```text
Decision: PROMOTE
```

The evaluation script raises an error when a candidate fails the gate, preventing downstream pipeline completion.

---

## Model Features

The flight model uses **26 engineered features**.

### Numerical

```text
time
distance
```

### Flight Type

```text
flightType_economic
flightType_firstClass
flightType_premium
```

### Airline Agency

```text
agency_CloudFy
agency_FlyingDrops
agency_Rainbow
```

### Origin

```text
from_Aracaju (SE)
from_Brasilia (DF)
from_Campo Grande (MS)
from_Florianopolis (SC)
from_Natal (RN)
from_Recife (PE)
from_Rio de Janeiro (RJ)
from_Salvador (BH)
from_Sao Paulo (SP)
```

### Destination

```text
to_Aracaju (SE)
to_Brasilia (DF)
to_Campo Grande (MS)
to_Florianopolis (SC)
to_Natal (RN)
to_Recife (PE)
to_Rio de Janeiro (RJ)
to_Salvador (BH)
to_Sao Paulo (SP)
```

---

## MLflow Integration

MLflow is used for:

- Experiment tracking
- Parameter logging
- Metric logging
- Model logging
- Model versioning
- Model Registry

Registered model:

```text
flight-price-xgboost
```

Active local registered version:

```text
Version 4
```

The FastAPI application loads the registered model and serves predictions through the REST API.

MLflow model registration is implemented separately from the Airflow DAG. Automated Airflow-to-MLflow registration is **not** part of the current Airflow pipeline.

---

# 2. Hotel Recommendation System

## Recommendation Approach

The hotel recommendation model uses **implicit user feedback** derived from booking history.

For each observed user-hotel pair:

```text
interaction_strength = log1p(booking_count)
```

The selected personalized model is:

```text
TruncatedSVD
n_components = 2
```

A popularity ranking is retained as a fallback strategy.

### Serving Strategy

```text
Known user with unseen hotels
        |
        v
Personalized TruncatedSVD ranking

Unknown user or no usable personalized candidates
        |
        v
Popularity Fallback
```

This is a serving fallback strategy, not a separately trained hybrid recommender.

---

## Recommendation Evaluation

Evaluation used **leave-one-out** methodology for users with at least two unique hotels.

- Total hotel users: 1,310
- Eligible evaluation users: 1,285
- Validation users: 899
- Final test users: 386
- Training interactions after holdout: 8,439
- Hotels: 9
- Primary metric: **Hit Rate@1**

### Final Test Results

| Model | Hit Rate@1 | Hit Rate@2 | Hit Rate@3 |
|---|---:|---:|---:|
| Popularity Baseline | 0.7280 | 0.8938 | 0.9482 |
| Personalized SVD | 0.7383 | 0.8834 | 0.9482 |

The tuned SVD model was selected because it performs slightly better on the strict primary metric **Hit Rate@1** and provides user-specific rankings.

The improvement is modest, which is expected because the dataset contains only **9 hotels** and most users have already interacted with a large part of the catalog.

---

## Recommendation Data Limitation

Each destination maps to only one hotel in this dataset.

Therefore, the application does **not** claim to rank multiple hotels within a selected destination.

Instead, recommendations are generated across the available hotel catalog based on user booking history.

Known users may receive fewer recommendations than requested because previously seen hotels are excluded.

---

## Recommendation Artifacts

```text
models/recommendation/
├── hotel_metadata.pkl
├── popularity_ranking.pkl
├── recommendation_metadata.json
├── svd_model.pkl
├── svd_scores.pkl
└── train_interactions.pkl
```

The recommendation model is served through:

```text
api/recommendation_service.py
```

---

# 3. Apache Airflow Pipeline

Apache Airflow orchestrates the automated data and flight-model workflow.

DAG:

```text
travel_data_ingestion_validation
```

The verified pipeline contains **10 tasks**:

```text
1. check_source_files
2. validate_users_data
3. validate_flights_data
4. validate_hotels_data
5. cross_table_validation
6. save_validated_data
7. feature_engineering
8. model_training
9. model_evaluation
10. pipeline_complete
```

Pipeline flow:

```text
                    check_source_files
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
   validate_users   validate_flights  validate_hotels
          \                |                /
           \               |               /
            +------ cross_table_validation
                           |
                           v
                  save_validated_data
                           |
                           v
                  feature_engineering
                           |
                           v
                    model_training
                           |
                           v
                   model_evaluation
                           |
                           v
                   pipeline_complete
```

The full 10-task DAG was successfully executed and verified with all tasks completing successfully.

`max_active_runs=1` is used to prevent overlapping resource-intensive pipeline runs.

> Automatic Airflow-to-MLflow model registration is intentionally not included in the verified pipeline.

---

# 4. FastAPI REST API

FastAPI provides the REST service layer for both ML use cases.

Available endpoints:

```text
GET  /
GET  /health
GET  /features
POST /predict
GET  /recommendations/{user_id}?top_n=3
```

Local API:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

Example health response:

```json
{
  "status": "healthy",
  "model_status": "Ready",
  "model_name": "flight-price-xgboost",
  "model_version": "4",
  "recommendation_service": "Ready"
}
```

### Flight Prediction

`POST /predict`

Example response:

```json
{
  "predicted_price": 1234.56
}
```

### Hotel Recommendations

`GET /recommendations/1104?top_n=3`

Known users receive personalized SVD recommendations when unseen candidate hotels are available.

Unknown users receive popularity-based fallback recommendations.

The API validates `top_n` within the supported range.

---

# 5. Streamlit Application

The Streamlit frontend provides an interactive interface for both ML systems.

Pages:

```text
🔮 Prediction
🏨 Hotel Recommendations
📊 Model Insights
ℹ️ About
```

### Flight Prediction Inputs

- Travel date
- Flight type
- Airline agency
- Origin
- Destination
- Distance
- Flight time

The application converts the inputs into the expected 26-feature model structure and sends them to FastAPI.

### Hotel Recommendation Inputs

Users provide:

- `userCode`
- Number of requested recommendations

The page displays:

- Hotel
- Location
- Average hotel price
- Average stay duration
- Booking metadata
- Recommendation type
- Recommendation score when applicable

The frontend distinguishes between:

```text
Personalized SVD
Popularity Fallback
```

---

# 6. Docker and Docker Compose

The application is containerized with Docker.

Docker Compose runs:

```text
Docker Compose
     |
     +---- MLflow       :5000
     |
     +---- FastAPI      :8000
     |
     +---- Streamlit    :8501
```

The rebuilt Dockerized application has been verified with both:

- Flight price prediction
- Hotel recommendations

Start services:

```bash
docker compose up -d
```

Check services:

```bash
docker compose ps
```

Stop services:

```bash
docker compose down
```

### Service URLs

| Service | Local URL |
|---|---|
| Streamlit | `http://localhost:8501` |
| FastAPI | `http://localhost:8000` |
| Swagger Docs | `http://localhost:8000/docs` |
| Health Check | `http://localhost:8000/health` |
| MLflow | `http://localhost:5000` |

---

# 7. Automated Testing and GitHub Actions

The FastAPI automated suite contains **10 tests**.

Coverage includes:

1. Root endpoint
2. Health endpoint
3. Feature endpoint
4. Flight prediction endpoint
5. Invalid prediction request
6. Personalized hotel recommendation
7. Cold-start popularity recommendation
8. Recommendation `top_n` behavior
9. Invalid `top_n=0`
10. Invalid `top_n=10`

The test suite has been verified:

```text
10 passed
```

It has also passed against the Dockerized API.

GitHub Actions CI has been successfully executed on the latest recommendation-integrated project state.

The workflow performs:

```text
Checkout
   |
   v
Install Dependencies
   |
   v
Start MLflow
   |
   v
Register CI Model
   |
   v
Start FastAPI
   |
   v
Run API Tests
   |
   v
Build Docker API Image
   |
   v
Run Dockerized API
   |
   v
Test Dockerized API
```

---

# 8. Jenkins CI/CD

A declarative `Jenkinsfile` is included.

Configured stages include:

```text
Checkout
   |
   v
Install Dependencies
   |
   v
Run Tests
   |
   v
Build FastAPI Docker Image
   |
   v
Build Streamlit Docker Image
   |
   v
Validate Kubernetes Manifests
```

The Jenkins pipeline definition is included in the repository.

> Jenkins itself was not installed or executed in the local development environment. A successful live Jenkins pipeline run is therefore not claimed.

---

# 9. Kubernetes

Kubernetes manifests are provided for the FastAPI and Streamlit layers.

```text
k8s/
├── api-deployment.yaml
├── api-service.yaml
├── streamlit-deployment.yaml
└── streamlit-service.yaml
```

The four manifests were successfully validated for YAML syntax.

They define:

- FastAPI Deployment
- FastAPI ClusterIP Service
- Streamlit Deployment
- Streamlit NodePort Service
- Environment-based service communication
- FastAPI readiness probe
- FastAPI liveness probe

> A live Kubernetes cluster deployment is not claimed. The manifests are deployment definitions and have been syntax validated locally.

The MLflow URI in the current Kubernetes API manifest is intended for a local Docker Desktop-style development setup and should be changed for a production Kubernetes environment.

---

# 10. Project Structure

```text
Travel-Analytics-MLOps/
|
├── .github/
│   └── workflows/
│       └── ci.yml
|
├── airflow/
│   ├── dags/
│   │   └── travel_data_ingestion_dag.py
│   ├── Dockerfile
│   └── docker-compose.yaml
|
├── api/
│   ├── app.py
│   └── recommendation_service.py
|
├── data/
│   └── streamlit_app/
│       ├── app.py
│       └── Dockerfile
|
├── k8s/
│   ├── api-deployment.yaml
│   ├── api-service.yaml
│   ├── streamlit-deployment.yaml
│   └── streamlit-service.yaml
|
├── models/
│   ├── recommendation/
│   │   ├── hotel_metadata.pkl
│   │   ├── popularity_ranking.pkl
│   │   ├── recommendation_metadata.json
│   │   ├── svd_model.pkl
│   │   ├── svd_scores.pkl
│   │   └── train_interactions.pkl
│   ├── best_hyperparameters.json
│   ├── candidate_model_metrics.json
│   ├── feature_columns.pkl
│   ├── final_model_metrics.json
│   ├── model_evaluation_report.json
│   ├── model_metadata.json
│   └── xgb_flight_price_model.pkl
|
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 05_hotel_recommendation.ipynb
│   └── MLFlow.ipynb
|
├── scripts/
│   ├── compare_models.py
│   ├── evaluate_candidate.py
│   ├── feature_engineering.py
│   ├── register_model.py
│   └── train_model.py
|
├── tests/
│   └── test_api.py
|
├── Dockerfile
├── docker-compose.yml
├── Jenkinsfile
├── requirements.txt
├── requirements-docker.txt
├── .gitignore
└── README.md
```

Generated Airflow logs, processed runtime files, virtual environments, generated cache files, and the reproducible candidate-model binary are excluded from version control.

---

# 11. Technology Stack

### Data and Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TruncatedSVD
- Optuna
- Matplotlib

### MLOps

- MLflow
- MLflow Model Registry
- Apache Airflow
- FastAPI
- REST API

### Application

- Streamlit

### Containers and Deployment

- Docker
- Docker Compose
- Kubernetes manifests

### CI/CD and Testing

- GitHub Actions
- Jenkins Pipeline configuration
- Pytest

### Development

- Jupyter Notebook
- Git
- GitHub

---

# 12. Running the Project Locally

## 1. Clone the repository

```bash
git clone https://github.com/Tanmayg19/flight-price-prediction-mlops.git
cd flight-price-prediction-mlops
```

## 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Start the application

```bash
docker compose up -d
```

## 5. Check services

```bash
docker compose ps
```

## 6. Open the application

```text
Streamlit:       http://localhost:8501
FastAPI Docs:    http://localhost:8000/docs
FastAPI Health:  http://localhost:8000/health
MLflow:          http://localhost:5000
```

---

# 13. Configuration

FastAPI to MLflow:

```text
MLFLOW_TRACKING_URI=http://mlflow:5000
```

Streamlit to FastAPI:

```text
API_URL=http://api:8000
```

This allows the Docker Compose services to communicate using service names.

The local FastAPI application defaults to:

```text
MLFLOW_MODEL_NAME=flight-price-xgboost
MLFLOW_MODEL_VERSION=4
```

The GitHub Actions workflow can override the model version for its isolated CI model registration.

---

# 14. Model Artifacts

## Flight Model

```text
models/
├── best_hyperparameters.json
├── candidate_model_metrics.json
├── feature_columns.pkl
├── final_model_metrics.json
├── model_evaluation_report.json
├── model_metadata.json
└── xgb_flight_price_model.pkl
```

The Airflow pipeline also generates:

```text
xgb_flight_price_candidate.pkl
```

The candidate binary is excluded from Git because it is reproducible through the training pipeline.

## Recommendation Model

```text
models/recommendation/
├── hotel_metadata.pkl
├── popularity_ranking.pkl
├── recommendation_metadata.json
├── svd_model.pkl
├── svd_scores.pkl
└── train_interactions.pkl
```

---

# 15. Implementation Status

| Component | Status |
|---|---|
| Exploratory Data Analysis | ✅ Implemented |
| Flight Feature Engineering | ✅ Implemented |
| XGBoost Training | ✅ Implemented |
| Hyperparameter Tuning | ✅ Implemented |
| Flight Candidate Performance Gate | ✅ Implemented and executed |
| MLflow Tracking | ✅ Implemented |
| MLflow Model Registry | ✅ Implemented and verified |
| Hotel Recommendation EDA | ✅ Implemented |
| TruncatedSVD Recommender | ✅ Implemented and evaluated |
| Popularity Fallback | ✅ Implemented |
| Recommendation Production Service | ✅ Implemented and tested |
| FastAPI REST API | ✅ Implemented and tested |
| Streamlit Frontend | ✅ Implemented and tested |
| Docker | ✅ Implemented and tested |
| Docker Compose | ✅ Implemented and tested |
| API Automated Tests | ✅ 10 tests passing |
| GitHub Actions CI | ✅ Successfully executed on latest project state |
| Apache Airflow | ✅ 10-task pipeline successfully executed |
| Kubernetes | 🟡 Manifests created and syntax validated |
| Jenkins | 🟡 Pipeline definition configured; live execution not performed |
| Airflow → MLflow Auto-registration | ⏭ Intentionally not implemented |
| Live Kubernetes Deployment | ⏭ Not performed |
| SHAP Explainability | ⏭ Not implemented |

---

# 16. Key Project Highlights

- End-to-end regression ML workflow
- Personalized hotel recommendation system
- Optimized XGBoost regression model
- Tuned implicit-feedback TruncatedSVD recommender
- Popularity-based cold-start fallback
- 26-feature reproducible flight feature pipeline
- MLflow experiment tracking and Model Registry
- FastAPI serving two machine learning use cases
- Interactive multi-page Streamlit frontend
- Dockerized multi-service application
- 10 automated API tests
- Successful latest-state GitHub Actions CI
- Successful 10-task Apache Airflow ML pipeline
- Automated candidate-vs-production performance gate
- Kubernetes deployment definitions with syntax validation
- Jenkins declarative pipeline configuration
- Reproducible project structure

---

# 17. Limitations and Future Improvements

Potential future improvements include:

- Automate Airflow-to-MLflow model registration after the performance gate
- Deploy the Kubernetes manifests to a persistent Kubernetes cluster
- Execute and validate the Jenkins pipeline on a dedicated Jenkins server
- Add model monitoring and prediction logging
- Add data and model drift detection
- Add formal data/model versioning
- Deploy the complete stack to a cloud environment
- Expand the hotel catalog to support richer recommendation evaluation
- Add more item and user-side recommendation features
- Use native XGBoost model serialization and explicit version pinning for stronger cross-version compatibility

---

## Author

**Tanmay Gautam**

Data Analytics / Machine Learning / MLOps Portfolio Project
