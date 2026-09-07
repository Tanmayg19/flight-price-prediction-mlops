# Flight Price Prediction - End-to-End MLOps Project

An end-to-end Machine Learning and MLOps project for predicting flight prices using an optimized **XGBoost regression model**.

The project demonstrates a production-oriented machine learning lifecycle including data validation, exploratory data analysis, feature engineering, model training, experiment tracking, model registry, automated testing, REST API development, an interactive frontend, containerization, workflow orchestration, CI/CD configuration, and Kubernetes deployment definitions.

---

## Project Overview

Flight prices are influenced by factors such as travel time, distance, flight type, airline agency, origin, and destination.

This project builds a machine learning system to estimate flight prices and packages the solution into a modular MLOps architecture.

Users interact with a **Streamlit web application**, which sends prediction requests to a **FastAPI REST API**. The API uses the trained XGBoost model managed through **MLflow Model Registry**.

The application is containerized using **Docker**, while **Apache Airflow** orchestrates the automated data and model pipeline.

The repository also includes:

- GitHub Actions automated CI
- Jenkins declarative CI/CD pipeline configuration
- Kubernetes deployment and service manifests

---

## Project Objectives

The main objectives are:

- Explore and understand travel and flight data
- Perform exploratory data analysis
- Engineer reproducible machine learning features
- Train and optimize an XGBoost regression model
- Track experiments with MLflow
- Register and version models using MLflow Model Registry
- Build a REST prediction API using FastAPI
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
                           | REST Request
                           v
                      FastAPI API
                       Port 8000
                           |
                           v
                    MLflow Registry
                       Port 5000
                           |
                           v
                  XGBoost Regressor
                           |
                           v
                  Predicted Flight Price
```

The supporting MLOps workflow is:

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
Apache Airflow
GitHub Actions
Jenkins Pipeline Definition
Kubernetes Manifests
```

---

## Machine Learning Model

The prediction model is an **XGBoost Regressor**.

| Property | Value |
|---|---|
| Problem Type | Regression |
| Target | Flight Price |
| Algorithm | XGBoost Regressor |
| Registered Model | `flight-price-xgboost` |
| Registered Version | 1 |

### Original Tuned Model Performance

The original tuned model achieved:

| Metric | Value |
|---|---:|
| R2 Score | 0.9784 |
| RMSE | 62.30 |
| MAE | 50.58 |

An R2 score of approximately 0.9784 indicates that the model explains a high proportion of variation in flight prices on its original evaluation setup.

---

## Automated Candidate Model Evaluation

The Airflow training pipeline generates a candidate model and evaluates it against the existing production model using the **same holdout dataset**.

Evaluation configuration:

```text
Test size:      20%
Random state:   42
Evaluation rows: 54,378
```

### Production vs Candidate

| Metric | Production | Candidate |
|---|---:|---:|
| R2 | 0.991312 | 0.994826 |
| RMSE | 33.835476 | 26.111473 |
| MAE | 24.320074 | 20.872142 |

Improvement:

```text
R2 difference:    +0.003514
RMSE reduction:    7.724003
MAE reduction:     3.447933
```

### Performance Gate

The candidate is accepted only when all three conditions are satisfied:

```text
Candidate R2   > Production R2
Candidate RMSE < Production RMSE
Candidate MAE  < Production MAE
```

Result of the verified pipeline run:

```text
Decision: PROMOTE
```

The evaluation script raises an error when a candidate fails the gate, preventing downstream pipeline completion.

> The original tuned-model metrics and the Airflow production/candidate comparison use different evaluation setups and therefore should not be compared directly.

---

## Model Features

The model uses **26 engineered features**.

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

MLflow is used for machine learning lifecycle management.

Implemented capabilities include:

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

Registered version:

```text
Version 1
```

The FastAPI application uses the registered model for prediction.

MLflow model registration has been implemented separately from the Airflow DAG. Automated Airflow-to-MLflow model registration is not part of the current Airflow pipeline.

---

## Apache Airflow Pipeline

Apache Airflow orchestrates the automated data and machine learning workflow.

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

---

## FastAPI REST API

FastAPI provides the REST prediction layer.

Available endpoints:

```text
GET  /
GET  /health
GET  /features
POST /predict
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
  "model_version": "1"
}
```

The `/predict` endpoint accepts flight information and returns a predicted flight price.

Example:

```json
{
  "predicted_price": 1234.56
}
```

---

## Streamlit Application

The Streamlit frontend provides an interactive prediction interface.

Users can provide:

- Travel date
- Flight type
- Airline agency
- Origin
- Destination
- Distance
- Flight time

The application:

```text
User Input
    |
    v
26-Feature Model Structure
    |
    v
FastAPI Request
    |
    v
Model Prediction
    |
    v
Estimated Flight Price
```

The application also includes:

- Prediction page
- Model Insights
- Model performance information
- Model metadata
- Project information

---

## Docker and Docker Compose

The application is containerized using Docker.

Docker images are provided for:

- FastAPI
- Streamlit

MLflow runs as a separate service through Docker Compose.

```text
Docker Compose
     |
     +---- MLflow       :5000
     |
     +---- FastAPI      :8000
     |
     +---- Streamlit    :8501
```

Start the services:

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

## Automated Testing and GitHub Actions

The project includes automated FastAPI tests covering:

- Root endpoint
- Health endpoint
- Feature endpoint
- Prediction endpoint
- Invalid prediction request

The GitHub Actions CI workflow has been successfully executed.

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
Register Model
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

This verifies both the application API and its containerized execution.

---

## Jenkins CI/CD

A declarative `Jenkinsfile` is included to demonstrate a Jenkins-based CI/CD workflow.

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

> The Jenkinsfile has been configured as part of the project, but a live Jenkins execution is not claimed.

---

## Kubernetes

Kubernetes manifests are provided for the FastAPI and Streamlit application layers.

Files:

```text
k8s/
├── api-deployment.yaml
├── api-service.yaml
├── streamlit-deployment.yaml
└── streamlit-service.yaml
```

The manifests define:

- FastAPI Deployment
- FastAPI ClusterIP Service
- Streamlit Deployment
- Streamlit NodePort Service
- Environment-based service communication
- FastAPI readiness probe
- FastAPI liveness probe

Kubernetes service communication:

```text
Streamlit Pod
     |
     v
flight-api-service:8000
     |
     v
FastAPI Pod
```

The four YAML manifests were successfully validated for YAML syntax.

> A live Kubernetes cluster deployment is not claimed. The manifests are provided as deployment definitions and were validated locally.

The current MLflow URI in the Kubernetes API manifest is intended for a local Docker Desktop development environment and should be changed for a production Kubernetes deployment.

---

## Project Structure

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
│   └── app.py
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

Generated Airflow logs, environment configuration, processed datasets, MLflow runtime files, virtual environments, and the generated candidate-model binary are excluded from version control.

---

## Technology Stack

### Data and Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
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
- Kubernetes

### CI/CD and Testing

- GitHub Actions
- Jenkins Pipeline
- Pytest

### Development

- Jupyter Notebook
- Git
- GitHub

---

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Tanmayg19/flight-price-prediction-mlops.git
cd flight-price-prediction-mlops
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
docker compose up -d
```

### 5. Check services

```bash
docker compose ps
```

### 6. Open the application

```text
Streamlit:       http://localhost:8501
FastAPI Docs:    http://localhost:8000/docs
FastAPI Health:  http://localhost:8000/health
MLflow:          http://localhost:5000
```

---

## Configuration

Docker Compose uses environment variables for service communication.

FastAPI to MLflow:

```text
MLFLOW_TRACKING_URI=http://mlflow:5000
```

Streamlit to FastAPI:

```text
API_URL=http://api:8000
```

This allows containers to communicate using Docker Compose service names.

---

## Model Artifacts

Version-controlled model metadata includes:

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

The Airflow pipeline generates:

```text
xgb_flight_price_candidate.pkl
```

The candidate binary is excluded from Git because it is reproducible through the training pipeline.

---

## Implementation Status

| Component | Status |
|---|---|
| Exploratory Data Analysis | Implemented |
| Feature Engineering | Implemented |
| XGBoost Training | Implemented |
| Hyperparameter Tuning | Implemented |
| MLflow Tracking | Implemented |
| MLflow Model Registry | Implemented |
| FastAPI REST API | Implemented and tested |
| Streamlit Frontend | Implemented and tested |
| Docker | Implemented and tested |
| Docker Compose | Implemented and tested |
| API Automated Tests | Implemented and passing |
| GitHub Actions CI | Implemented and successfully executed |
| Apache Airflow | Implemented and successfully executed |
| Model Performance Gate | Implemented and successfully executed |
| Kubernetes | Manifests created and syntax validated |
| Jenkins | Pipeline definition configured |

---

## Key Project Highlights

- End-to-end regression ML workflow
- Optimized XGBoost model
- 26-feature reproducible feature pipeline
- MLflow experiment tracking and Model Registry
- FastAPI REST prediction service
- Interactive Streamlit frontend
- Dockerized multi-service application
- Automated API testing
- Successful GitHub Actions CI
- Successful 10-task Apache Airflow ML pipeline
- Automated candidate-vs-production performance gate
- Kubernetes deployment definitions
- Jenkins declarative pipeline configuration
- Reproducible project structure

---

## Limitations and Future Improvements

Potential future improvements include:

- Automate Airflow-to-MLflow model registration after the performance gate
- Deploy the Kubernetes manifests to a persistent Kubernetes cluster
- Execute and validate the Jenkins pipeline on a dedicated Jenkins server
- Add model monitoring and prediction logging
- Add data and model drift detection
- Add formal data/model versioning
- Deploy the complete stack to a cloud environment
- Use native XGBoost model serialization and explicit version pinning for stronger cross-version compatibility

---

## Author

**Tanmay Gautam**

Data Analytics / Machine Learning / MLOps Portfolio Project
