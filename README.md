# ✈️ Flight Price Prediction — End-to-End MLOps Project

An end-to-end Machine Learning and MLOps project that predicts flight prices using an optimized **XGBoost regression model** and demonstrates the complete ML lifecycle — from data understanding and feature engineering to experiment tracking, model registry, REST API deployment, containerization, and interactive prediction.

---

## 📌 Project Overview

Flight pricing is influenced by several factors such as travel time, distance, flight type, airline agency, origin, and destination.

This project develops a machine learning solution to estimate flight prices and packages the model into a production-style MLOps architecture.

The application provides an interactive **Streamlit web interface** where users can enter flight details and receive a predicted flight price.

The prediction request is handled through a **FastAPI REST API**, while the trained XGBoost model is managed and served through **MLflow Model Registry**.

The complete application is containerized using **Docker and Docker Compose**.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Understand and explore the flight pricing dataset
- Perform exploratory data analysis
- Engineer meaningful machine learning features
- Train and optimize an XGBoost regression model
- Track experiments using MLflow
- Register and version the trained model using MLflow Model Registry
- Build a REST API using FastAPI
- Connect the REST API with a Streamlit frontend
- Containerize the complete application using Docker
- Orchestrate MLflow, FastAPI, and Streamlit using Docker Compose
- Create a reproducible and portfolio-ready MLOps workflow

---

# 🏗️ Project Architecture

```text
                         ┌──────────────────────┐
                         │      User            │
                         │  Flight Information  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Streamlit       │
                         │   Web Application     │
                         │      Port 8501       │
                         └──────────┬───────────┘
                                    │
                              REST API Request
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │     REST API         │
                         │      Port 8000       │
                         └──────────┬───────────┘
                                    │
                              Model Request
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │        MLflow        │
                         │  Model Registry      │
                         │      Port 5000       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   XGBoost Regressor  │
                         │  Registered Model    │
                         │      Version 1       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Predicted Flight     │
                         │       Price          │
                         └──────────────────────┘

🔄 End-to-End ML Workflow

Data
  │
  ▼
Data Understanding
  │
  ▼
Exploratory Data Analysis
  │
  ▼
Feature Engineering
  │
  ▼
Model Training
  │
  ▼
Hyperparameter Tuning
  │
  ▼
XGBoost Model
  │
  ▼
MLflow Experiment Tracking
  │
  ▼
MLflow Model Registry
  │
  ▼
FastAPI REST API
  │
  ▼
Streamlit Application
  │
  ▼
Docker / Docker Compose

🤖 Machine Learning Model

The project uses an XGBoost Regressor for flight price prediction.

MODEL

Algorithm: XGBoost Regressor
Problem Type: Regression
Target: Flight Price
Registered Model: flight-price-xgboost
Model Version: 1

The final tuned model achieved approximately:

| Metric   |  Value |
| -------- | -----: |
| R² Score | 0.9784 |
| RMSE     |  62.30 |
| MAE      |  50.58 |


** Interpretation :- **

An R² score of approximately 0.9784 indicates that the model explains a very high proportion of the variation in flight prices within the evaluation data.

📊 Model Features

The final model uses 26 features.

Numerical Features :
#time
#distance

Flight Type
#flightType_economic
#flightType_firstClass
#flightType_premium

Airline Agency
#agency_CloudFy
#agency_FlyingDrops
#agency_Rainbow

Origin
from_Aracaju (SE)
from_Brasilia (DF)
from_Campo Grande (MS)
from_Florianopolis (SC)
from_Natal (RN)
from_Recife (PE)
from_Rio de Janeiro (RJ)
from_Salvador (BH)
from_Sao Paulo (SP)

Destination
to_Aracaju (SE)
to_Brasilia (DF)
to_Campo Grande (MS)
to_Florianopolis (SC)
to_Natal (RN)
to_Recife (PE)
to_Rio de Janeiro (RJ)
to_Salvador (BH)
to_Sao Paulo (SP)

🔬 MLflow Integration

MLflow is used to manage the machine learning lifecycle.

The project uses MLflow for:

Experiment tracking
Parameter logging
Metric logging
Model logging
Model versioning
Model Registry

The registered model is:
flight-price-xgboost

Current model version:
Version 1

The model is loaded by the FastAPI service using the MLflow Model Registry.

🌐 FastAPI REST API

A FastAPI REST API provides the prediction service between the frontend and machine learning model.

API Base URL
http://localhost:8000

Health Check
GET /health

Example response:
{
  "status": "healthy",
  "model_status": "Ready",
  "model_name": "flight-price-xgboost",
  "model_version": "1"
}

Prediction Endpoint
POST /predict

The API accepts the model features and returns the predicted flight price
{
  "predicted_price": 1234.56
}

Feature Endpoint
GET /features

Returns the feature structure expected by the registered model.

🖥️ Streamlit Application

The Streamlit application provides an interactive interface for users.

Users can enter:
Travel date
Flight type
Airline agency
Origin
Destination
Distance
Flight time

The application then:

Creates the required model feature structure
Converts the input into the expected feature format
Sends the request to FastAPI
Receives the prediction
Displays the estimated flight price

The application also includes:

Model Insights
Model performance metrics
Model information
MLflow information
Technical details
Project information

🐳 Dockerization

The application is containerized using Docker.

The project uses separate Docker images for:

FastAPI
Streamlit

MLflow runs as a separate service through Docker Compose.

🧩 Docker Compose

Docker Compose orchestrates the three services:

┌─────────────────────────────┐
│        Docker Compose       │
│                             │
│  ┌──────────┐               │
│  │ MLflow   │ Port 5000     │
│  └────┬─────┘               │
│       │                     │
│  ┌────▼─────┐               │
│  │ FastAPI  │ Port 8000     │
│  └────┬─────┘               │
│       │                     │
│  ┌────▼──────┐              │
│  │ Streamlit │ Port 8501    │
│  └───────────┘              │
│                             │
└─────────────────────────────┘

Start the complete application

From the project root:
docker compose up -d

Check running containers
docker compose ps

View logs
docker compose logs -f

Stop the application
docker compose down

🌍 Application URLs

Once Docker Compose is running:
| Service              | URL                                                          |
| -------------------- | ------------------------------------------------------------ |
| Streamlit            | [http://localhost:8501](http://localhost:8501)               |
| FastAPI              | [http://localhost:8000](http://localhost:8000)               |
| FastAPI Swagger Docs | [http://localhost:8000/docs](http://localhost:8000/docs)     |
| FastAPI Health Check | [http://localhost:8000/health](http://localhost:8000/health) |
| MLflow               | [http://localhost:5000](http://localhost:5000)               |


📁 Project Structure

Travel-Analytics-MLOps/
│
├── api/
│   └── app.py
│
├── data/
│   └── streamlit_app/
│       ├── app.py
│       └── Dockerfile
│
├── models/
│   ├── best_hyperparameters.json
│   ├── feature_columns.pkl
│   ├── final_model_metrics.json
│   ├── model_metadata.json
│   └── xgb_flight_price_model.pkl
│
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   └── MLFlow.ipynb
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
├── .gitignore
└── README.md

🛠️ Technology Stack

Data & Machine Learning
Python
Pandas
NumPy
Scikit-learn
XGBoost
Joblib
Matplotlib
Seaborn

MLOps
MLflow
MLflow Model Registry
FastAPI
REST API

Application
Streamlit

Deployment & Infrastructure
Docker
Docker Compose

Development
Jupyter Notebook
Git
GitHub

🚀 Running the Project Locally
1. Clone the repository
git clone https://github.com/<your-username>/flight-price-prediction-mlops.git

Navigate into the project:
cd flight-price-prediction-mlops

2. Create a virtual environment
Windows:
python -m venv venv

Activate:
.\venv\Scripts\Activate.ps1

3. Install dependencies
pip install -r requirements.txt

4. Start using Docker Compose
The recommended approach is:
docker compose up -d

Check the services:
docker compose ps

5. Open the application

Streamlit: http://localhost:8501
FastAPI Swagger documentation:  http://localhost:8000/docs
MLflow:  http://localhost:5000

🧪 API Testing

FastAPI provides interactive Swagger documentation.
Open:  http://localhost:8000/docs

Available endpoints include:
GET  /
GET  /health
GET  /features
POST /predict
The /predict endpoint can be tested directly through Swagger UI.

🔐 Configuration

The application uses environment variables for service communication inside Docker Compose.

FastAPI → MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

Streamlit → FastAPI
API_URL=http://api:8000

This allows the containers to communicate using their Docker Compose service names.

📦 Model Artifacts

The repository contains the model-related metadata required by the application:
models/
├── best_hyperparameters.json
├── feature_columns.pkl
├── final_model_metrics.json
├── model_metadata.json
└── xgb_flight_price_model.pkl

Large raw and processed datasets, local MLflow databases, MLflow runs, and virtual environments are excluded through .gitignore.


🔄 MLOps Architecture

This project demonstrates the following MLOps concepts:
                    DATA
                      │
                      ▼
              DATA PREPARATION
                      │
                      ▼
           FEATURE ENGINEERING
                      │
                      ▼
             MODEL TRAINING
                      │
                      ▼
          HYPERPARAMETER TUNING
                      │
                      ▼
             MLFLOW TRACKING
                      │
                      ▼
            MODEL REGISTRY
                      │
                      ▼
               FASTAPI
              REST SERVICE
                      │
                      ▼
               STREAMLIT
             USER INTERFACE
                      │
                      ▼
                  DOCKER
                      │
                      ▼
             DOCKER COMPOSE


🎯 Key Project Highlights
End-to-end machine learning workflow
XGBoost regression model
Hyperparameter tuning
High-performing model with approximately 0.9784 R²
MLflow experiment tracking
MLflow Model Registry
Versioned model deployment
FastAPI REST API
Interactive Streamlit application
Dockerized ML application
Multi-container orchestration using Docker Compose
Reproducible project structure
Production-oriented MLOps architecture

📈 Future Improvements

Potential future improvements include:

Automated CI/CD pipeline using Jenkins or GitHub Actions
Automated model retraining
Data and model versioning
Model monitoring
Data drift detection
Prediction logging
Automated testing
Cloud deployment
Kubernetes deployment
Apache Airflow pipeline orchestration
Model explainability using SHAP

👨‍💻 Author

Tanmay Gautam

Data Analytics / Machine Learning / MLOps Project

⭐ If you found this project useful

Feel free to explore the repository, review the notebooks, experiment with the API, and try the interactive prediction application.


### One important correction before you put this on GitHub

I deliberately wrote the README around the **components you have actually implemented**:

**XGBoost → MLflow → FastAPI REST API → Streamlit → Docker → Docker Compose**

I listed **Jenkins, Kubernetes, and Airflow only under "Future Improvements"**, rather than claiming that they're already implemented. That's important for a portfolio project—you don't want the README to claim functionality that isn't actually in the repository.

Also, I left the GitHub URL as:

```text
https://github.com/tanmayg19/flight-price-prediction-mlops.git

