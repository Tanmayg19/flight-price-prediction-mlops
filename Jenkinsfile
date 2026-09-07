pipeline {
    agent any

    environment {
        API_IMAGE = "travel-analytics-mlops-api"
        STREAMLIT_IMAGE = "travel-analytics-mlops-streamlit"
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                bat '''
                python -m pip install --upgrade pip
                pip install -r requirements-docker.txt
                '''
            }
        }

        stage('Run Tests') {
            steps {
                bat '''
                pytest tests/test_api.py -v
                '''
            }
        }

        stage('Build API Docker Image') {
            steps {
                bat '''
                docker build -t %API_IMAGE%:latest -f Dockerfile .
                '''
            }
        }

        stage('Build Streamlit Docker Image') {
            steps {
                bat '''
                docker build -t %STREAMLIT_IMAGE%:latest -f data/streamlit_app/Dockerfile .
                '''
            }
        }

        stage('Validate Kubernetes Manifests') {
            steps {
                bat '''
                python -c "import yaml, pathlib; files=list(pathlib.Path('k8s').glob('*.yaml')); [yaml.safe_load(f.read_text()) for f in files]; print('Kubernetes YAML syntax OK:', len(files), 'files')"
                '''
            }
        }
    }

    post {
        success {
            echo 'Jenkins pipeline completed successfully.'
        }

        failure {
            echo 'Jenkins pipeline failed. Check the stage logs.'
        }
    }
}