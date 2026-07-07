# Loan Risk Prediction — MLOps Pipeline

A production-grade MLOps pipeline for predicting loan default risk using the Home Credit Default Risk dataset.

## Architecture
Dataset → Preprocess → Train (LightGBM) → Evaluate → FastAPI → Docker → CI/CD → Monitor

## Tech Stack

| Component | Tool |
|-----------|------|
| Version Control | Git + GitHub |
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| ML Model | LightGBM |
| API | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |
| Drift Detection | Evidently AI |
| Security | Trivy |

## Quick Start

**1. Clone and install**
```bash
git clone https://github.com/impanarn/loan-risk-mlops.git
cd loan-risk-mlops
pip install -r requirements.txt
```

**2. Run the ML pipeline**
```bash
dvc repro
```

**3. Start all services**
```bash
docker-compose up -d
```

**4. Test the API**
- Health: http://localhost:8000/health
- Predict: http://localhost:8000/predict
- Metrics: http://localhost:8000/metrics
- Swagger: http://localhost:8000/docs

## Model Performance

| Metric | Value |
|--------|-------|
| AUC | 0.7557 |
| Accuracy | 0.7083 |
| F1 Score | 0.2687 |
| Precision | 0.1685 |
| Recall | 0.6638 |

High recall (0.66) prioritized to minimize missed loan defaults.

## CI/CD Pipeline

GitHub Actions runs on every push:
1. Lint with flake8
2. Unit tests with pytest
3. Docker image build
4. Push to DockerHub (main branch)

## Monitoring

- Prometheus scrapes /metrics every 15 seconds
- Grafana dashboard at http://localhost:3000
- Evidently AI drift report in reports/drift_report.html

## Security

- Trivy scan: 58 total (CRITICAL: 5, HIGH: 34 — all in base OS packages)
- No application-level vulnerabilities
- Secrets via GitHub Secrets, .env excluded from git

## Dataset

Home Credit Default Risk (Kaggle) — application_train.csv
- 307,511 applicants, 122 features
- Binary target: default (1) or repaid (0)
- Class imbalance handled via scale_pos_weight=11
