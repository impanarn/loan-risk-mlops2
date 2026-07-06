import pandas as pd
import numpy as np
import yaml
import json
import joblib
import mlflow
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, classification_report
)

def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def evaluate(params):
    out_dir = params["data"]["processed_path"]
    threshold = params["evaluate"]["threshold"]
    metrics_path = params["evaluate"]["metrics_path"]

    X_test = pd.read_csv(f"{out_dir}/X_test.csv")
    y_test = pd.read_csv(f"{out_dir}/y_test.csv").squeeze()

    model = joblib.load("models/model.pkl")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "auc":       round(roc_auc_score(y_test, y_proba), 4),
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
    }

    print("[evaluate] Metrics:", json.dumps(metrics, indent=2))
    print(classification_report(y_test, y_pred))

    import os; os.makedirs("reports", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("loan-risk-prediction")
    with mlflow.start_run(run_name="lgbm-evaluation"):
        mlflow.log_metrics(metrics)
    print(f"[evaluate] Metrics saved to {metrics_path}")

if __name__ == "__main__":
    params = load_params()
    evaluate(params)