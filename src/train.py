import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import joblib
import os
from sklearn.metrics import roc_auc_score

def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def train(params):
    out_dir = params["data"]["processed_path"]

    X_train = pd.read_csv(f"{out_dir}/X_train.csv")
    y_train = pd.read_csv(f"{out_dir}/y_train.csv").squeeze()

    mp = params["model"]

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("loan-risk-prediction")

    with mlflow.start_run(run_name="lgbm-training"):
        mlflow.log_params({
            "n_estimators":      mp["n_estimators"],
            "learning_rate":     mp["learning_rate"],
            "max_depth":         mp["max_depth"],
            "num_leaves":        mp["num_leaves"],
            "min_child_samples": mp["min_child_samples"],
        })

        model = lgb.LGBMClassifier(
            n_estimators=mp["n_estimators"],
            learning_rate=mp["learning_rate"],
            max_depth=mp["max_depth"],
            num_leaves=mp["num_leaves"],
            min_child_samples=mp["min_child_samples"],
            random_state=mp["random_state"],
            scale_pos_weight=mp.get("scale_pos_weight", 11),
            verbose=-1
        )
        model.fit(X_train, y_train)

        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        mlflow.log_metric("train_auc", train_auc)
        print(f"[train] Train AUC: {train_auc:.4f}")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        # Use native LightGBM flavor instead of sklearn flavor — avoids skops trust error
        mlflow.lightgbm.log_model(model, "model")
        print("[train] Model saved to models/model.pkl and logged to MLflow")

if __name__ == "__main__":
    params = load_params()
    train(params)