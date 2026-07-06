import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def preprocess(params):
    raw_path = params["data"]["raw_path"]
    out_dir  = params["data"]["processed_path"]
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(raw_path)
    print(f"[preprocess] Loaded {len(df)} rows, {df.shape[1]} columns")

    # Drop columns with >40% missing
    thresh = 0.4
    missing_frac = df.isnull().mean()
    cols_to_drop = missing_frac[missing_frac > thresh].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"[preprocess] Dropped {len(cols_to_drop)} high-missing columns")

    # Separate target
    y = df["TARGET"]
    df.drop(columns=["TARGET", "SK_ID_CURR"], inplace=True, errors="ignore")

    # Encode categoricals
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
    print(f"[preprocess] Encoded {len(cat_cols)} categorical columns: {cat_cols}")

    # Fill remaining nulls with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["random_state"],
        stratify=y
    )

    X_train.to_csv(f"{out_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{out_dir}/X_test.csv",  index=False)
    y_train.to_csv(f"{out_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{out_dir}/y_test.csv",  index=False)

    # Save feature names for API use
    feature_names = X_train.columns.tolist()
    import json
    with open(f"{out_dir}/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    print(f"[preprocess] Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"[preprocess] Features: {len(feature_names)}")
    print(f"[preprocess] Target distribution:\n{y_train.value_counts(normalize=True).round(3)}")

if __name__ == "__main__":
    params = load_params()
    preprocess(params)