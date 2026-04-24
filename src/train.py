"""
train.py
Model training stage.
Loads feature set, trains a RandomForest classifier,
logs accuracy and feature importance, and saves the model.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

FEATURES_PATH = "data/features/features.parquet"
MODEL_DIR = "models"
METRICS_PATH = "models/metrics.json"

FEATURE_COLS = [
    "temperature_2m", "windspeed_10m", "relativehumidity_2m",
    "temp_lag_1h", "temp_lag_3h", "wind_lag_1h",
    "temp_rolling_6h", "humidity_rolling_3h",
    "hour", "day_of_week", "month", "is_daytime",
]
TARGET_COL = "will_precipitate"


def load_features(path: str = FEATURES_PATH) -> pd.DataFrame:
    """Load engineered features from Parquet."""
    df = pd.read_parquet(path)
    print(f"[train] Loaded {len(df)} rows, {len(FEATURE_COLS)} features.")
    return df


def split_data(df: pd.DataFrame):
    """Split into train/test sets (80/20)."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[train] Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train) -> RandomForestClassifier:
    """Train RandomForest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[train] Model training complete.")
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    importance = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    metrics = {
        "accuracy": round(accuracy, 4),
        "classification_report": report,
        "feature_importance": importance,
    }
    print(f"[train] Accuracy: {accuracy:.4f}")
    return metrics


def save_model(model, model_dir: str = MODEL_DIR) -> str:
    """Serialize model to disk."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[train] Model saved to {path}")
    return path


def save_metrics(metrics: dict, path: str = METRICS_PATH) -> None:
    """Write metrics JSON to disk."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Metrics saved to {path}")


def run() -> dict:
    """Full training stage."""
    df = load_features()
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model)
    save_metrics(metrics)
    return metrics


if __name__ == "__main__":
    run()
