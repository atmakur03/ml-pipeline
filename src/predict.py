"""
predict.py
Batch inference stage.
Loads trained model, runs predictions on latest feature set,
and writes results to a local predictions CSV (simulating Redshift write).
"""

import os
import pickle
import pandas as pd
from datetime import datetime

FEATURES_PATH = "data/features/features.parquet"
MODEL_PATH = "models/model.pkl"
OUTPUT_DIR = "data/predictions"

FEATURE_COLS = [
    "temperature_2m", "windspeed_10m", "relativehumidity_2m",
    "temp_lag_1h", "temp_lag_3h", "wind_lag_1h",
    "temp_rolling_6h", "humidity_rolling_3h",
    "hour", "day_of_week", "month", "is_daytime",
]


def load_model(path: str = MODEL_PATH):
    """Load serialized model from disk."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[predict] Model loaded from {path}")
    return model


def load_features(path: str = FEATURES_PATH) -> pd.DataFrame:
    """Load feature set for inference."""
    df = pd.read_parquet(path)
    print(f"[predict] Loaded {len(df)} rows for inference.")
    return df


def run_inference(model, df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions and probabilities."""
    X = df[FEATURE_COLS]
    df["prediction"] = model.predict(X)
    df["probability"] = model.predict_proba(X)[:, 1].round(4)
    df["predicted_at"] = datetime.utcnow().isoformat()
    print(f"[predict] Generated {len(df)} predictions.")
    return df


def write_predictions(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> str:
    """Write predictions to CSV (Redshift simulation)."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"predictions_{timestamp}.csv")
    cols = ["time", "prediction", "probability", "predicted_at"]
    df[cols].to_csv(filepath, index=False)
    print(f"[predict] Predictions written to {filepath}")
    return filepath


def run() -> str:
    """Full inference stage."""
    model = load_model()
    df = load_features()
    df = run_inference(model, df)
    path = write_predictions(df)
    return path


if __name__ == "__main__":
    run()
