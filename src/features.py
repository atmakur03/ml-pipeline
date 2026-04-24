"""
features.py
PySpark feature engineering stage.
Reads raw Parquet, applies transformations, and writes
a clean feature set ready for model training.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/features"


def load_raw(input_dir: str = INPUT_DIR) -> pd.DataFrame:
    """Load most recent raw Parquet partition."""
    frames = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".parquet"):
                frames.append(pd.read_parquet(os.path.join(root, f)))
    if not frames:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    df = pd.concat(frames, ignore_index=True)
    print(f"[features] Loaded {len(df)} raw rows.")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day-of-week, and month from timestamp."""
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] <= 20)).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1-hour and 3-hour lag features for temperature and wind speed."""
    df = df.sort_values("time").reset_index(drop=True)
    df["temp_lag_1h"] = df["temperature_2m"].shift(1)
    df["temp_lag_3h"] = df["temperature_2m"].shift(3)
    df["wind_lag_1h"] = df["windspeed_10m"].shift(1)
    df["temp_rolling_6h"] = df["temperature_2m"].rolling(6, min_periods=1).mean()
    df["humidity_rolling_3h"] = df["relativehumidity_2m"].rolling(3, min_periods=1).mean()
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target: will_precipitate (next hour has precipitation > 0)."""
    df["will_precipitate"] = (df["precipitation"].shift(-1) > 0).astype(int)
    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize numeric feature columns."""
    numeric_cols = [
        "temperature_2m", "windspeed_10m", "relativehumidity_2m",
        "temp_lag_1h", "temp_lag_3h", "wind_lag_1h",
        "temp_rolling_6h", "humidity_rolling_3h",
    ]
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)
    return df


def write_features(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> str:
    """Write feature set to Parquet."""
    df = df.dropna()
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "features.parquet")
    df.to_parquet(filepath, index=False)
    print(f"[features] Wrote {len(df)} feature rows to {filepath}")
    return filepath


def run() -> str:
    """Full feature engineering stage."""
    df = load_raw()
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_target(df)
    df = normalize_features(df)
    path = write_features(df)
    return path


if __name__ == "__main__":
    run()
