"""
ingest.py
Fetches raw data from a public REST API (Open-Meteo weather API),
validates the schema, and writes partitioned Parquet to a local
data lake (simulating S3 in local mode).
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta

API_URL = "https://api.open-meteo.com/v1/forecast"
OUTPUT_DIR = "data/raw"

DEFAULT_PARAMS = {
    "latitude": 38.8977,
    "longitude": -77.0365,
    "hourly": "temperature_2m,precipitation,windspeed_10m,relativehumidity_2m",
    "timezone": "America/New_York",
    "past_days": 7,
}

REQUIRED_COLUMNS = [
    "time",
    "temperature_2m",
    "precipitation",
    "windspeed_10m",
    "relativehumidity_2m",
]


def fetch_data(params: dict = DEFAULT_PARAMS) -> dict:
    """Call the REST API and return raw JSON response."""
    print(f"[ingest] Fetching data from {API_URL} ...")
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_response(raw: dict) -> pd.DataFrame:
    """Parse JSON response into a flat DataFrame."""
    hourly = raw.get("hourly", {})
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Assert required columns exist and drop rows with all-null feature values."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[ingest] Missing columns in API response: {missing}")

    before = len(df)
    df = df.dropna(subset=[c for c in REQUIRED_COLUMNS if c != "time"])
    dropped = before - len(df)
    if dropped:
        print(f"[ingest] Dropped {dropped} rows with null values.")
    return df


def write_parquet(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> str:
    """Write DataFrame to partitioned Parquet file."""
    partition = datetime.utcnow().strftime("%Y/%m/%d")
    path = os.path.join(output_dir, partition)
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, "raw.parquet")
    df.to_parquet(filepath, index=False)
    print(f"[ingest] Wrote {len(df)} rows to {filepath}")
    return filepath


def run() -> str:
    """Full ingestion stage: fetch -> parse -> validate -> write."""
    raw = fetch_data()
    df = parse_response(raw)
    df = validate_schema(df)
    path = write_parquet(df)
    return path


if __name__ == "__main__":
    run()
