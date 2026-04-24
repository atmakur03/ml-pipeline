"""
app.py
Streamlit dashboard: visualizes pipeline predictions,
feature distributions, and model metrics.
"""

import os
import json
import glob
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ML Pipeline Dashboard",
    page_icon="pipeline",
    layout="wide",
)

st.title("End-to-End ML Pipeline Dashboard")
st.caption("Weather-based precipitation prediction | Data Engineer: Vani Atmakur")


# ── Sidebar ────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "View",
    ["Predictions", "Feature Distributions", "Model Metrics"],
)


# ── Load data ──────────────────────────────────────
@st.cache_data
def load_predictions():
    files = sorted(glob.glob("data/predictions/*.csv"))
    if not files:
        return None
    return pd.read_csv(files[-1], parse_dates=["time"])


@st.cache_data
def load_features():
    path = "data/features/features.parquet"
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_metrics():
    path = "models/metrics.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Pages ──────────────────────────────────────────
if page == "Predictions":
    st.header("Batch Predictions")
    df = load_predictions()
    if df is None:
        st.warning("No prediction files found. Run pipeline.py first.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", len(df))
        col2.metric("Will Precipitate", int(df["prediction"].sum()))
        col3.metric("Avg Probability", f"{df['probability'].mean():.2%}")

        st.subheader("Prediction Timeline")
        st.line_chart(df.set_index("time")["probability"])

        st.subheader("Raw Predictions")
        st.dataframe(df.tail(48), use_container_width=True)


elif page == "Feature Distributions":
    st.header("Feature Distributions")
    df = load_features()
    if df is None:
        st.warning("No feature file found. Run pipeline.py first.")
    else:
        feature = st.selectbox(
            "Select Feature",
            ["temperature_2m", "windspeed_10m", "relativehumidity_2m",
             "temp_rolling_6h", "humidity_rolling_3h"],
        )
        st.bar_chart(df[feature])
        st.write(df[[feature]].describe())


elif page == "Model Metrics":
    st.header("Model Performance")
    metrics = load_metrics()
    if metrics is None:
        st.warning("No metrics file found. Run pipeline.py first.")
    else:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")

        st.subheader("Feature Importance")
        importance = metrics.get("feature_importance", {})
        imp_df = pd.DataFrame(
            list(importance.items()), columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=False)
        st.bar_chart(imp_df.set_index("Feature"))

        st.subheader("Classification Report")
        report = metrics.get("classification_report", {})
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
