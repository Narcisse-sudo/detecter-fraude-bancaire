from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("Credit Card Fraud Detection")
st.caption("Upload a CSV or run a single transaction prediction.")


@st.cache_data(show_spinner=False)
def get_model_info() -> dict[str, Any]:
    try:
        return requests.get(f"{API_URL}/model-info", timeout=10).json()
    except requests.RequestException:
        return {"model_loaded": False}


@st.cache_data(show_spinner=False)
def get_health() -> dict[str, Any]:
    try:
        return requests.get(f"{API_URL}/health", timeout=5).json()
    except requests.RequestException:
        return {"status": "down"}


with st.sidebar:
    st.caption(f"API URL: {API_URL}")
    health = get_health()
    if health.get("status") == "ok":
        st.success("API: healthy")
    else:
        st.error("API: unreachable")

    info = get_model_info()
    st.subheader("Model Info")

    # Key performance metrics on the held-out test set (F1 first: classes are imbalanced).
    test_metrics = info.get("test_metrics", {})
    if test_metrics:
        st.caption("Test set performance")
        c1, c2 = st.columns(2)
        c1.metric("F1-score", f"{test_metrics.get('f1', 0):.3f}")
        c2.metric("PR-AUC", f"{test_metrics.get('pr_auc', 0):.3f}")
        c3, c4 = st.columns(2)
        c3.metric("Precision", f"{test_metrics.get('precision', 0):.3f}")
        c4.metric("Recall", f"{test_metrics.get('recall', 0):.3f}")
        st.caption(f"Decision threshold: {info.get('threshold', 0.5):.4f}")

    with st.expander("Raw model info"):
        st.json(info)

    if "top_features" in info:
        st.subheader("Top Features")
        fi_df = pd.DataFrame(info["top_features"], columns=["feature", "importance"])
        st.bar_chart(fi_df.set_index("feature"))


st.subheader("Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview", df.head())
    if st.button("Predict Batch"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        try:
            response = requests.post(f"{API_URL}/predict_batch", files=files, timeout=60)
            response.raise_for_status()
            results = response.json()
            preds = pd.DataFrame(results["results"])
            output = pd.concat([df.reset_index(drop=True), preds], axis=1)
            st.write(output.head())
            st.download_button(
                "Download Predictions",
                output.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except requests.RequestException as exc:
            st.error(f"Prediction failed: {exc}")

st.subheader("Single Transaction")

with st.form("single_tx"):
    cols = st.columns(4)
    inputs = {}
    for idx, col_name in enumerate(["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]):
        with cols[idx % 4]:
            inputs[col_name] = st.number_input(col_name, value=0.0, format="%.6f")
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        response = requests.post(f"{API_URL}/predict", json=inputs, timeout=30)
        response.raise_for_status()
        st.json(response.json())
    except requests.RequestException as exc:
        st.error(f"Prediction failed: {exc}")
