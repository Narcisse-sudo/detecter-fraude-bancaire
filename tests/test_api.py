from __future__ import annotations

import io

import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def sample_payload():
    return {
        "Time": 0.0,
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "Amount": 0.0,
    }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_lists_endpoints():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["predict"] == "/predict"


def test_predict():
    response = client.post("/predict", json=sample_payload())
    # 200 when a trained model is present, 503 when artifacts are absent.
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert 0.0 <= data["probability"] <= 1.0
        assert isinstance(data["is_fraud"], bool)
        assert "threshold" in data


def test_predict_rejects_extra_fields():
    payload = sample_payload()
    payload["unexpected"] = 1.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch_requires_input():
    response = client.post("/predict_batch")
    assert response.status_code == 400


def test_predict_batch_rejects_csv_with_missing_columns():
    csv = io.BytesIO(b"Time,Amount\n0,10\n")
    response = client.post("/predict_batch", files={"file": ("bad.csv", csv, "text/csv")})
    assert response.status_code == 422


def test_predict_batch_accepts_valid_csv():
    df = pd.DataFrame([sample_payload(), sample_payload()])
    csv = io.BytesIO(df.to_csv(index=False).encode())
    response = client.post("/predict_batch", files={"file": ("ok.csv", csv, "text/csv")})
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert body["count"] == 2
        assert len(body["results"]) == 2
