# %%
from __future__ import annotations

import json

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from utils.config import ARTIFACTS_DIR, EXPECTED_COLUMNS

app = FastAPI(title="Fraud Detection API", version="0.1.0")

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
PREPROCESS_PATH = ARTIFACTS_DIR / "preprocess.joblib"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
TEST_METRICS_PATH = ARTIFACTS_DIR / "test_metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.json"


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Time: float = Field(..., ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    probability: float
    is_fraud: bool
    threshold: float
    model_loaded: bool


def _load_threshold() -> float:
    """Lit le seuil de décision optimisé à l'entraînement (0.5 par défaut s'il est absent)."""
    if THRESHOLD_PATH.exists():
        return json.loads(THRESHOLD_PATH.read_text()).get("threshold", 0.5)
    return 0.5


def _load_model():
    """Charge le modèle et le pipeline de préprocessing depuis ``artifacts/``.

    Renvoie ``(None, None)`` si les artefacts n'existent pas encore (modèle non entraîné),
    ce que les endpoints traduisent en HTTP 503.
    """
    if MODEL_PATH.exists() and PREPROCESS_PATH.exists():
        import joblib

        model = joblib.load(MODEL_PATH)
        preprocess = joblib.load(PREPROCESS_PATH)
        return model, preprocess
    return None, None


FEATURE_COLUMNS = EXPECTED_COLUMNS[:-1]


def _predict_df(df: pd.DataFrame) -> list[float]:
    """Renvoie la probabilité de fraude pour chaque ligne du DataFrame.

    Valide la présence des colonnes attendues (422 sinon), applique le préprocessing puis
    le modèle. Lève 503 si aucun modèle n'est chargé.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns: {missing}")
    df = df[FEATURE_COLUMNS]

    model, preprocess = _load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = preprocess.transform(df)
    probs = model.predict_proba(features)[:, 1]
    return [float(p) for p in probs]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
    }


@app.get("/")
def root() -> dict:
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "predict": "/predict",
        "predict_batch": "/predict_batch",
    }


@app.get("/model-info")
def model_info() -> dict:
    """Expose l'état du modèle : métriques validation/test (dont F1), seuil et top features."""
    payload = {
        "model_loaded": MODEL_PATH.exists(),
        "metrics": json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {},
        "test_metrics": (
            json.loads(TEST_METRICS_PATH.read_text()) if TEST_METRICS_PATH.exists() else {}
        ),
        "threshold": _load_threshold(),
    }
    if FEATURE_IMPORTANCE_PATH.exists():
        fi = json.loads(FEATURE_IMPORTANCE_PATH.read_text())
        top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
        payload["top_features"] = top
    return payload


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction) -> PredictionResponse:
    df = pd.DataFrame([transaction.model_dump()])
    probs = _predict_df(df)
    threshold = _load_threshold()
    model_loaded = MODEL_PATH.exists() and PREPROCESS_PATH.exists()
    probability = probs[0]
    return PredictionResponse(
        probability=probability,
        is_fraud=probability >= threshold,
        threshold=threshold,
        model_loaded=model_loaded,
    )


@app.post("/predict_batch")
def predict_batch(
    transactions: list[Transaction] | None = None,
    file: UploadFile | None = File(default=None),
) -> dict:
    if file is None and transactions is None:
        raise HTTPException(status_code=400, detail="Provide transactions or CSV file")

    if file is not None:
        try:
            df = pd.read_csv(file.file)
        except Exception as exc:  # noqa: BLE001 - surface parse errors to the client
            raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc
    else:
        df = pd.DataFrame([t.model_dump() for t in transactions or []])

    probs = _predict_df(df)
    threshold = _load_threshold()
    results = [
        {"probability": p, "is_fraud": p >= threshold, "threshold": threshold} for p in probs
    ]
    return {"count": len(results), "results": results}


# %%
