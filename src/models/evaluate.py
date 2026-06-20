# %%
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.config import ARTIFACTS_DIR, PROCESSED_DIR


def _load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Charge un split parquet et le sépare en features ``X`` et cible ``y`` (colonne ``Class``)."""
    df = pd.read_parquet(path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def evaluate(
    model_path: Path, test_path: Path, threshold_path: Path | None = None
) -> dict[str, float]:
    """Évalue un modèle sauvegardé sur le jeu de test au seuil fourni.

    Le seuil est lu depuis ``threshold_path`` (celui optimisé au F1 à l'entraînement) ;
    à défaut, on retombe sur 0.5. Les métriques (dont F1 et PR-AUC, pertinentes en
    contexte déséquilibré) sont écrites dans ``artifacts/test_metrics.json``.
    """
    model = joblib.load(model_path)
    X_test, y_test = _load_split(test_path)

    threshold = 0.5
    if threshold_path and threshold_path.exists():
        threshold = json.loads(threshold_path.read_text()).get("threshold", 0.5)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    metrics = {
        "pr_auc": float(average_precision_score(y_test, probs)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", default=None)
    parser.add_argument("test_path", nargs="?", default=None)
    parser.add_argument("--threshold", default=None)
    args, _ = parser.parse_known_args()

    default_model = ARTIFACTS_DIR / "model.joblib"
    default_test = PROCESSED_DIR / "test.parquet"
    model_path = Path(args.model_path) if args.model_path else default_model
    test_path = Path(args.test_path) if args.test_path else default_test
    threshold_path = Path(args.threshold) if args.threshold else None

    metrics = evaluate(model_path, test_path, threshold_path)
    print(metrics)

# %%
