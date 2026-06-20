# %%
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.config import ARTIFACTS_DIR, PROCESSED_DIR


@dataclass
class TrainingArtifacts:
    model_path: Path
    metrics_path: Path
    threshold_path: Path


def _load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Charge un split parquet et le sépare en features ``X`` et cible ``y`` (colonne ``Class``)."""
    df = pd.read_parquet(path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def _select_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, min_recall: float | None = None
) -> float:
    """Choisit le seuil de décision sur la courbe précision/rappel.

    Deux stratégies, selon le besoin métier :

    - ``min_recall=None`` (défaut) : **maximise le F1**. Sur des classes déséquilibrées, le
      seuil 0.5 par défaut est rarement optimal ; on balaie tous les seuils et on garde celui
      qui équilibre au mieux précision et rappel. ``+1e-9`` évite la division par zéro.
    - ``min_recall=x`` : retient le seuil de **précision maximale parmi ceux atteignant un
      rappel ≥ x**. Utile quand rater une fraude coûte plus cher qu'une fausse alerte.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if min_recall is not None:
        # recall[:-1] est aligné sur thresholds et décroît quand le seuil augmente :
        # les indices valides forment un préfixe ; le dernier maximise donc la précision.
        valid = np.where(recall[:-1] >= min_recall)[0]
        if len(valid):
            return float(thresholds[valid[-1]])
        return 0.0  # contrainte inatteignable -> on prédit tout positif (rappel maximal)

    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
    best_idx = np.nanargmax(f1_scores)
    # precision_recall_curve renvoie un seuil de moins que de points (p, r) :
    # si l'optimum tombe sur le dernier point (sans seuil associé), on retombe sur 0.5.
    if best_idx >= len(thresholds):
        return 0.5
    return float(thresholds[best_idx])


def _evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Calcule les métriques de classification au seuil donné.

    Inclut les métriques adaptées au déséquilibre (PR-AUC, F1) en plus du ROC-AUC et de
    la matrice de confusion. ``pr_auc`` et ``roc_auc`` sont indépendants du seuil ;
    ``f1``/``precision``/``recall`` en dépendent.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _feature_importance(model, feature_names) -> dict[str, float]:
    """Associe chaque feature à son importance, si le modèle l'expose (sinon dict vide)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return {name: float(val) for name, val in zip(feature_names, importances, strict=False)}
    return {}


def train_models(
    train_path: Path,
    val_path: Path,
    calibrate: bool = True,
    min_recall: float | None = None,
) -> TrainingArtifacts:
    """Entraîne, calibre et sérialise le modèle de détection de fraude.

    Compare une baseline (régression logistique) à un modèle avancé (LightGBM), calibre
    optionnellement les probabilités (isotonic), sélectionne le seuil sur la validation
    (au F1, ou sous contrainte de ``min_recall``), puis sauvegarde modèle, métriques,
    seuil et importances dans ``artifacts/``.
    """
    X_train, y_train = _load_split(train_path)
    X_val, y_val = _load_split(val_path)

    print("Starting model training Logistic Regression...")
    print(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples.")
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    baseline.fit(X_train, y_train)
    base_probs = baseline.predict_proba(X_val)[:, 1]
    base_threshold = _select_threshold(y_val.values, base_probs, min_recall=min_recall)
    baseline_metrics = _evaluate(y_val.values, base_probs, base_threshold)

    print("Starting model training LightGBM...")
    print(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples.")
    advanced = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    advanced.fit(X_train, y_train)

    model = advanced
    print("Starting model calibration...")
    if calibrate:
        model = CalibratedClassifierCV(advanced, method="isotonic", cv=3)
        model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    threshold = _select_threshold(y_val.values, val_probs, min_recall=min_recall)
    metrics = _evaluate(y_val.values, val_probs, threshold)

    print("Saving artifacts...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "model.joblib"
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    threshold_path = ARTIFACTS_DIR / "threshold.json"
    featimp_path = ARTIFACTS_DIR / "feature_importance.json"

    joblib.dump(model, model_path)

    payload = {
        "baseline": baseline_metrics,
        "advanced": metrics,
        "calibrated": calibrate,
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    threshold_path.write_text(json.dumps({"threshold": threshold}, indent=2))
    feature_importance = _feature_importance(advanced, list(X_train.columns))
    featimp_path.write_text(json.dumps(feature_importance, indent=2))

    return TrainingArtifacts(model_path, metrics_path, threshold_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", nargs="?", default=None)
    parser.add_argument("val_path", nargs="?", default=None)
    parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="Sélectionne le seuil de précision max atteignant ce rappel (sinon max F1).",
    )
    args, _ = parser.parse_known_args()

    default_train = PROCESSED_DIR / "train.parquet"
    default_val = PROCESSED_DIR / "val.parquet"
    train_path = Path(args.train_path) if args.train_path else default_train
    val_path = Path(args.val_path) if args.val_path else default_val

    artifacts = train_models(train_path, val_path, min_recall=args.min_recall)


# %%
