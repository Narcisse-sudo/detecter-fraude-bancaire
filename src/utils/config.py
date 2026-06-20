from __future__ import annotations

import os
from pathlib import Path

# Le défaut (relatif au code) convient en local (install editable). En conteneur, le package
# vit dans site-packages : on surcharge alors les chemins via les variables d'environnement
# FRAUD_DATA_DIR / FRAUD_ARTIFACTS_DIR, qui pointent vers les volumes montés.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FRAUD_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path(os.environ.get("FRAUD_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
METADATA_PATH = DATA_DIR / "metadata.json"

RAW_FILENAME = "creditcard.csv"

EXPECTED_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class",
]
