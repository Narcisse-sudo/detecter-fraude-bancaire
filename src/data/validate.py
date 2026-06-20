# %%
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pandera import Check, Column, DataFrameSchema

from utils.config import EXPECTED_COLUMNS, METADATA_PATH
from utils.io import write_json


def build_schema() -> DataFrameSchema:
    """Définit le schéma pandera attendu : types, non-nullité, bornes et domaine de ``Class``."""
    columns = {
        "Time": Column(float, nullable=False, checks=Check.ge(0), coerce=True),
        **{f"V{i}": Column(float, nullable=False, coerce=True) for i in range(1, 29)},
        "Amount": Column(float, nullable=False, checks=Check.ge(0), coerce=True),
        "Class": Column(int, nullable=False, checks=Check.isin([0, 1]), coerce=True),
    }
    return DataFrameSchema(columns=columns, strict=False)


def validate_dataset(csv_path: Path) -> dict[str, Any]:
    """Valide le CSV contre le schéma et calcule des métadonnées de qualité des données.

    Vérifie la présence des colonnes attendues, valide le schéma (lève si non conforme),
    puis collecte des stats (nb de lignes, doublons, valeurs manquantes, distribution des
    classes, stats Amount/Time). Le résultat est aussi écrit dans ``data/metadata.json``.
    """
    df = pd.read_csv(csv_path)
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    schema = build_schema()
    schema.validate(df[EXPECTED_COLUMNS], lazy=True)

    duplicates = int(df.duplicated().sum())
    missing_total = int(df.isna().sum().sum())

    metadata = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "expected_columns": EXPECTED_COLUMNS,
        "missing_total": missing_total,
        "duplicates": duplicates,
        "class_distribution": df["Class"].value_counts(normalize=True).to_dict(),
        "amount_stats": {
            "min": float(df["Amount"].min()),
            "max": float(df["Amount"].max()),
            "mean": float(df["Amount"].mean()),
            "median": float(df["Amount"].median()),
        },
        "time_stats": {
            "min": float(df["Time"].min()),
            "max": float(df["Time"].max()),
            "mean": float(df["Time"].mean()),
            "median": float(df["Time"].median()),
        },
    }

    write_json(METADATA_PATH, metadata)
    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default=None)
    args = parser.parse_args()

    default_path = Path("data/raw/creditcard.csv")
    csv_path = Path(args.csv_path) if args.csv_path else default_path
    meta = validate_dataset(csv_path)
    print(meta)

# %%
