# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import EXPECTED_COLUMNS
from utils.io import write_json


@dataclass
class DriftReport:
    psi: dict[str, float]
    overall: float


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calcule le Population Stability Index entre deux distributions.

    Le PSI mesure le déplacement d'une distribution par rapport à une référence : on
    découpe la référence en déciles, puis on compare les proportions tombant dans chaque
    bin. Repères usuels : < 0.1 stable, 0.1–0.25 dérive modérée, > 0.25 dérive forte.
    ``eps`` évite les ``log(0)`` / divisions par zéro sur les bins vides.
    """
    eps = 1e-6
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.quantile(expected, quantiles)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = expected_counts / (expected_counts.sum() + eps)
    actual_perc = actual_counts / (actual_counts.sum() + eps)

    return float(
        np.sum((expected_perc - actual_perc) * np.log((expected_perc + eps) / (actual_perc + eps)))
    )


def compute_drift(reference_csv: Path, current_csv: Path, out_path: Path) -> DriftReport:
    """Calcule le PSI par feature entre un jeu de référence et un jeu courant.

    Renvoie le PSI de chaque colonne + un PSI moyen global, et écrit le rapport en JSON.
    Utile pour détecter quand les données de production s'écartent de celles d'entraînement
    (signal qu'un ré-entraînement peut être nécessaire).
    """
    ref = pd.read_csv(reference_csv)[EXPECTED_COLUMNS[:-1]]
    cur = pd.read_csv(current_csv)[EXPECTED_COLUMNS[:-1]]

    psi_scores = {}
    for col in ref.columns:
        psi_scores[col] = _psi(ref[col].values, cur[col].values)

    overall = float(np.mean(list(psi_scores.values())))
    report = DriftReport(psi=psi_scores, overall=overall)
    write_json(out_path, {"overall": overall, "psi": psi_scores})
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("reference_csv", nargs="?", default=None)
    parser.add_argument("current_csv", nargs="?", default=None)
    parser.add_argument("out_path", nargs="?", default=None)
    args = parser.parse_args()

    default_ref = Path("data/raw/creditcard.csv")
    default_cur = Path("data/raw/creditcard.csv")
    default_out = Path("data/processed/drift_report.json")

    reference = Path(args.reference_csv) if args.reference_csv else default_ref
    current = Path(args.current_csv) if args.current_csv else default_cur
    out_path = Path(args.out_path) if args.out_path else default_out

    report = compute_drift(reference, current, out_path)
    print(report)

# %%
