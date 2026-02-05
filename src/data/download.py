from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from src.config import PATHS

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
MIRROR_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DATA_FILENAME = "creditcard.csv"

EXPECTED_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class",
]


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    source: str
    sha256: str


def ensure_dirs() -> None:
    PATHS.raw_dir.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_schema(path: Path, expected: Iterable[str]) -> None:
    df = pd.read_csv(path, nrows=5)
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")


def try_kaggle_download(target_path: Path) -> bool:
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    if not (kaggle_username and kaggle_key):
        return False

    if target_path.exists():
        return True

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(PATHS.raw_dir),
        "--unzip",
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "Kaggle CLI introuvable. Installez-le via `pip install kaggle` "
            "ou utilisez le fallback miroir."
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Échec du téléchargement via Kaggle.") from exc

    return target_path.exists()


def download_from_mirror(target_path: Path) -> bool:
    if target_path.exists():
        return True

    response = requests.get(MIRROR_URL, timeout=60)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    return target_path.exists()


def create_synthetic_sample(target_path: Path, n_rows: int = 200) -> None:
    rng = np.random.default_rng(42)
    data = {
        "Time": rng.uniform(0, 172800, size=n_rows),
        **{f"V{i}": rng.normal(size=n_rows) for i in range(1, 29)},
        "Amount": rng.lognormal(mean=3.0, sigma=1.0, size=n_rows),
        "Class": rng.choice([0, 1], size=n_rows, p=[0.995, 0.005]),
    }
    df = pd.DataFrame(data)
    df.to_csv(target_path, index=False)


def write_metadata(result: DownloadResult) -> None:
    metadata = {
        "path": str(result.path),
        "source": result.source,
        "sha256": result.sha256,
    }
    (PATHS.raw_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def download_dataset() -> DownloadResult:
    ensure_dirs()
    target_path = PATHS.raw_dir / DATA_FILENAME

    if try_kaggle_download(target_path):
        validate_schema(target_path, EXPECTED_COLUMNS)
        result = DownloadResult(target_path, "kaggle", sha256_file(target_path))
        write_metadata(result)
        return result

    try:
        if download_from_mirror(target_path):
            validate_schema(target_path, EXPECTED_COLUMNS)
            result = DownloadResult(target_path, "mirror", sha256_file(target_path))
            write_metadata(result)
            return result
    except Exception:
        pass

    sample_path = PATHS.raw_dir / "creditcard_sample.csv"
    create_synthetic_sample(sample_path)
    validate_schema(sample_path, EXPECTED_COLUMNS)
    result = DownloadResult(sample_path, "synthetic", sha256_file(sample_path))
    write_metadata(result)
    return result


def main() -> None:
    result = download_dataset()
    print(f"Dataset prêt: {result.path} (source={result.source})")


if __name__ == "__main__":
    main()
