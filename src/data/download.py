# %%
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from utils.config import RAW_DIR, RAW_FILENAME

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"


def _write_kaggle_config(tmp_dir: Path, username: str, key: str) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_dir / "kaggle.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(config_path, 0o600)


def _download_from_kaggle(dest: Path) -> Path | None:
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key:
        return None

    from kaggle.api.kaggle_api_extended import KaggleApi

    kaggle_dir = dest / ".kaggle"
    _write_kaggle_config(kaggle_dir, username, key)
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=dest, unzip=True)
    csv_path = dest / RAW_FILENAME
    return csv_path if csv_path.exists() else None


def _download_from_mirror(dest: Path) -> Path | None:
    mirror_url = os.getenv("FRAUD_DATA_MIRROR_URL")
    if not mirror_url:
        return None

    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / RAW_FILENAME
    with requests.get(mirror_url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with csv_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return csv_path


def _generate_synthetic(dest: Path, rows: int = 1000, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    data = {
        "Time": rng.integers(0, 172800, size=rows),
        **{f"V{i}": rng.normal(0, 1, size=rows) for i in range(1, 29)},
        "Amount": np.abs(rng.normal(50, 30, size=rows)),
    }
    df = pd.DataFrame(data)
    df["Class"] = (rng.random(rows) < 0.02).astype(int)

    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / RAW_FILENAME
    df.to_csv(csv_path, index=False)
    return csv_path


def download_raw_dataset(dest: Path = RAW_DIR, download: bool = False) -> Path:
    dest.mkdir(parents=True, exist_ok=True)

    kaggle_path = _download_from_kaggle(dest)
    if kaggle_path:
        return kaggle_path

    mirror_path = _download_from_mirror(dest)
    if mirror_path:
        return mirror_path

    return _generate_synthetic(dest)


if __name__ == "__main__":
    data_exist = True
    if not data_exist:
        print("Downloading dataset...")
        path = download_raw_dataset()
        print(f"Dataset ready at: {path}")

# %%
