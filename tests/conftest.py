from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.config import EXPECTED_COLUMNS


def make_synthetic_df(rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """A small, well-formed dataset matching the expected creditcard schema."""
    rng = np.random.default_rng(seed)
    data = {
        "Time": rng.integers(0, 172800, size=rows).astype(float),
        **{f"V{i}": rng.normal(0, 1, size=rows) for i in range(1, 29)},
        "Amount": np.abs(rng.normal(50, 30, size=rows)),
    }
    df = pd.DataFrame(data)
    # ~10% positives so stratified splits keep both classes in every fold.
    df["Class"] = (rng.random(rows) < 0.1).astype(int)
    return df[EXPECTED_COLUMNS]


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_df()


@pytest.fixture
def synthetic_csv(tmp_path: Path, synthetic_df: pd.DataFrame) -> Path:
    csv_path = tmp_path / "creditcard.csv"
    synthetic_df.to_csv(csv_path, index=False)
    return csv_path
