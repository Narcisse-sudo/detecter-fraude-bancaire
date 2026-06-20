from __future__ import annotations

import pandas as pd
import pytest

from data.validate import validate_dataset


def test_validate_returns_metadata(synthetic_csv, monkeypatch, tmp_path):
    # Redirect the metadata output so the test does not touch the repo.
    monkeypatch.setattr("data.validate.METADATA_PATH", tmp_path / "metadata.json")
    meta = validate_dataset(synthetic_csv)

    assert meta["rows"] == len(pd.read_csv(synthetic_csv))
    assert meta["duplicates"] >= 0
    assert meta["missing_total"] == 0
    assert set(meta["class_distribution"].keys()) <= {0, 1}
    assert meta["amount_stats"]["min"] >= 0


def test_validate_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"Time": [1.0], "Amount": [2.0], "Class": [0]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        validate_dataset(bad)
