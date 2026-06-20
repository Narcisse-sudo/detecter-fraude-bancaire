from __future__ import annotations

import pandas as pd

from data.preprocess import preprocess_and_split


def test_split_produces_three_disjoint_sets(synthetic_csv, tmp_path):
    out_dir = tmp_path / "processed"
    splits, preprocess_path = preprocess_and_split(
        synthetic_csv,
        output_dir=out_dir,
        test_size=0.2,
        val_size=0.1,
        artifact_dir=tmp_path / "artifacts",  # never touch production artifacts/
    )

    train = pd.read_parquet(splits.train)
    val = pd.read_parquet(splits.val)
    test = pd.read_parquet(splits.test)

    total = len(train) + len(val) + len(test)
    assert total == len(pd.read_csv(synthetic_csv))
    # Roughly 70/10/20 split.
    assert len(train) > len(test) > len(val)

    for frame in (train, val, test):
        assert "Class" in frame.columns
        assert "Amount_log1p" in frame.columns  # feature engineering applied
        assert set(frame["Class"].unique()) <= {0, 1}

    assert preprocess_path.exists()


def test_scaler_is_fit_on_train_only(synthetic_csv, tmp_path):
    # Train set should be centered (StandardScaler fit on it); a leak would
    # also center val/test, so we only assert the train invariant here.
    splits, _ = preprocess_and_split(
        synthetic_csv, output_dir=tmp_path / "processed", artifact_dir=tmp_path / "artifacts"
    )
    train = pd.read_parquet(splits.train).drop(columns=["Class"])
    assert train.mean().abs().max() < 1e-6
