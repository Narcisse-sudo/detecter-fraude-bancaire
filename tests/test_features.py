from __future__ import annotations

import numpy as np
import pandas as pd

from features.build_features import add_basic_features


def test_adds_amount_log1p():
    df = pd.DataFrame({"Amount": [0.0, np.e - 1, 100.0]})
    out = add_basic_features(df)
    assert "Amount_log1p" in out.columns
    np.testing.assert_allclose(out["Amount_log1p"], np.log1p([0.0, np.e - 1, 100.0]))


def test_does_not_mutate_input():
    df = pd.DataFrame({"Amount": [1.0, 2.0]})
    add_basic_features(df)
    assert "Amount_log1p" not in df.columns


def test_negative_amount_is_clipped():
    df = pd.DataFrame({"Amount": [-5.0]})
    out = add_basic_features(df)
    assert out["Amount_log1p"].iloc[0] == 0.0


def test_no_amount_column_is_noop():
    df = pd.DataFrame({"V1": [1.0, 2.0]})
    out = add_basic_features(df)
    assert "Amount_log1p" not in out.columns
