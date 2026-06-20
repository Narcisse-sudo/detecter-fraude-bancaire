from __future__ import annotations

import numpy as np

from data.monitor import _psi, compute_drift


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=5000)
    assert _psi(x, x) < 1e-3


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=5000)
    shifted = rng.normal(3, 1, size=5000)
    assert _psi(ref, shifted) > _psi(ref, ref + 0.05)


def test_compute_drift_writes_report(synthetic_csv, tmp_path):
    out = tmp_path / "drift.json"
    report = compute_drift(synthetic_csv, synthetic_csv, out)
    assert out.exists()
    assert report.overall < 1e-2  # same file -> no drift
    assert "Amount" in report.psi
