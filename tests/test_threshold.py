from __future__ import annotations

import numpy as np

from models.train import _evaluate, _select_threshold


def test_threshold_is_a_probability():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.random(200)
    threshold = _select_threshold(y_true, y_prob)
    assert 0.0 <= threshold <= 1.0


def test_threshold_separates_clean_signal():
    # Perfectly separable: positives have high scores, negatives low.
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])
    threshold = _select_threshold(y_true, y_prob)
    preds = (y_prob >= threshold).astype(int)
    assert (preds == y_true).all()


def test_min_recall_constraint_is_met():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.55, 0.6, 0.3, 0.65, 0.8, 0.9])
    # F1-optimal threshold may sacrifice recall; min_recall forces >= 0.75.
    threshold = _select_threshold(y_true, y_prob, min_recall=0.75)
    recall = (y_prob[y_true == 1] >= threshold).mean()
    assert recall >= 0.75


def test_min_recall_trades_precision_for_recall():
    # 5 negatives / 5 positives, positives generally scored higher.
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.45, 0.55, 0.4, 0.6, 0.7, 0.8, 0.9])

    # F1-optimal sacrifices the hardest positive (recall 0.8, threshold ~0.6).
    f1_thr = _select_threshold(y_true, y_prob)
    assert (y_prob[y_true == 1] >= f1_thr).mean() < 1.0

    # Forcing full recall lowers the threshold to catch every positive.
    recall_thr = _select_threshold(y_true, y_prob, min_recall=0.95)
    assert (y_prob[y_true == 1] >= recall_thr).mean() == 1.0
    assert recall_thr <= f1_thr


def test_evaluate_reports_confusion_matrix():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = _evaluate(y_true, y_prob, threshold=0.5)
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0  # perfect predictions -> perfect F1
    assert 0.0 <= metrics["pr_auc"] <= 1.0
