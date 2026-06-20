# %%
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import joblib
import pandas as pd

from utils.config import ARTIFACTS_DIR, EXPECTED_COLUMNS, PROCESSED_DIR


def load_artifacts(model_path: Path, preprocess_path: Path):
    model = joblib.load(model_path)
    preprocess = joblib.load(preprocess_path)
    return model, preprocess


def predict_probabilities(
    records: Iterable[dict],
    model_path: Path,
    preprocess_path: Path,
) -> list[float]:
    df = pd.DataFrame(list(records))
    df = df[EXPECTED_COLUMNS[:-1]]
    model, preprocess = load_artifacts(model_path, preprocess_path)
    features = preprocess.transform(df)
    probs = model.predict_proba(features)[:, 1]
    return [float(p) for p in probs]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", nargs="?", default=None)
    parser.add_argument("model_path", nargs="?", default=None)
    parser.add_argument("preprocess_path", nargs="?", default=None)
    args, _ = parser.parse_known_args()

    default_input = PROCESSED_DIR / "sample_payload.json"
    default_model = ARTIFACTS_DIR / "model.joblib"
    default_preprocess = ARTIFACTS_DIR / "preprocess.joblib"

    input_json = Path(args.input_json) if args.input_json else default_input
    model_path = Path(args.model_path) if args.model_path else default_model
    preprocess_path = Path(args.preprocess_path) if args.preprocess_path else default_preprocess

    payload = json.loads(input_json.read_text())
    probs = predict_probabilities(payload, model_path, preprocess_path)
    print(probs)

# %%
