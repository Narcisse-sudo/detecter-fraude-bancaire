# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from features.build_features import add_basic_features
from utils.config import ARTIFACTS_DIR, EXPECTED_COLUMNS, PROCESSED_DIR


@dataclass
class SplitPaths:
    train: Path
    val: Path
    test: Path


def build_preprocess_pipeline() -> Pipeline:
    """Construit le pipeline de préprocessing : feature engineering puis standardisation.

    Encapsuler ces étapes dans un ``Pipeline`` sklearn garantit que les mêmes
    transformations (apprises sur le train) sont rejouées à l'identique à l'inférence.
    """
    return Pipeline(
        [
            ("feature_builder", FunctionTransformer(add_basic_features, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )


def preprocess_and_split(
    csv_path: Path,
    output_dir: Path = PROCESSED_DIR,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    artifact_dir: Path | None = ARTIFACTS_DIR,
) -> tuple[SplitPaths, Path]:
    """Découpe les données en train/val/test stratifiés et applique le préprocessing.

    Le découpage est **stratifié** sur ``Class`` pour conserver la proportion de fraudes
    dans chaque split. Crucial : le pipeline est ``fit`` **uniquement sur le train** puis
    appliqué (``transform``) à val/test — ce qui évite toute fuite de données. Les splits
    sont écrits en parquet et le pipeline sérialisé dans ``output_dir`` ; une copie est
    aussi déposée dans ``artifact_dir`` pour l'API (passer ``None`` pour la désactiver,
    p. ex. dans les tests, afin de ne pas écraser les artefacts de production).
    """
    df = pd.read_csv(csv_path)
    df = df[EXPECTED_COLUMNS]

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, stratify=y, random_state=random_state
    )
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, stratify=y_temp, random_state=random_state
    )

    pipeline = build_preprocess_pipeline()
    X_train_scaled = pipeline.fit_transform(X_train)
    X_val_scaled = pipeline.transform(X_val)
    X_test_scaled = pipeline.transform(X_test)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    feature_cols = add_basic_features(X_train).columns
    pd.DataFrame(X_train_scaled, columns=feature_cols).assign(Class=y_train.values).to_parquet(
        train_path, index=False
    )
    pd.DataFrame(X_val_scaled, columns=feature_cols).assign(Class=y_val.values).to_parquet(
        val_path, index=False
    )
    pd.DataFrame(X_test_scaled, columns=feature_cols).assign(Class=y_test.values).to_parquet(
        test_path, index=False
    )

    preprocess_path = output_dir / "preprocess.joblib"
    joblib.dump(pipeline, preprocess_path)

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, artifact_dir / "preprocess.joblib")

    return SplitPaths(train=train_path, val=val_path, test=test_path), preprocess_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default=None)
    args = parser.parse_args()

    default_path = Path("data/raw/creditcard.csv")
    csv_path = Path(args.csv_path) if args.csv_path else default_path
    splits, preprocess_path = preprocess_and_split(csv_path)
    print(splits)
    print(preprocess_path)

# %%
