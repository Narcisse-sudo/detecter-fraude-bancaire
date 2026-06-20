# Credit Card Fraud Detection

End-to-end project: data ingestion, EDA, preprocessing, modeling, evaluation, and
deployment with FastAPI + Streamlit + Docker.

## Dataset

[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(`mlg-ulb/creditcardfraud`): 284,807 transactions, only **492 frauds (0.17 %)**. Features
`V1..V28` are anonymized PCA components; `Time` and `Amount` are raw. Target: `Class`
(1 = fraud). The strong class imbalance is why we rely on **F1 / PR-AUC** rather than
accuracy.

## Results

Held-out test set (LightGBM, isotonic-calibrated, threshold tuned for recall ≥ 0.80):

| Metric    | Value |
|-----------|-------|
| PR-AUC    | 0.836 |
| ROC-AUC   | 0.965 |
| **F1**    | **0.831** |
| Precision | 0.844 |
| Recall    | 0.818 |

Confusion matrix: TN 56849 · FP 15 · FN 18 · **TP 81** (81 / 99 frauds caught for 15 false
alerts). The decision threshold is selected on the validation set — by default to maximize
F1, or under a recall constraint via `--min-recall` (see step 4).

## Quickstart

### 1) Create uv environment + install deps

```bash
uv venv .venv
source .venv/bin/activate
UV_CACHE_DIR=.uv-cache uv pip install -e '.[dev,notebooks]'
```

### 2) Data download + validation

```bash
python -m data.download
python -m data.validate data/raw/creditcard.csv
```

> Kaggle download uses `KAGGLE_USERNAME` and `KAGGLE_KEY` (create an API token at
> kaggle.com/settings, and accept the dataset terms once on its page). If not set, fall back
> to `FRAUD_DATA_MIRROR_URL`. Last resort generates synthetic data (random labels — for
> pipeline/tests only, the model cannot learn from it).

### 2b) Drift monitoring (optional)

```bash
python -m data.monitor data/raw/creditcard.csv data/raw/creditcard.csv data/processed/drift_report.json
```

Reports per-feature PSI (Population Stability Index) between a reference and a current set.

### 3) Preprocess

```bash
python -m data.preprocess data/raw/creditcard.csv
```

Stratified train/val/test split + a fit-on-train-only pipeline (no leakage).

### 4) Train + evaluate

```bash
# Default: threshold chosen to maximize F1
python -m models.train data/processed/train.parquet data/processed/val.parquet

# Optional: bias the threshold toward catching more frauds (higher recall)
python -m models.train --min-recall 0.80

python -m models.evaluate artifacts/model.joblib data/processed/test.parquet --threshold artifacts/threshold.json
```

Both scripts default to the paths under `data/processed/` and `artifacts/`, so the
positional arguments are optional.

### 5) Run API + UI

```bash
uvicorn app.api.main:app --reload
streamlit run app/ui/streamlit_app.py
```

API root: `http://127.0.0.1:8000/`
API docs: `http://127.0.0.1:8000/docs`

Endpoints: `/health`, `/model-info` (metrics incl. F1 + threshold), `/predict`,
`/predict_batch` (CSV upload).

If your API runs elsewhere:

```bash
API_URL=http://127.0.0.1:8000 streamlit run app/ui/streamlit_app.py
```

### 6) Docker Compose

```bash
docker compose up --build
```

## Development

```bash
pytest            # test suite
ruff check .      # lint
black .           # format
pre-commit install  # run ruff + black automatically on commit
```

## Structure

- `notebooks/` — `01_eda.ipynb`, `02_preprocessing.ipynb`, `03_modeling.ipynb`
- `src/` — `data/` (download, validate, preprocess, monitor) + `features/` + `models/` (train, evaluate, predict) + `utils/`
- `app/api/main.py` — FastAPI service
- `app/ui/streamlit_app.py` — Streamlit UI
- `tests/` — pytest suite

## Notes

- Raw dataset and processed splits live in `data/` (gitignored).
- Model artifacts are saved to `artifacts/`.
- For notebooks, the `src/` directory is added to `PYTHONPATH` (handled in the notebooks).
```
