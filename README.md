# Détection de fraude bancaire

Projet end-to-end de détection de fraude carte bancaire (datasets publics), avec pipeline data, modélisation, API FastAPI et UI Streamlit.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]

# Téléchargement des données
python scripts/download_data.py
```

### Docker (à venir)

```bash
docker compose up --build
```

## Structure du repo

```
.
├── app/                # API FastAPI + UI Streamlit
├── data/               # data/raw et data/processed (ignorés par git)
├── notebooks/          # EDA
├── scripts/            # scripts CLI
├── src/                # code Python
└── tests/              # tests unitaires
```

## Données

Le dataset principal attendu est **Credit Card Fraud Detection** (transactions réelles anonymisées). Téléchargement supporté via :

1. **Kaggle API** si `KAGGLE_USERNAME` et `KAGGLE_KEY` sont disponibles.
2. **Fallback** vers un miroir public si Kaggle n'est pas configuré.
3. **Échantillon synthétique** uniquement pour tests si le téléchargement échoue (pas pour l'entraînement final).

Voir `scripts/download_data.py` pour les détails.
