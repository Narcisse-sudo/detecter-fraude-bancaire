# %%
from __future__ import annotations

import numpy as np
import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features dérivées au DataFrame, sans muter l'entrée.

    Pour l'instant : ``Amount_log1p``, transformation log(1+x) du montant. Les montants de
    transaction sont très asymétriques (quelques très gros montants) ; le log compresse cette
    distribution et aide les modèles. ``clip(lower=0)`` protège contre d'éventuels négatifs.
    Si la colonne ``Amount`` est absente, la fonction est un no-op (renvoie une copie).
    """
    df = df.copy()
    if "Amount" in df.columns:
        df["Amount_log1p"] = np.log1p(df["Amount"].clip(lower=0))
    return df


# %%
