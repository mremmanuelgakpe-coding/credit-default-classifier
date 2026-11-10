"""Chargement, contrôle d'intégrité et nettoyage des données."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path = config.DATA_RAW, verify: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Voir data/README.md pour télécharger le dataset."
        )
    if verify:
        digest = sha256_of(path)
        if digest != config.DATA_SHA256:
            log.warning("Empreinte SHA-256 inattendue (%s) : fichier modifié ?", digest)
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Règles de nettoyage documentées. Retourne aussi un journal des corrections."""
    report: dict[str, int] = {"rows_in": len(df)}

    dups = df.duplicated()
    report["duplicates_removed"] = int(dups.sum())
    df = df.loc[~dups].copy()

    # Âges impossibles (> 100 ans) : erreurs de saisie, lignes supprimées
    bad_age = df["person_age"] > 100
    report["age_over_100_removed"] = int(bad_age.sum())
    df = df.loc[~bad_age]

    # Ancienneté pro incohérente (travail avant 14 ans) : mise à manquant,
    # l'imputation du pipeline prendra le relais
    bad_emp = df["person_emp_length"] > (df["person_age"] - 14)
    report["emp_length_set_to_nan"] = int(bad_emp.sum())
    df.loc[bad_emp, "person_emp_length"] = np.nan

    report["rows_out"] = len(df)
    report["default_rate"] = round(float(df[config.TARGET].mean()), 4)
    return df.reset_index(drop=True), report


def split_xy(df: pd.DataFrame, feature_set: str = config.DEFAULT_FEATURE_SET):
    num, cat = config.FEATURE_SETS[feature_set]
    return df[num + cat], df[config.TARGET].astype(int)
