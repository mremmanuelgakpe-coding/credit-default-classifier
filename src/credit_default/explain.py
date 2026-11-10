"""Explications individuelles (valeurs SHAP) et motifs de décision.

On utilise le TreeSHAP natif de LightGBM / XGBoost (`pred_contrib`) : valeurs SHAP
exactes, sans dépendance supplémentaire en production. Les contributions sont en
log-odds : base + somme des contributions = logit(probabilité de défaut).
Les colonnes issues du prétraitement (one-hot, indicateurs de manquants) sont
ré-agrégées sur les variables d'origine, compréhensibles par un analyste crédit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS_FR = {
    "person_age": "Âge",
    "person_income": "Revenu annuel",
    "person_home_ownership": "Statut d'occupation du logement",
    "person_emp_length": "Ancienneté professionnelle",
    "loan_intent": "Objet du prêt",
    "loan_grade": "Note interne du prêt",
    "loan_amnt": "Montant du prêt",
    "loan_int_rate": "Taux d'intérêt",
    "loan_percent_income": "Part du revenu consacrée au prêt",
    "cb_person_default_on_file": "Défaut antérieur au bureau de crédit",
    "cb_person_cred_hist_length": "Ancienneté de l'historique de crédit",
}


def _feature_map(transformed_names, raw_features, categorical):
    """Associe chaque colonne transformée à sa variable d'origine."""
    cats = sorted(categorical, key=len, reverse=True)  # préfixe le plus long d'abord
    mapping = []
    for name in transformed_names:
        if name in raw_features:
            mapping.append(name)
        elif name.startswith("missingindicator_"):
            mapping.append(name.removeprefix("missingindicator_"))
        else:
            mapping.append(next(c for c in cats if name.startswith(c + "_")))
    return mapping


def shap_values(pipeline, X: pd.DataFrame, categorical: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Valeurs SHAP par variable d'origine (log-odds) + valeur de base."""
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    Xt = prep.transform(X)
    names = list(prep.get_feature_names_out())
    if hasattr(model, "booster_"):  # LightGBM
        contrib = model.predict(Xt, pred_contrib=True)
    elif hasattr(model, "get_booster"):  # XGBoost
        import xgboost as xgb

        contrib = model.get_booster().predict(xgb.DMatrix(Xt, feature_names=None), pred_contribs=True)
    else:
        raise TypeError(f"Explications non supportées pour {type(model).__name__}")
    contrib = np.asarray(contrib)
    base, contrib = contrib[:, -1], contrib[:, :-1]
    mapping = _feature_map(names, list(X.columns), categorical)
    agg = pd.DataFrame(contrib, columns=mapping).T.groupby(level=0, sort=False).sum().T
    return agg[list(X.columns)], base


def reason_codes(X: pd.DataFrame, shap_df: pd.DataFrame, n_risk: int = 3, n_protect: int = 2) -> list[dict]:
    """Pour chaque dossier : principaux facteurs aggravants et atténuants."""
    out = []
    for i in range(len(X)):
        row = shap_df.iloc[i].sort_values()

        def item(feat):
            val = X.iloc[i][feat]
            val = None if pd.isna(val) else (val.item() if hasattr(val, "item") else val)
            return {
                "feature": feat,
                "label": LABELS_FR.get(feat, feat),
                "value": val,
                "impact": round(float(row[feat]), 4),
            }

        risk = [item(f) for f in row.index[::-1][:n_risk] if row[f] > 0]
        protect = [item(f) for f in row.index[:n_protect] if row[f] < 0]
        out.append({"risk_factors": risk, "protective_factors": protect})
    return out
