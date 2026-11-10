"""Configuration centrale du projet (chemins, variables, paramètres métier)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "credit_risk_dataset.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Empreinte SHA-256 du fichier Kaggle "laotse/credit-risk-dataset" (32 581 lignes).
# Permet de vérifier qu'on entraîne bien sur la donnée de référence.
DATA_SHA256 = "ce3c6d2167717bf1627d1c0c81cbccd28323cd4aa7b96d542599366d5ff6aac8"

TARGET = "loan_status"  # 1 = défaut, 0 = remboursé
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

NUMERIC_APPLICATION = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL_APPLICATION = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]

# Variables fixées par le prêteur APRÈS sa propre évaluation du risque.
# Elles sont très prédictives mais reflètent une notation déjà existante :
# on les isole pour pouvoir mesurer leur poids (analyse de sensibilité).
NUMERIC_PRICING = ["loan_int_rate"]
CATEGORICAL_PRICING = ["loan_grade"]

FEATURE_SETS = {
    # Score d'octroi : uniquement ce que l'on sait du client et de sa demande
    "application": (NUMERIC_APPLICATION, CATEGORICAL_APPLICATION),
    # Score complet : ajoute la note interne et le taux (benchmark Kaggle)
    "full": (
        NUMERIC_APPLICATION + NUMERIC_PRICING,
        CATEGORICAL_APPLICATION + CATEGORICAL_PRICING,
    ),
}
DEFAULT_FEATURE_SET = "full"

# Matrice de coûts métier (relative) pour choisir le seuil de décision.
# Un défaut non détecté (faux négatif) coûte ~ perte sur le capital ;
# un bon client refusé (faux positif) coûte ~ marge d'intérêt perdue.
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0
