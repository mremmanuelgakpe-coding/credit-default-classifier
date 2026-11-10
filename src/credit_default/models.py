"""Prétraitement et catalogue des classifieurs comparés."""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import numpy as np

from . import config

# Variables très asymétriques : passage au log avant standardisation
SKEWED = {"person_income", "loan_amnt"}


def make_preprocessor(feature_set: str) -> ColumnTransformer:
    num, cat = config.FEATURE_SETS[feature_set]
    skewed = [c for c in num if c in SKEWED]
    other = [c for c in num if c not in SKEWED]
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    skewed_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("num", numeric, other),
            ("skew", skewed_pipe, skewed),
            ("cat", categorical, cat),
        ],
        verbose_feature_names_out=False,
    )


def candidate_models(random_state: int = config.RANDOM_STATE) -> dict:
    """Les familles du projet 5 : linéaire, distance, probabiliste, ensembles."""
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0),
        "knn": KNeighborsClassifier(n_neighbors=50, weights="distance"),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=random_state
        ),
        "xgboost": XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.8,
            eval_metric="auc",
            n_jobs=-1,
            random_state=random_state,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
        ),
    }


def build_pipeline(model, feature_set: str) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(feature_set)), ("model", model)])
