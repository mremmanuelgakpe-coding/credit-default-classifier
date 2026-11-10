"""Métriques usuelles en risque de crédit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true, proba) -> float:
    """Kolmogorov-Smirnov : écart max entre les distributions de score bons / mauvais."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(np.max(tpr - fpr))


def score_metrics(y_true, proba) -> dict:
    auc = roc_auc_score(y_true, proba)
    return {
        "roc_auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(ks_statistic(y_true, proba), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
    }


def expected_cost(y_true, proba, threshold, cost_fn, cost_fp) -> float:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return (cost_fn * fn + cost_fp * fp) / len(y_true)


def best_threshold(y_true, proba, cost_fn, cost_fp) -> tuple[float, float]:
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    costs = [expected_cost(y_true, proba, t, cost_fn, cost_fp) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def decision_metrics(y_true, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision_default": round(float(precision_score(y_true, pred)), 4),
        "recall_default": round(float(recall_score(y_true, pred)), 4),
        "rejection_rate": round(float(pred.mean()), 4),
        "default_rate_accepted": round(float(y_true[pred == 0].mean()), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def decile_table(y_true, proba) -> pd.DataFrame:
    """Taux de défaut observé par décile de score (1 = plus risqué)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": proba})
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    out = df.groupby("decile", observed=True).agg(
        n=("y", "size"), pd_moyenne=("p", "mean"), taux_defaut_observe=("y", "mean")
    )
    out["capture_cumulee_defauts"] = out["taux_defaut_observe"].mul(out["n"]).cumsum() / df["y"].sum()
    return out.round(4).reset_index()
