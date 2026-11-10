"""Entraînement, comparaison des modèles et export de l'artefact.

Usage :
    python -m credit_default.train                       # jeu de variables "full"
    python -m credit_default.train --feature-set application
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from . import __version__, config, plots
from .data import clean, load_raw, sha256_of, split_xy
from .explain import shap_values
from .metrics import best_threshold, decile_table, decision_metrics, expected_cost, score_metrics
from .models import build_pipeline, candidate_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def run(feature_set: str) -> dict:
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tag = feature_set

    raw = load_raw()
    df, cleaning = clean(raw)
    log.info("Nettoyage : %s", cleaning)

    X, y = split_xy(df, feature_set)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    # 1) Comparaison en validation croisée (le test n'est jamais utilisé pour choisir)
    rows, oof, test_probas, fitted = [], {}, {}, {}
    for name, model in candidate_models().items():
        t0 = time.perf_counter()
        pipe = build_pipeline(model, feature_set)
        p_oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        fold_auc = []
        for _, va in cv.split(X_tr, y_tr):
            fold_auc.append(score_metrics(y_tr.iloc[va], p_oof[va])["roc_auc"])
        m = score_metrics(y_tr, p_oof)
        pipe.fit(X_tr, y_tr)
        p_te = pipe.predict_proba(X_te)[:, 1]
        mt = score_metrics(y_te, p_te)
        rows.append(
            {
                "model": name,
                "cv_roc_auc": m["roc_auc"],
                "cv_roc_auc_std": round(float(np.std(fold_auc)), 4),
                "cv_gini": m["gini"],
                "cv_ks": m["ks"],
                "cv_pr_auc": m["pr_auc"],
                "cv_brier": m["brier"],
                "test_roc_auc": mt["roc_auc"],
                "train_seconds": round(time.perf_counter() - t0, 1),
            }
        )
        oof[name], test_probas[name], fitted[name] = p_oof, p_te, pipe
        log.info("%-20s CV AUC=%.4f  test AUC=%.4f", name, m["roc_auc"], mt["roc_auc"])

    comp = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    comp.to_csv(config.REPORTS_DIR / f"model_comparison_{tag}.csv", index=False)
    best = comp.iloc[0]["model"]
    log.info("Modèle retenu : %s", best)

    # 2) Seuil de décision choisi sur les prédictions out-of-fold (pas sur le test)
    thr, _ = best_threshold(y_tr, oof[best], config.COST_FALSE_NEGATIVE, config.COST_FALSE_POSITIVE)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    costs = [
        expected_cost(y_tr, oof[best], t, config.COST_FALSE_NEGATIVE, config.COST_FALSE_POSITIVE)
        for t in grid
    ]

    # 3) Évaluation finale sur le jeu de test, une seule fois
    model = fitted[best]
    p_te = test_probas[best]
    test_scores = score_metrics(y_te, p_te)
    test_decision = decision_metrics(y_te.to_numpy(), p_te, thr)
    deciles = decile_table(y_te, p_te)
    deciles.to_csv(config.REPORTS_DIR / f"deciles_{tag}.csv", index=False)

    imp = permutation_importance(
        model, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    importance = (
        pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean, "std": imp.importances_std})
        .sort_values("importance", ascending=False)
        .round(4)
    )
    importance.to_csv(config.REPORTS_DIR / f"feature_importance_{tag}.csv", index=False)

    # 3 bis) Explicabilité : SHAP global (test) + un dossier refusé en exemple
    shap_summary = None
    if best in ("lightgbm", "xgboost"):
        _, cat = config.FEATURE_SETS[feature_set]
        sv, base = shap_values(model, X_te, cat)
        mean_abs = sv.abs().mean().sort_values(ascending=False)
        shap_summary = mean_abs.round(4).to_dict()
        mean_abs.round(4).rename("mean_abs_shap").to_csv(config.REPORTS_DIR / f"shap_global_{tag}.csv")
        plots.shap_global(list(sv.columns), sv.abs().mean().to_numpy(), config.FIGURES_DIR / f"{tag}_shap_global.png")
        # dossier refusé « typique » : PD la plus proche de 0,6
        refused = np.where(p_te >= thr)[0]
        i = refused[np.argmin(np.abs(p_te[refused] - 0.6))]
        plots.shap_individual(
            X_te.iloc[i].to_dict(), sv.iloc[i].to_dict(),
            float(1 / (1 + np.exp(-base[i]))), float(p_te[i]),
            config.FIGURES_DIR / f"{tag}_shap_example.png",
        )

    # 4) Figures
    fd = config.FIGURES_DIR
    plots.comparison_bar(comp, fd / f"{tag}_model_comparison.png")
    plots.roc_pr_curves(y_te, test_probas, fd / f"{tag}_roc_pr.png")
    plots.calibration_plot(y_te, p_te, fd / f"{tag}_calibration.png")
    plots.importance_plot(
        list(importance["feature"]), list(importance["importance"]), list(importance["std"]),
        fd / f"{tag}_feature_importance.png",
    )
    plots.cost_curve(grid, costs, thr, fd / f"{tag}_threshold_cost.png")
    if feature_set == "full":
        plots.default_rate_by(df, "loan_grade", config.TARGET, fd / "eda_default_by_grade.png",
                              "Taux de défaut par note interne (loan_grade)")
        plots.default_rate_by(df, "loan_intent", config.TARGET, fd / "eda_default_by_intent.png",
                              "Taux de défaut par objet du prêt")

    # 5) Export de l'artefact + métadonnées (traçabilité / audit)
    model_path = config.MODELS_DIR / f"model_{tag}.joblib"
    joblib.dump(model, model_path)
    num, cat = config.FEATURE_SETS[feature_set]
    meta = {
        "project": "credit-default-classifier",
        "code_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_set": feature_set,
        "numeric_features": num,
        "categorical_features": cat,
        "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in cat},
        "model": best,
        "threshold": thr,
        "costs": {"false_negative": config.COST_FALSE_NEGATIVE, "false_positive": config.COST_FALSE_POSITIVE},
        "data_sha256": sha256_of(config.DATA_RAW),
        "model_sha256": sha256_of(model_path),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "test_metrics": test_scores,
    }
    (config.MODELS_DIR / f"model_{tag}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    report = {
        "feature_set": feature_set,
        "cleaning": cleaning,
        "selected_model": best,
        "comparison": comp.to_dict(orient="records"),
        "test_metrics": test_scores,
        "test_decision": test_decision,
        "deciles": deciles.to_dict(orient="records"),
        "top_features": importance.head(10).to_dict(orient="records"),
        "shap_mean_abs": shap_summary,
    }
    (config.REPORTS_DIR / f"metrics_{tag}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("Test : %s | décision : %s", test_scores, test_decision)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-set", choices=list(config.FEATURE_SETS), default=config.DEFAULT_FEATURE_SET)
    run(ap.parse_args().feature_set)


if __name__ == "__main__":
    main()
