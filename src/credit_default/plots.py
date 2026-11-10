"""Figures statiques (PNG) pour le README et le rapport."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

# Palette catégorielle validée (ordre fixe, jamais cyclée)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODEL_ORDER = ["logistic_regression", "knn", "naive_bayes", "random_forest", "xgboost", "lightgbm"]
COLOR = dict(zip(MODEL_ORDER, PALETTE))
LABEL = {
    "logistic_regression": "Régression logistique",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def roc_pr_curves(y, probas: dict, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, lw=2, color=COLOR[name], label=LABEL[name])
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, lw=2, color=COLOR[name], label=LABEL[name])
    a1.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2)
    a2.axhline(y.mean(), ls="--", lw=1, color=INK2)
    _style(a1, "Courbes ROC (jeu de test)")
    _style(a2, "Précision-Rappel (jeu de test)")
    a1.set_xlabel("Taux de faux positifs")
    a1.set_ylabel("Taux de vrais positifs")
    a2.set_xlabel("Rappel (défauts détectés)")
    a2.set_ylabel("Précision")
    a2.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def comparison_bar(df, path):
    """AUC moyenne en validation croisée, avec écart-type."""
    df = df.sort_values("cv_roc_auc")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh(
        [LABEL[m] for m in df["model"]],
        df["cv_roc_auc"],
        xerr=df["cv_roc_auc_std"],
        color=[COLOR[m] for m in df["model"]],
        height=0.6,
        error_kw={"ecolor": INK2, "lw": 1},
    )
    for i, v in enumerate(df["cv_roc_auc"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("ROC AUC (validation croisée 5 plis)")
    _style(ax, "Comparaison des classifieurs")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def calibration_plot(y, p, path):
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2, label="Calibration parfaite")
    ax.plot(mean, frac, marker="o", ms=6, lw=2, color=PALETTE[0], label="Modèle retenu")
    ax.set_xlabel("Probabilité de défaut prédite")
    ax.set_ylabel("Taux de défaut observé")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    _style(ax, "Calibration (déciles de score)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def importance_plot(names, values, stds, path, top=12):
    order = np.argsort(values)[-top:]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.barh(
        [names[i] for i in order],
        [values[i] for i in order],
        xerr=[stds[i] for i in order],
        color=PALETTE[0],
        height=0.6,
        error_kw={"ecolor": INK2, "lw": 1},
    )
    ax.set_xlabel("Baisse d'AUC quand la variable est permutée")
    _style(ax, "Importance des variables (permutation, test)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cost_curve(thresholds, costs, best, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(thresholds, costs, lw=2, color=PALETTE[0])
    ax.axvline(best, ls="--", lw=1, color=INK2)
    ax.text(best + 0.01, max(costs) * 0.95, f"seuil retenu = {best:.2f}", color=INK, fontsize=9)
    ax.set_xlabel("Seuil de refus (probabilité de défaut)")
    ax.set_ylabel("Coût moyen par dossier")
    _style(ax, "Choix du seuil par coût métier (hors test)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def default_rate_by(df, col, target, path, title):
    g = df.groupby(col)[target].mean().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(g.index.astype(str), g.values, color=PALETTE[0], width=0.6)
    for i, v in enumerate(g.values):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9, color=INK)
    ax.set_ylabel("Taux de défaut")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(ax, title)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def shap_global(names, mean_abs, path, top=11):
    from .explain import LABELS_FR

    order = np.argsort(mean_abs)[-top:]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.barh([LABELS_FR.get(names[i], names[i]) for i in order], [mean_abs[i] for i in order],
            color=PALETTE[0], height=0.6)
    ax.set_xlabel("|SHAP| moyen (log-odds)")
    _style(ax, "Contribution moyenne au score (SHAP, test)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def shap_individual(row_values: dict, contribs: dict, base_proba: float, proba: float, path):
    """Motifs d'un dossier : barres signées, rouge = aggrave le risque, bleu = l'atténue."""
    from .explain import LABELS_FR

    items = sorted(contribs.items(), key=lambda kv: abs(kv[1]))
    labels = [f"{LABELS_FR.get(k, k)} = {_fmt(row_values[k])}" for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#d03b3b" if v > 0 else PALETTE[0] for v in vals]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(labels, vals, color=colors, height=0.6)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("Contribution SHAP (log-odds) — à droite : augmente le risque")
    _style(ax, f"Exemple de dossier refusé : PD {proba:.0%} (score de base {base_proba:.0%})")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if float(v).is_integer():
            return f"{v:,.0f}".replace(",", " ")
        if abs(v) < 1:
            return f"{v:.2f}"
        return f"{v:.1f}" if abs(v) < 100 else f"{v:,.0f}".replace(",", " ")
    return str(v)
