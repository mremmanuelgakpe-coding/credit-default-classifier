"""Démo interactive du score de défaut de crédit (Streamlit).

La démo charge directement le modèle (pas d'appel à l'API) : aucune clé à exposer,
aucune donnée saisie n'est enregistrée. Lancer :  streamlit run demo/app.py
"""
from __future__ import annotations

import hashlib
import html
import json
import sys
from datetime import date
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from credit_default.explain import LABELS_FR, shap_values  # noqa: E402

GITHUB_URL = "https://github.com/mremmanuelgakpe-coding/credit-default-classifier"
ORANGE, ORANGE_2, CREAM, INK, MUTED = "#E85A0C", "#F5AE86", "#FBE3CC", "#171717", "#6B6A66"
SLATE = "#2F6CA8"  # facteurs atténuants : teinte froide opposée à l'orange
BANDS = [("A", 0.00, 0.05), ("B", 0.05, 0.15), ("C", 0.15, 0.30), ("D", 0.30, 0.50), ("E", 0.50, 1.00)]
BAND_COLORS = ["#FBE3CC", "#F8C7A4", "#F5AE86", "#EE7F45", "#E85A0C"]

st.set_page_config(page_title="Scoring crédit", page_icon="📊", layout="wide")


# ---------- Chargement sécurisé ----------
@st.cache_resource
def load(feature_set: str):
    path = ROOT / "models" / f"model_{feature_set}.joblib"
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["model_sha256"]:
        st.error("Le fichier du modèle ne correspond pas à son empreinte. Chargement interrompu.")
        st.stop()
    report = json.loads((ROOT / "reports" / f"metrics_{feature_set}.json").read_text())
    return joblib.load(path), meta, report


# ---------- Style ----------
st.markdown(
    """
<style>
@font-face { font-family:'Montserrat'; src:url('app/static/fonts/montserrat-latin-wght-normal.woff2') format('woff2');
             font-weight:100 900; font-style:normal; font-display:swap; }
@font-face { font-family:'Montserrat'; src:url('app/static/fonts/montserrat-latin-wght-italic.woff2') format('woff2');
             font-weight:100 900; font-style:italic; font-display:swap; }
:root { --page:#E4E3E1; --panel:#EEEDEB; --card:#F9F9F8; --ink:#171717; --muted:#6B6A66;
        --orange:#E85A0C; --orange2:#F5AE86; --cream:#FBE3CC; --black:#151515; --line:#E2E1DE; }
.stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button { font-family: 'Montserrat', system-ui, sans-serif; }
[data-testid="stTooltipIcon"] button { background:transparent !important; color:var(--muted) !important; }
.stApp { background: var(--page); color: var(--ink); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1240px; padding: 1.4rem 1.6rem 3rem; background: var(--panel);
                   border-radius: 30px; margin-top: 1.2rem; margin-bottom: 1.5rem; }
h1, h2, h3 { font-family: 'Montserrat', sans-serif !important; color: var(--ink); letter-spacing: -0.02em; }
.num { font-variant-numeric: tabular-nums; }

/* barre du haut */
.topbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; }
.brand { display:flex; align-items:center; gap:.5rem; font-weight:700; font-size:1.15rem; letter-spacing:-0.03em; }
.brand-mark { width:26px; height:26px; border-radius:8px; background:var(--ink); position:relative; }
.brand-mark::after { content:""; position:absolute; right:-4px; bottom:-4px; width:12px; height:12px;
                     border-radius:50%; background:var(--orange); border:3px solid var(--panel); }
.author { color:var(--muted); font-size:.875rem; }
.author a { color:var(--ink); font-weight:600; text-decoration:none; border-bottom:1.5px solid var(--orange); }
.title-row { display:flex; align-items:center; gap:.75rem; margin:1.1rem 0 .3rem; flex-wrap:wrap; }
.title-row h1 { font-size:1.9rem; font-weight:500; margin:0; padding:0; }
.chip { background:var(--card); border-radius:999px; padding:.25rem .7rem; font-size:.875rem; color:var(--muted); }
.lead { color:var(--muted); font-size:.92rem; max-width:70ch; margin-bottom:.6rem; }

/* onglets en pilules */
.stTabs [role="tablist"] { gap:.25rem; background:var(--card); border-radius:999px; padding:.3rem;
                           width:fit-content; max-width:100%; overflow-x:auto; border:none !important; box-shadow:none !important; }
.stTabs [role="tab"] { border-radius:999px; padding:.4rem 1rem; color:var(--muted); border:none !important; }
.stTabs [role="tab"] p { font-size:.875rem; font-weight:500; }
.stTabs [role="tab"][data-selected="true"] { background:var(--panel); }
.stTabs [role="tab"][data-selected="true"] p { color:var(--orange); font-weight:600; }
.stTabs [role="tab"]:focus-visible { outline:3px solid var(--orange); outline-offset:1px; }
.stTabs .react-aria-SelectionIndicator { display:none !important; }

/* cartes */
.card { background:var(--card); border-radius:22px; padding:1.1rem 1.2rem; height:100%; }
.card .label { font-size:.95rem; color:var(--ink); font-weight:500; margin-top:.9rem; }
.card .value { font-size:1.9rem; font-weight:600; letter-spacing:-0.03em; margin-top:.15rem; }
.card .sub { font-size:.875rem; color:var(--muted); margin-top:.2rem; }
.icon { width:26px; height:26px; border-radius:7px; background:var(--ink); color:#fff; display:inline-flex;
        align-items:center; justify-content:center; font-size:.875rem; font-weight:700; }
.hero { background: radial-gradient(120% 120% at 100% 0%, #FF8A3D 0%, #E85A0C 38%, #7A2A06 75%, #1E1209 100%);
        color:#fff; border-radius:22px; padding:1.1rem 1.2rem; height:100%; position:relative; overflow:hidden; }
.hero .label { font-size:.95rem; margin-top:.9rem; color:#FFE9DA; }
.hero .value { font-size:2.6rem; font-weight:600; letter-spacing:-0.04em; line-height:1.1; }
.hero .icon { background:#fff; color:var(--ink); }
.badge { position:absolute; top:1rem; right:1rem; background:rgba(255,255,255,.18); border-radius:999px;
         padding:.15rem .55rem; font-size:.875rem; }
.badge.light { background:var(--panel); color:var(--muted); }
.decision { display:inline-block; border-radius:999px; padding:.2rem .8rem; font-weight:600; font-size:1.1rem; margin-top:.3rem; }
.decision.ok { background:#E6EFE6; color:#1F6B2A; }
.decision.ko { background:var(--cream); color:#B2410A; }

/* échelle de risque */
.scale { display:flex; gap:6px; margin-top:.8rem; }
.seg { flex:1; height:46px; border-radius:12px; position:relative; }
.seg.on { outline:3px solid var(--ink); outline-offset:2px; }
.seg span { position:absolute; bottom:-1.6rem; left:0; right:0; text-align:center; font-size:.875rem; color:var(--muted); }
.scale-top { display:flex; justify-content:space-between; font-size:.875rem; color:var(--muted); }
.scale-top b { color:var(--ink); font-size:1.4rem; font-weight:600; }

/* carte noire des motifs */
.dark { background:var(--black); color:#F3F2EF; border-radius:24px; padding:1.3rem 1.3rem 1rem; }
.dark h3 { color:#fff; font-size:1.15rem; font-weight:500; margin:0 0 .2rem; }
.dark .hint { color:#9C9B96; font-size:.875rem; margin-bottom:.8rem; }
.dark .count { float:right; background:#B07A1E; color:#1a1206; border-radius:999px; padding:.05rem .55rem; font-size:.875rem; font-weight:700; }
.reason { display:flex; justify-content:space-between; align-items:center; gap:.8rem; padding:.55rem 0;
          border-top:1px solid #2A2A28; }
.reason .t { font-size:.875rem; color:#F3F2EF; }
.reason .v { font-size:.875rem; color:#9C9B96; }
.dot { width:24px; height:24px; flex:0 0 24px; border-radius:50%; display:flex; align-items:center;
       justify-content:center; font-size:.875rem; font-weight:700; }
.dot.up { background:var(--orange); color:#fff; }
.dot.down { background:#2F6CA8; color:#fff; }
.group { font-size:.875rem; color:#9C9B96; margin:.7rem 0 .1rem; }

/* conteneurs Streamlit utilisés comme cartes */
.st-key-form, .st-key-chart, .st-key-perf1, .st-key-perf2, .st-key-perf3 {
  background:var(--card); border-radius:22px; padding:1.1rem 1.2rem 1.2rem; }
.st-key-chart { margin-top:12px; }
.hero, .card { margin-bottom:12px; }
.st-key-form h3, .st-key-chart h3, [class*="st-key-perf"] h3 { font-size:1.1rem; font-weight:500; margin:0; }
div[data-baseweb="select"] > div, .stNumberInput input { border-radius:12px !important; background:#fff !important; }
.stButton button { border-radius:999px; border:none; background:var(--panel); color:var(--ink); font-weight:600; font-size:.875rem; }
.stButton button:hover { background:var(--cream); color:#B2410A; }
.stButton button:focus-visible { outline:3px solid var(--orange); outline-offset:2px; }

/* contributions SHAP */
.slegend { display:flex; gap:1.2rem; font-size:.875rem; color:var(--muted); margin:.4rem 0 .6rem; }
.slegend i { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:.4rem; vertical-align:-1px; }
.srow { display:grid; grid-template-columns: minmax(180px, 1.1fr) 2fr 56px; gap:.9rem; align-items:center;
        padding:.45rem 0; border-top:1px solid var(--line); }
.slab { font-size:.875rem; color:var(--ink); }
.slab span { display:block; font-size:.875rem; color:var(--muted); }
.strack { position:relative; height:14px; }
.smid { position:absolute; left:50%; top:-6px; bottom:-6px; width:1.5px; background:#CFCDC8; }
.sbar { position:absolute; top:0; height:14px; border-radius:4px; min-width:2px; }
.sval { font-size:.875rem; font-weight:600; text-align:right; }

/* grille des déciles */
.grid { display:grid; grid-template-columns: repeat(10, 1fr); gap:6px; margin-top:.8rem; }
.dcol { display:flex; flex-direction:column; align-items:center; gap:4px; }
.dcol .cell { width:100%; }
.cell { height:40px; border-radius:10px; display:flex; align-items:center; justify-content:center;
        font-size:.875rem; font-weight:600; }
.rowlab { font-size:.875rem; color:var(--muted); text-align:center; }
.foot { color:var(--muted); font-size:.875rem; margin-top:1.4rem; }
@media (max-width: 640px) {
  .block-container { padding:1rem .9rem 2rem; border-radius:22px; }
  .hero .value { font-size:2.1rem; } .grid { grid-template-columns: repeat(5, 1fr); gap:8px; }
  .srow { grid-template-columns: 1fr 48px; gap:.3rem .6rem; }
  .srow .strack { grid-column: 1 / -1; grid-row: 2; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Profils pré-remplis ----------
PRESETS = {
    "Profil solide": dict(age=41, income=95000, home="MORTGAGE", emp=12.0, intent="HOMEIMPROVEMENT",
                          grade="A", amount=8000, rate=7.5, prior="N", hist=14),
    "Profil limite": dict(age=27, income=42000, home="RENT", emp=3.0, intent="PERSONAL",
                          grade="C", amount=9000, rate=13.5, prior="N", hist=4),
    "Profil à risque": dict(age=23, income=21000, home="RENT", emp=1.0, intent="MEDICAL",
                            grade="D", amount=8000, rate=15.9, prior="Y", hist=2),
}
for k, v in PRESETS["Profil limite"].items():
    st.session_state.setdefault(k, v)


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


HOME = {"RENT": "Locataire", "MORTGAGE": "Propriétaire avec crédit", "OWN": "Propriétaire", "OTHER": "Autre"}
INTENT = {"EDUCATION": "Études", "MEDICAL": "Frais médicaux", "VENTURE": "Création d'entreprise",
          "PERSONAL": "Usage personnel", "DEBTCONSOLIDATION": "Rachat de crédits",
          "HOMEIMPROVEMENT": "Travaux"}
VALUE_FR = {**HOME, **INTENT, "Y": "Oui", "N": "Non"}


def fr(x: float, nd: int = 1) -> str:
    """Nombre au format français (virgule décimale)."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x: float, nd: int = 1) -> str:
    return f"{fr(100 * x, nd)} %"


def fmt_value(feat, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "non renseigné"
    if feat in ("person_income", "loan_amnt"):
        return f"{v:,.0f} $".replace(",", " ")
    if feat == "loan_percent_income":
        return pct(v, 0)
    if feat == "loan_int_rate":
        return f"{fr(v)} %"
    if feat in ("person_age", "person_emp_length", "cb_person_cred_hist_length"):
        return f"{v:.0f} an" if v < 2 else f"{v:.0f} ans"
    return VALUE_FR.get(v, str(v))


# ---------- En-tête ----------
st.markdown(
    f"""
<div class="topbar">
  <div class="brand"><div class="brand-mark"></div>Scoring crédit</div>
  <div class="author">Projet portfolio de <b>Emmanuel</b> · <a href="{GITHUB_URL}" target="_blank" rel="noopener">voir le code</a></div>
</div>
<div class="title-row"><h1>Évaluer une demande de prêt</h1><span class="chip">{date.today().strftime('%d/%m/%Y')}</span></div>
<div class="lead">Modifiez le dossier : le modèle recalcule la probabilité de défaut, la décision et les motifs à chaque changement.
Modèle LightGBM entraîné sur 32 411 prêts publics (Kaggle).</div>
""",
    unsafe_allow_html=True,
)

tab_sim, tab_perf, tab_method = st.tabs(["Simulateur", "Performance du modèle", "Méthode"])

# =====================================================================
with tab_sim:
    left, right = st.columns([0.95, 1.35], gap="medium")

    with left:
        with st.container(key="form"):
            st.markdown("<h3>Dossier client</h3>", unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            for col, name in zip((b1, b2, b3), PRESETS):
                col.button(name, on_click=apply_preset, args=(name,), use_container_width=True)
            with_grade = st.toggle(
                "Utiliser la note interne et le taux du prêteur", value=True)
            st.caption("Désactivé : score d'octroi, basé uniquement sur le client et sa demande." if not with_grade
                       else "Activé : le modèle tient compte de la note A à G et du taux fixés par le prêteur.")
            c1, c2 = st.columns(2)
            c1.number_input("Âge", 18, 100, key="age", step=1)
            c2.number_input("Revenu annuel ($)", 1000, 2_000_000, key="income", step=1000)
            c1.selectbox("Logement", list(HOME), format_func=HOME.get, key="home")
            c2.number_input("Ancienneté pro. (ans)", 0.0, 60.0, key="emp", step=1.0, format="%.0f")
            st.selectbox("Objet du prêt", list(INTENT), format_func=INTENT.get, key="intent")
            c3, c4 = st.columns(2)
            c3.number_input("Montant demandé ($)", 500, 35_000, key="amount", step=500)
            c4.number_input("Historique de crédit (ans)", 0, 60, key="hist", step=1)
            if with_grade:
                c5, c6 = st.columns(2)
                c5.selectbox("Note interne", list("ABCDEFG"), key="grade")
                c6.number_input("Taux d'intérêt (%)", 5.0, 25.0, key="rate", step=0.1, format="%.1f")
            st.radio("Défaut de paiement par le passé", ["N", "Y"], format_func=VALUE_FR.get,
                     key="prior", horizontal=True)

    s = st.session_state
    feature_set = "full" if with_grade else "application"
    model, meta, report = load(feature_set)
    pct_income = min(s.amount / s.income, 5.0)
    row = {
        "person_age": s.age, "person_income": float(s.income), "person_home_ownership": s.home,
        "person_emp_length": float(s.emp), "loan_intent": s.intent, "loan_amnt": float(s.amount),
        "loan_percent_income": round(pct_income, 2), "cb_person_default_on_file": s.prior,
        "cb_person_cred_hist_length": s.hist, "loan_grade": s.get("grade", "C"),
        "loan_int_rate": float(s.get("rate", 13.5)),
    }
    cols = meta["numeric_features"] + meta["categorical_features"]
    X = pd.DataFrame([row])[cols]
    p = float(model.predict_proba(X)[:, 1][0])
    sv, base = shap_values(model, X, meta["categorical_features"])
    contrib = sv.iloc[0]
    thr = meta["threshold"]
    refused = p >= thr
    band_i = next(i for i, (_, lo, hi) in enumerate(BANDS) if lo <= p < hi or (i == 4 and p >= lo))

    with right:
        k1, k2 = st.columns(2, gap="small")
        k1.markdown(
            f"""<div class="hero"><span class="badge">seuil {pct(thr, 0)}</span><span class="icon">%</span>
            <div class="label">Probabilité de défaut</div><div class="value num">{pct(p)}</div></div>""",
            unsafe_allow_html=True,
        )
        cls = "ko" if refused else "ok"
        word = "Refus" if refused else "Accord"
        k2.markdown(
            f"""<div class="card"><span class="badge light">{pct(pct_income, 0)} du revenu</span><span class="icon">✓</span>
            <div class="label">Décision du modèle</div><div class="decision {cls}">{word}</div>
            <div class="sub">Refus dès {pct(thr, 0)} de risque : un défaut coûte 5 fois plus qu'un bon client refusé.</div></div>""",
            unsafe_allow_html=True,
        )
        segs = "".join(
            f'<div class="seg{" on" if i == band_i else ""}" style="background:{BAND_COLORS[i]}" '
            f'title="Classe {b} : {pct(lo, 0)} à {pct(hi, 0)}"><span>{b}</span></div>'
            for i, (b, lo, hi) in enumerate(BANDS)
        )
        st.markdown(
            f"""<div class="card" style="margin-top:12px; padding-bottom:1.8rem">
            <div class="scale-top"><div><span style="font-size:1rem;color:#171717;font-weight:500">Classe de risque</span>
            <div class="sub">de {pct(BANDS[band_i][1], 0)} à {pct(BANDS[band_i][2], 0)} de probabilité de défaut</div></div>
            <b class="num">{BANDS[band_i][0]}</b></div><div class="scale">{segs}</div></div>""",
            unsafe_allow_html=True,
        )

        # carte noire : motifs
        ranked = contrib.sort_values()
        ups = [f for f in ranked.index[::-1] if ranked[f] > 0][:3]
        downs = [f for f in ranked.index if ranked[f] < 0][:3]

        def line(f, up):
            arrow = "▲" if up else "▼"
            return (f'<div class="reason"><div><div class="t">{html.escape(LABELS_FR[f])}</div>'
                    f'<div class="v">{html.escape(fmt_value(f, row[f]))}</div></div>'
                    f'<div class="dot {"up" if up else "down"}" aria-label="{"augmente" if up else "réduit"} le risque">{arrow}</div></div>')

        body = ""
        if ups:
            body += '<div class="group">Ce qui augmente le risque</div>' + "".join(line(f, True) for f in ups)
        if downs:
            body += '<div class="group">Ce qui le réduit</div>' + "".join(line(f, False) for f in downs)
        st.markdown(
            f"""<div class="dark" style="margin-top:12px"><span class="count">{len(ups) + len(downs)} motifs</span>
            <h3>Motifs de la décision</h3><div class="hint">Calculés pour ce dossier avec les valeurs SHAP du modèle.</div>
            {body}</div>""",
            unsafe_allow_html=True,
        )

    with st.container(key="chart"):
        st.markdown("<h3>Contribution de chaque variable</h3>", unsafe_allow_html=True)
        base_p = 1 / (1 + np.exp(-base[0]))
        st.caption(f"Le modèle part d'un score de base de {pct(base_p, 0)}. Chaque barre montre de combien la variable "
                   "pousse le risque vers le haut ou vers le bas (contribution SHAP, en log-odds).")
        order_f = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
        vmax = max(float(order_f.abs().max()), 1e-9)
        rows_html = ""
        for f, v in order_f.items():
            w = abs(v) / vmax * 50  # demi-piste = 50 %
            side = f"left:50%;width:{w:.1f}%" if v > 0 else f"left:{50 - w:.1f}%;width:{w:.1f}%"
            color = ORANGE if v > 0 else SLATE
            sens = "augmente le risque" if v > 0 else "réduit le risque"
            rows_html += (
                f'<div class="srow" title="{html.escape(LABELS_FR[f])} {sens} de {abs(v):.2f} (log-odds)">'
                f'<div class="slab">{html.escape(LABELS_FR[f])}<span>{html.escape(fmt_value(f, row[f]))}</span></div>'
                f'<div class="strack"><div class="smid"></div><div class="sbar" style="{side};background:{color}"></div></div>'
                f'<div class="sval num" style="color:{color}">{"+" if v > 0 else "−"}{fr(abs(v), 2)}</div></div>'
            )
        st.markdown(
            '<div class="slegend"><span><i style="background:%s"></i>Augmente le risque</span>'
            '<span><i style="background:%s"></i>Réduit le risque</span></div>%s' % (ORANGE, SLATE, rows_html),
            unsafe_allow_html=True,
        )

# =====================================================================
with tab_perf:
    _, meta_f, rep = load("full")
    _, _, rep_a = load("application")
    tm, td = rep["test_metrics"], rep["test_decision"]
    base_rate = rep["cleaning"]["default_rate"]
    st.markdown("<div class='lead' style='margin-top:.6rem'>Mesuré sur 6 483 dossiers mis de côté avant "
                "l'entraînement et jamais vus par le modèle.</div>", unsafe_allow_html=True)
    k = st.columns(4, gap="small")
    cards = [
        ("hero", "AUC sur le test", fr(tm['roc_auc'], 3), f"Gini {fr(tm['gini'], 2)}, KS {fr(tm['ks'], 2)}"),
        ("card", "Défauts détectés", pct(td['recall_default'], 0), f"au seuil de {pct(td['threshold'], 0)}"),
        ("card", "Défaut des dossiers acceptés", pct(td['default_rate_accepted']), f"contre {pct(base_rate)} sans modèle"),
        ("card", "Taux de refus", pct(td['rejection_rate'], 0), "des demandes"),
    ]
    for col, (kind, lab, val, sub) in zip(k, cards):
        sub_style = "color:#FFE9DA" if kind == "hero" else ""
        col.markdown(f"""<div class="{kind}"><div class="label" style="margin-top:.2rem">{lab}</div>
        <div class="value num">{val}</div><div class="sub" style="{sub_style}">{sub}</div></div>""",
                     unsafe_allow_html=True)

    st.write("")
    a, b = st.columns([1.1, 1], gap="medium")
    with a:
        with st.container(key="perf1"):
            st.markdown("<h3>Six modèles comparés</h3>", unsafe_allow_html=True)
            st.caption("AUC moyenne en validation croisée sur 5 plis. LightGBM est retenu.")
            names = {"logistic_regression": "Régression logistique", "knn": "KNN", "naive_bayes": "Naive Bayes",
                     "random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
            comp = pd.DataFrame(rep["comparison"])
            comp["Modèle"] = comp["model"].map(names)
            order = list(comp.sort_values("cv_roc_auc", ascending=False)["Modèle"])
            comp["auc_fr"] = comp["cv_roc_auc"].map(lambda v: fr(v, 3))
            comp["debut"] = 0.5
            comp["retenu"] = np.where(comp["model"] == rep["selected_model"], "Retenu", "Comparé")
            bar = (
                alt.Chart(comp)
                .mark_bar(cornerRadiusEnd=4, height=18)
                .encode(
                    x=alt.X("cv_roc_auc:Q", title="AUC (validation croisée)", scale=alt.Scale(domain=[0.5, 1])),
                    x2="debut:Q",
                    y=alt.Y("Modèle:N", sort=order, title=None,
                            axis=alt.Axis(labelColor=INK, labelFontSize=14, labelLimit=200, ticks=False)),
                    color=alt.Color("retenu:N", scale=alt.Scale(domain=["Retenu", "Comparé"], range=[ORANGE, "#C9C7C2"]),
                                    legend=None),
                    tooltip=[alt.Tooltip("Modèle:N"), alt.Tooltip("cv_roc_auc:Q", title="AUC", format=".3f"),
                             alt.Tooltip("cv_gini:Q", title="Gini", format=".3f"),
                             alt.Tooltip("test_roc_auc:Q", title="AUC test", format=".3f")],
                )
            )
            labels = bar.mark_text(align="left", dx=5, fontSize=14, color=INK).encode(
                text="auc_fr:N", color=alt.value(INK))
            st.altair_chart((bar + labels).properties(height=260, padding={"left": 5, "right": 30, "top": 5, "bottom": 5}).configure_view(strokeWidth=0)
                            .configure(background="transparent", font="Montserrat, system-ui, sans-serif")
                            .configure_axis(domainColor="#D6D5D1", tickColor="#D6D5D1", titleColor=MUTED,
                                            labelColor=MUTED, gridColor="#ECEBE8", labelFontSize=14,
                                            titleFontSize=14, titleFontWeight=500),
                            use_container_width=True)
    with b:
        with st.container(key="perf2"):
            st.markdown("<h3>Taux de défaut par décile de score</h3>", unsafe_allow_html=True)
            st.caption("Taux de défaut observé (en %) des dossiers du test, des plus risqués (1) aux moins risqués (10).")
            dec = pd.DataFrame(rep["deciles"])

            def shade(v):
                ramp = ["#F2F1EE", "#FBE3CC", "#F8C7A4", "#F5AE86", "#EE7F45", "#E85A0C"]
                i = min(int(v * len(ramp)), len(ramp) - 1)
                return ramp[i], ("#fff" if i >= 4 else INK)

            cells = ""
            for d, v in zip(dec["decile"], dec["taux_defaut_observe"]):
                bg, fg = shade(v)
                cells += (f'<div class="dcol"><div class="rowlab">{int(d)}</div>'
                          f'<div class="cell num" style="background:{bg};color:{fg}" title="Décile {int(d)} : {pct(v)}">'
                          f'{fr(100 * v, 0)}</div></div>')
            st.markdown(f'<div class="grid">{cells}</div>', unsafe_allow_html=True)
            top2 = dec["capture_cumulee_defauts"].iloc[1]
            st.markdown(f"<div class='sub' style='color:#6B6A66;font-size:.875rem;margin-top:1rem'>Les 20 % de "
                        f"dossiers les plus risqués concentrent <b style='color:#171717'>{pct(top2, 0)}</b> des défauts.</div>",
                        unsafe_allow_html=True)

    with st.container(key="perf3"):
        st.markdown("<h3>Avec ou sans la note interne du prêteur</h3>", unsafe_allow_html=True)
        st.caption("La note et le taux sont fixés par le prêteur après sa propre analyse. Sans eux, le modèle "
                   "reste très discriminant : il ne se contente pas de recopier la notation existante.")
        tbl = pd.DataFrame([
            {"Variables": "Complètes (avec note et taux)", "AUC": rep["test_metrics"]["roc_auc"],
             "Gini": rep["test_metrics"]["gini"],
             "Défaut des acceptés": pct(rep["test_decision"]["default_rate_accepted"])},
            {"Variables": "Octroi (client et demande seulement)", "AUC": rep_a["test_metrics"]["roc_auc"],
             "Gini": rep_a["test_metrics"]["gini"],
             "Défaut des acceptés": pct(rep_a["test_decision"]["default_rate_accepted"])},
        ])
        tbl["AUC"] = tbl["AUC"].map(lambda v: fr(v, 3))
        tbl["Gini"] = tbl["Gini"].map(lambda v: fr(v, 3))
        st.dataframe(tbl, hide_index=True, use_container_width=True)

# =====================================================================
with tab_method:
    st.markdown(
        f"""
<div class="card" style="margin-top:.6rem">
<h3 style="font-size:1.15rem;font-weight:500">Comment le modèle a été construit</h3>
<div style="max-width:72ch; color:#3B3A37; font-size:.93rem; line-height:1.65">
<p><b>Données.</b> 32 581 prêts publics (Kaggle, <i>Credit Risk Dataset</i>), 21,8 % de défauts.
Nettoyage documenté : 165 doublons et 5 âges impossibles supprimés, valeurs manquantes imputées dans le pipeline.</p>
<p><b>Comparaison.</b> Régression logistique, KNN, Naive Bayes, Random Forest, XGBoost et LightGBM, évalués en
validation croisée sur 80 % des données. Les 20 % restants ne servent qu'à la mesure finale.</p>
<p><b>Décision.</b> Le seuil de refus minimise un coût métier (un défaut non détecté coûte 5, un bon client refusé
coûte 1). Il est choisi hors du jeu de test.</p>
<p><b>Explications.</b> Chaque score est décomposé en valeurs SHAP exactes (TreeSHAP de LightGBM), regroupées sur les
variables d'origine pour être lisibles par un analyste crédit.</p>
<p><b>Mise en production.</b> Le même modèle est servi par une API FastAPI : clé API, validation stricte des entrées,
contrôle d'intégrité du fichier modèle, conteneur Docker sans droits administrateur, 17 tests automatisés.</p>
<p><b>Limites.</b> Données publiques sans dates, donc pas de validation dans le temps. Les classes A à E sont
illustratives. SHAP explique le modèle, pas une causalité.</p>
<p><a href="{GITHUB_URL}" target="_blank" rel="noopener" style="color:#171717;font-weight:600;border-bottom:1.5px solid #E85A0C;text-decoration:none">Lire le code et le rapport complet sur GitHub</a></p>
</div></div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<div class='foot'>Démonstration pédagogique sur données publiques. Aucune donnée saisie n'est "
            "enregistrée. Ne constitue pas une décision de crédit réelle.</div>", unsafe_allow_html=True)
