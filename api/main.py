"""API de scoring de défaut de crédit (FastAPI).

Sécurité :
- authentification par clé API (en-tête X-API-Key, comparaison à temps constant) ;
- validation stricte des entrées (types, bornes, valeurs autorisées, champs inconnus refusés) ;
- vérification d'intégrité SHA-256 de l'artefact avant chargement (joblib = pickle) ;
- aucune donnée client journalisée ; documentation interactive désactivable en production ;
- taille des lots plafonnée.

Explicabilité : chaque décision est accompagnée de ses principaux motifs
(valeurs SHAP exactes), ce qui permet de motiver un refus auprès du client.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from credit_default.explain import reason_codes, shap_values

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model_full.joblib"))
META_PATH = MODEL_PATH.with_suffix(".meta.json")
MAX_BATCH = int(os.getenv("MAX_BATCH", "500"))

log = logging.getLogger("api")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ---------- Schémas ----------
class LoanApplication(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "person_age": 29,
                "person_income": 48000,
                "person_home_ownership": "RENT",
                "person_emp_length": 4,
                "loan_intent": "EDUCATION",
                "loan_grade": "B",
                "loan_amnt": 9000,
                "loan_int_rate": 11.2,
                "loan_percent_income": 0.19,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 5,
            }
        },
    )

    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0, le=10_000_000)
    person_home_ownership: Literal["RENT", "MORTGAGE", "OWN", "OTHER"]
    person_emp_length: float | None = Field(default=None, ge=0, le=60)
    loan_intent: Literal[
        "EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"] | None = None
    loan_amnt: float = Field(gt=0, le=1_000_000)
    loan_int_rate: float | None = Field(default=None, ge=0, le=40)
    loan_percent_income: float = Field(ge=0, le=5)
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(ge=0, le=80)


class Factor(BaseModel):
    feature: str
    label: str
    value: float | str | None
    impact: float = Field(description="Contribution SHAP en log-odds (> 0 : augmente le risque)")


class Explanation(BaseModel):
    risk_factors: list[Factor]
    protective_factors: list[Factor]


class Prediction(BaseModel):
    probability_default: float
    decision: Literal["ACCEPT", "REFUSE"]
    risk_band: Literal["A", "B", "C", "D", "E"]
    threshold: float
    model_version: str
    explanation: Explanation | None = None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    applications: list[LoanApplication] = Field(min_length=1)


# ---------- Chargement sécurisé du modèle ----------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_artifact() -> tuple[object, dict]:
    meta = json.loads(META_PATH.read_text())
    digest = _sha256(MODEL_PATH)
    if not secrets.compare_digest(digest, meta["model_sha256"]):
        raise RuntimeError("Empreinte du modèle invalide : artefact altéré, chargement refusé.")
    return joblib.load(MODEL_PATH), meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.meta = load_artifact()
    log.info("Modèle chargé : %s (%s)", app.state.meta["model"], app.state.meta["feature_set"])
    yield


docs_on = os.getenv("ENABLE_DOCS", "false").lower() == "true"
app = FastAPI(
    title="Credit Default Scoring API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if docs_on else None,
    redoc_url=None,
    openapi_url="/openapi.json" if docs_on else None,
)


def require_api_key(key: str | None = Security(api_key_header)) -> None:
    expected = os.getenv("API_KEY")
    if not expected:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "API_KEY non configurée côté serveur.")
    if not key or not secrets.compare_digest(key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Clé API invalide.")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    return response


# ---------- Logique ----------
BANDS = [(0.05, "A"), (0.15, "B"), (0.30, "C"), (0.50, "D")]


def risk_band(p: float) -> str:
    for cut, band in BANDS:
        if p < cut:
            return band
    return "E"


def _score(apps: list[LoanApplication], explain: bool = False) -> list[Prediction]:
    meta = app.state.meta
    cols = meta["numeric_features"] + meta["categorical_features"]
    rows = [a.model_dump() for a in apps]
    missing = [c for c in ("loan_grade",) if c in cols and any(r[c] is None for r in rows)]
    if missing:
        raise HTTPException(422, f"Champ requis par ce modèle : {', '.join(missing)}")
    X = pd.DataFrame(rows)[cols]
    proba = app.state.model.predict_proba(X)[:, 1]
    thr = meta["threshold"]
    version = f"{meta['model']}-{meta['feature_set']}-{meta['trained_at'][:10]}"
    explanations = [None] * len(rows)
    if explain:
        sv, _ = shap_values(app.state.model, X, meta["categorical_features"])
        explanations = reason_codes(X, sv)
    return [
        Prediction(
            probability_default=round(float(p), 4),
            decision="REFUSE" if p >= thr else "ACCEPT",
            risk_band=risk_band(float(p)),
            threshold=thr,
            model_version=version,
            explanation=e,
        )
        for p, e in zip(proba, explanations)
    ]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "model")}


@app.get("/model-info", dependencies=[Depends(require_api_key)])
def model_info():
    m = app.state.meta
    return {k: m[k] for k in ("model", "feature_set", "threshold", "trained_at", "test_metrics", "n_train")}


@app.post("/predict", response_model=Prediction, dependencies=[Depends(require_api_key)])
def predict(
    application: LoanApplication,
    explain: bool = Query(True, description="Inclure les motifs (valeurs SHAP)"),
):
    return _score([application], explain)[0]


@app.post("/predict/batch", response_model=list[Prediction], dependencies=[Depends(require_api_key)])
def predict_batch(batch: BatchRequest, explain: bool = Query(False)):
    if len(batch.applications) > MAX_BATCH:
        raise HTTPException(413, f"Lot limité à {MAX_BATCH} dossiers.")
    return _score(batch.applications, explain)
