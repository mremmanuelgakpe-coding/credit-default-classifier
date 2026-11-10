import os

import pytest
from fastapi.testclient import TestClient

os.environ["API_KEY"] = "test-key-not-for-prod"

from api.main import app  # noqa: E402

GOOD = {
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
RISKY = {**GOOD, "person_income": 12000, "loan_amnt": 9000, "loan_percent_income": 0.75,
         "loan_grade": "E", "loan_int_rate": 19.5, "cb_person_default_on_file": "Y"}
H = {"X-API-Key": "test-key-not-for-prod"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_missing_or_wrong_key_rejected(client):
    assert client.post("/predict", json=GOOD).status_code == 401
    assert client.post("/predict", json=GOOD, headers={"X-API-Key": "wrong"}).status_code == 401


def test_predict_ok(client):
    r = client.post("/predict", json=GOOD, headers=H)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability_default"] <= 1
    assert body["decision"] in {"ACCEPT", "REFUSE"}


def test_risky_profile_scores_higher(client):
    p_good = client.post("/predict", json=GOOD, headers=H).json()["probability_default"]
    p_risky = client.post("/predict", json=RISKY, headers=H).json()["probability_default"]
    assert p_risky > p_good


@pytest.mark.parametrize(
    "patch",
    [
        {"person_age": 150},
        {"person_income": -5},
        {"loan_intent": "CASINO"},
        {"unexpected_field": 1},
    ],
)
def test_invalid_inputs_rejected(client, patch):
    assert client.post("/predict", json={**GOOD, **patch}, headers=H).status_code == 422


def test_batch_and_limit(client):
    r = client.post("/predict/batch", json={"applications": [GOOD, RISKY]}, headers=H)
    assert r.status_code == 200 and len(r.json()) == 2
    big = {"applications": [GOOD] * 501}
    assert client.post("/predict/batch", json=big, headers=H).status_code == 413


def test_docs_disabled_by_default(client):
    assert client.get("/docs").status_code == 404


def test_explanation_returned_and_consistent(client):
    body = client.post("/predict", json=RISKY, headers=H).json()
    exp = body["explanation"]
    assert exp is not None and len(exp["risk_factors"]) >= 1
    assert all(f["impact"] > 0 for f in exp["risk_factors"])
    assert all(f["impact"] < 0 for f in exp["protective_factors"])
    # libellés lisibles, variables d'origine (pas de colonnes one-hot)
    assert all("_" not in f["label"] for f in exp["risk_factors"])
    no_exp = client.post("/predict?explain=false", json=RISKY, headers=H).json()
    assert no_exp["explanation"] is None


def test_shap_sums_to_model_logit():
    import numpy as np
    import pandas as pd

    from credit_default.explain import shap_values

    model, meta = app.state.model, app.state.meta
    X = pd.DataFrame([GOOD, RISKY])[meta["numeric_features"] + meta["categorical_features"]]
    sv, base = shap_values(model, X, meta["categorical_features"])
    logit = base + sv.sum(axis=1).to_numpy()
    assert np.allclose(1 / (1 + np.exp(-logit)), model.predict_proba(X)[:, 1], atol=1e-6)
