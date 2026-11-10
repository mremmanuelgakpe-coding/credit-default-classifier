"""Test de fumée de la démo Streamlit : l'app s'exécute sans erreur sur les trois profils et sans la note interne."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "demo" / "app.py")


def run(at):
    at.run(timeout=60)
    assert not at.exception, at.exception
    return at


def test_demo_runs_with_presets():
    at = run(AppTest.from_file(APP, default_timeout=60))
    for i in range(3):  # Profil solide, limite, à risque
        at.button[i].click()
        run(at)
    assert at.session_state["prior"] == "Y"  # profil à risque appliqué


def test_demo_runs_without_internal_grade():
    at = run(AppTest.from_file(APP, default_timeout=60))
    at.toggle[0].set_value(False)
    run(at)
