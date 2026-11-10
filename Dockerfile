# ---- Image de production de l'API de scoring ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

# libgomp1 : runtime OpenMP requis par LightGBM / XGBoost
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Code + artefact uniquement (pas de données brutes dans l'image)
COPY api/ api/
COPY src/ src/
COPY models/model_full.joblib models/model_full.meta.json models/

# Utilisateur non-root, fichiers en lecture seule
RUN useradd --create-home --uid 10001 appuser && chown -R root:root /app && chmod -R a-w /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--no-server-header"]
