#!/bin/bash
# Lance la démo Streamlit en local (double-clic sur macOS, ou ./lancer_demo.command).
# Crée un environnement Python dédié (.venv-demo) au premier lancement, puis ouvre le navigateur.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv-demo"
if [ ! -x "$VENV/bin/streamlit" ]; then
  echo "Premier lancement : installation de l'environnement (1 à 2 minutes)…"
  if command -v uv >/dev/null 2>&1; then
    uv venv -q -p 3.12 "$VENV"
    VIRTUAL_ENV="$PWD/$VENV" uv pip install -q -r demo/requirements.txt
  else
    PY=""
    for c in python3.13 python3.12 python3.11 python3; do
      if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
        PY="$c"; break
      fi
    done
    if [ -z "$PY" ]; then
      echo "Python 3.11 ou plus récent est requis. Installez-le depuis https://www.python.org/downloads/ puis relancez."
      read -r -p "Appuyez sur Entrée pour fermer." _; exit 1
    fi
    "$PY" -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q -r demo/requirements.txt
  fi
fi

# Port dédié : on évite 8501, souvent déjà pris par une autre app Streamlit,
# et on prend le premier port libre à partir de 8505.
PORT=""
for p in $(seq 8505 8530); do
  if ! curl -s -o /dev/null --max-time 1 "http://localhost:$p"; then PORT="$p"; break; fi
done
[ -z "$PORT" ] && { echo "Aucun port libre entre 8505 et 8530."; exit 1; }
URL="http://localhost:$PORT"

echo "Démo « Scoring crédit » (projet 5) sur $URL"
echo "Pour l'arrêter : Ctrl+C ou fermer cette fenêtre."
# Ouvre le navigateur dès que CETTE démo répond (on vérifie un fichier qui n'existe que dans ce projet)
( for _ in $(seq 1 90); do
    if curl -s -o /dev/null -f "$URL/app/static/fonts/montserrat-latin-wght-normal.woff2"; then
      (open "$URL" || xdg-open "$URL") >/dev/null 2>&1; break
    fi
    sleep 1
  done ) &
# --server.address 127.0.0.1 : la démo n'est accessible que depuis ce Mac, pas depuis le réseau (Wi-Fi, etc.)
exec "$VENV/bin/streamlit" run demo/app.py --server.port "$PORT" --server.address 127.0.0.1 --server.headless true
