#!/usr/bin/env bash
# Startet Bildwerk unter macOS und Linux.
# Beim ersten Start wird eine virtuelle Python-Umgebung (.venv) angelegt und alles installiert.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python 3.10 oder neuer wird benoetigt: https://www.python.org/downloads/" >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Richte Bildwerk ein (einmalig, dauert ein bis zwei Minuten) ..."
  "$PYTHON" -m venv .venv
fi

if [ ! -f .venv/.installed ] || [ pyproject.toml -nt .venv/.installed ]; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e .
  # Lokale KI (ONNX Runtime) ist optional, Bildwerk laeuft auch ohne
  if ! .venv/bin/python -m pip install -e ".[local]"; then
    echo "Hinweis: Die lokale KI liess sich nicht installieren - Bildwerk startet ohne KI-Schaerfen auf diesem Rechner." >&2
  fi
  touch .venv/.installed
fi

exec .venv/bin/python -m bildwerk "$@"
