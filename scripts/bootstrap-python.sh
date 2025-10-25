#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PS_DIR="$ROOT_DIR/python_service"
VENV_DIR="$PS_DIR/.venv"

PYTHON_BIN="python3"
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[bootstrap] Creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "[bootstrap] Activating venv"
source "$VENV_DIR/bin/activate"

echo "[bootstrap] Upgrading pip"
pip install -U pip

echo "[bootstrap] Installing/upgrading requirements (latest compatible)"
pip install --upgrade --upgrade-strategy eager -r "$ROOT_DIR/requirements.txt"

echo "[bootstrap] Verifying environment (pip check)"
pip check || true

echo "[bootstrap] Done"


