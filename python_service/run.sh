#!/usr/bin/env bash
set -euo pipefail

export HF_TOKEN=${HF_TOKEN:-}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
VENV_DIR="$SCRIPT_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
  source "$VENV_DIR/bin/activate"
fi

cd "$SCRIPT_DIR"
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000


