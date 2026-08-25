#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if command -v python3 >/dev/null 2>&1; then
    python_cmd="python3"
elif command -v python >/dev/null 2>&1; then
    python_cmd="python"
else
    echo "Python 3.10 or newer is required."
    exit 1
fi

if ! "$python_cmd" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    echo "Python 3.10 or newer is required."
    exit 1
fi

venv_python="$script_dir/.venv/bin/python"

if [[ ! -x "$venv_python" ]]; then
    echo "Creating .venv..."
    "$python_cmd" -m venv .venv
fi

if ! "$venv_python" -c 'import streamlit, torch, transformers, sentencepiece' >/dev/null 2>&1; then
    echo "Installing dependencies. This may take several minutes..."
    "$venv_python" -m pip install --upgrade pip
    "$venv_python" -m pip install -e .
fi

exec "$venv_python" -m streamlit run app.py
