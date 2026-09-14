#!/usr/bin/env bash
# Loyiha Streamlit UI — conda base NumPy 2.x ni chetlab o‘tadi.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do
    conda deactivate 2>/dev/null || break
  done
fi

unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_SHLVL CONDA_PROMPT_MODIFIER
export PYTHONNOUSERSITE=1
export VIRTUAL_ENV="$ROOT/venv"
export PATH="$ROOT/venv/bin:${PATH}"

if [ ! -x "$ROOT/venv/bin/python" ]; then
  echo "venv yo‘q. Avval: python3.10 -m venv venv && ./venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

exec "$ROOT/venv/bin/python" -s -m streamlit run src/ui/app.py "$@"
