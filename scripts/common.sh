#!/usr/bin/env bash
# Shared path resolution only: no GPU reservation, audits or background jobs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${WORK_ROOT:-}" ]]; then
  if [[ -f "$ROOT/.work-root" ]]; then
    IFS= read -r WORK_ROOT < "$ROOT/.work-root"
  else
    WORK_ROOT="$(dirname "$ROOT")/refwave-assets"
  fi
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python || command -v python3 || printf '%s' python3)"
  fi
fi
export WORK_ROOT PYTHON_BIN
# Do not inherit another checkout's PYTHONPATH or user-installed Python modules.
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/src"
