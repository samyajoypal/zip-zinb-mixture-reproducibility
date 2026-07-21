#!/usr/bin/env bash
set -euo pipefail

# One-click local simulation run.
#
# Usage from this independent project root:
#   bash run_full_simulation.sh
#
# Optional overrides are passed through, e.g.
#   bash run_full_simulation.sh --n-jobs 8
#   bash run_full_simulation.sh --reps 10 --n 2000
#   bash run_full_simulation.sh --skip-consistency

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/zip_zinb_matplotlib}"

PYTHON_BIN="${PYTHON_BIN:-/home/samyajoypal/.local/share/pipx/venvs/spyder/bin/python}"

"${PYTHON_BIN}" scripts/run_simulation_analytical_full.py "$@"
