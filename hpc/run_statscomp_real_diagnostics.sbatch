#!/usr/bin/env bash
#SBATCH --job-name=sc_real_diag
#SBATCH --partition=epyc-256
#SBATCH --cpus-per-task=15
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/sc_real_diag_%j.out
#SBATCH --error=slurm_logs/sc_real_diag_%j.err

set -euo pipefail

cd "$HOME/zip_zinb_analytical_project"
mkdir -p slurm_logs

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/zip_zinb_matplotlib_${SLURM_JOB_ID}

N_STARTS=${N_STARTS:-5}
MAXITER=${MAXITER:-1500}

.venv/bin/python scripts/run_real_extended_diagnostics.py \
  --mode diagnostics \
  --datasets all \
  --links all \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --n-starts "${N_STARTS}" \
  --maxiter "${MAXITER}" \
  --out-dir outputs/robustness_checks/statscomp_real_diagnostics
