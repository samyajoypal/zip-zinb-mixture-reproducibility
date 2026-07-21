#!/usr/bin/env bash
#SBATCH --job-name=sc_syn_diag
#SBATCH --partition=epyc-256
#SBATCH --cpus-per-task=15
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/sc_syn_diag_%j.out
#SBATCH --error=slurm_logs/sc_syn_diag_%j.err

set -euo pipefail

cd "$HOME/zip_zinb_analytical_project"
mkdir -p slurm_logs

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/zip_zinb_matplotlib_${SLURM_JOB_ID}

REPS=${REPS:-5}
DIAGNOSTIC_REPS=${DIAGNOSTIC_REPS:-3}
N=${N:-20000}
N_STARTS=${N_STARTS:-5}
MAXITER=${MAXITER:-1500}

.venv/bin/python scripts/run_bootstrap_diagnostics.py \
  --scenarios all \
  --reps "${REPS}" \
  --n "${N}" \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --n-starts "${N_STARTS}" \
  --maxiter "${MAXITER}" \
  --n-test 5000 \
  --bootstrap-reps 0 \
  --bootstrap-models Z+P+NB \
  --diagnostic-reps "${DIAGNOSTIC_REPS}" \
  --out-dir outputs/robustness_checks/statscomp_synthetic_diagnostics
