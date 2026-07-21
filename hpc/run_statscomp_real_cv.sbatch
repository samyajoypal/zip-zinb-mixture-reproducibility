#!/usr/bin/env bash
#SBATCH --job-name=sc_real_cv
#SBATCH --partition=epyc-256
#SBATCH --cpus-per-task=15
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/sc_real_cv_%j.out
#SBATCH --error=slurm_logs/sc_real_cv_%j.err

set -euo pipefail

cd "$HOME/zip_zinb_analytical_project"
mkdir -p slurm_logs

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/zip_zinb_matplotlib_${SLURM_JOB_ID}

CV_FOLDS=${CV_FOLDS:-5}
N_STARTS=${N_STARTS:-5}
MAXITER=${MAXITER:-1500}

.venv/bin/python scripts/run_real_extended_diagnostics.py \
  --mode cv \
  --datasets all \
  --links all \
  --cv-folds "${CV_FOLDS}" \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --n-starts "${N_STARTS}" \
  --maxiter "${MAXITER}" \
  --out-dir outputs/robustness_checks/statscomp_real_cv
