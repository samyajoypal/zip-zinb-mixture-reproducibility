#!/usr/bin/env bash
#SBATCH --job-name=sc_syn_boot
#SBATCH --partition=epyc-256
#SBATCH --cpus-per-task=15
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-23%6
#SBATCH --output=slurm_logs/sc_syn_boot_%A_%a.out
#SBATCH --error=slurm_logs/sc_syn_boot_%A_%a.err

set -euo pipefail

cd "$HOME/zip_zinb_analytical_project"
mkdir -p slurm_logs

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/zip_zinb_matplotlib_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

SCENARIOS=(D1 D2 D3 D4)
MODELS=("Poisson" "NB" "P+NB" "ZIP" "ZINB" "Z+P+NB")

IDX=${SLURM_ARRAY_TASK_ID}
SCENARIO=${SCENARIOS[$((IDX / 6))]}
MODEL=${MODELS[$((IDX % 6))]}
SAFE_MODEL=${MODEL//+/_plus_}
SAFE_MODEL=${SAFE_MODEL//--/_}
SAFE_MODEL=${SAFE_MODEL// /_}

REPS=${REPS:-20}
BOOTSTRAP_REPS=${BOOTSTRAP_REPS:-99}
N=${N:-20000}
N_STARTS=${N_STARTS:-5}
MAXITER=${MAXITER:-1500}

OUT_DIR="outputs/robustness_checks/statscomp_synthetic_bootstrap_all/${SCENARIO}_${SAFE_MODEL}"

echo "Scenario=${SCENARIO}"
echo "Restricted model=${MODEL}"
echo "Output=${OUT_DIR}"
echo "REPS=${REPS}; BOOTSTRAP_REPS=${BOOTSTRAP_REPS}; N=${N}"

.venv/bin/python scripts/run_bootstrap_diagnostics.py \
  --scenarios "${SCENARIO}" \
  --reps "${REPS}" \
  --n "${N}" \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --n-starts "${N_STARTS}" \
  --maxiter "${MAXITER}" \
  --n-test 0 \
  --bootstrap-reps "${BOOTSTRAP_REPS}" \
  --bootstrap-models "${MODEL}" \
  --diagnostic-reps 0 \
  --out-dir "${OUT_DIR}"
