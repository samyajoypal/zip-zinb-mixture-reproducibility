#!/usr/bin/env bash
#SBATCH --job-name=sc_real_boot
#SBATCH --partition=epyc-256
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-47%32
#SBATCH --output=slurm_logs/sc_real_boot_%A_%a.out
#SBATCH --error=slurm_logs/sc_real_boot_%A_%a.err

set -euo pipefail

cd "$HOME/zip_zinb_analytical_project"
mkdir -p slurm_logs

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/zip_zinb_matplotlib_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

DATASETS=("C2" "C3" "H2AX1H60" "H2AX4H100")
LINKS=("identity" "log")
MODELS=("Poisson" "NB" "P+NB" "ZIP" "ZINB" "Z+P+NB")

IDX=${SLURM_ARRAY_TASK_ID}
DATASET=${DATASETS[$((IDX / 12))]}
LINK=${LINKS[$(((IDX / 6) % 2))]}
MODEL=${MODELS[$((IDX % 6))]}
SAFE_MODEL=${MODEL//+/_plus_}
SAFE_MODEL=${SAFE_MODEL//--/_}
SAFE_MODEL=${SAFE_MODEL// /_}

BOOTSTRAP_REPS=${BOOTSTRAP_REPS:-99}
N_STARTS=${N_STARTS:-5}
MAXITER=${MAXITER:-1500}

OUT_DIR="outputs/robustness_checks/statscomp_real_bootstrap_all/${DATASET}_${LINK}_${SAFE_MODEL}"

echo "Dataset=${DATASET}"
echo "Link=${LINK}"
echo "Restricted model=${MODEL}"
echo "Output=${OUT_DIR}"
echo "BOOTSTRAP_REPS=${BOOTSTRAP_REPS}"

.venv/bin/python scripts/run_real_extended_diagnostics.py \
  --mode bootstrap \
  --datasets "${DATASET}" \
  --links "${LINK}" \
  --bootstrap-models "${MODEL}" \
  --bootstrap-reps "${BOOTSTRAP_REPS}" \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --n-starts "${N_STARTS}" \
  --maxiter "${MAXITER}" \
  --out-dir "${OUT_DIR}"
