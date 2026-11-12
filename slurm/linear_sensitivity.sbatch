#!/usr/bin/env bash
#SBATCH --job-name=magset-linear-seeds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-4%4
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_CODE_SHA:?Set MAGSET_CODE_SHA}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_TRIAL=$((SLURM_ARRAY_TASK_ID + 1))
cd "$MAGSET_ROOT/repo"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.baseline \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" \
  --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
  --features "$MAGSET_ROOT/cache/features" --encoder phikon-v2 --method mean \
  --trial "$TASK_TRIAL" --seed 813 --code-sha "$MAGSET_CODE_SHA" \
  --output "$MAGSET_ROOT/runs/linear-sensitivity-v1/phikon-v2-mean-trial$TASK_TRIAL-seed813"
