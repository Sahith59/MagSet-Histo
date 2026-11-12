#!/usr/bin/env bash
#SBATCH --job-name=magset-matched-mean
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-9%4
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_CODE_SHA:?Set MAGSET_CODE_SHA}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_TRIAL=$((SLURM_ARRAY_TASK_ID % 5 + 1))
if (( SLURM_ARRAY_TASK_ID < 5 )); then TASK_SPLIT=812; else TASK_SPLIT=813; fi
cd "$MAGSET_ROOT/repo"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.baseline \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
  --features "$MAGSET_ROOT/cache/features" --encoder phikon-v2 --method mean \
  --normalization field-moments-v2 --trial "$TASK_TRIAL" --seed "$TASK_SPLIT" \
  --code-sha "$MAGSET_CODE_SHA" \
  --output "$MAGSET_ROOT/runs/matched-mean-v1/split$TASK_SPLIT/phikon-v2-mean-trial$TASK_TRIAL-seed$TASK_SPLIT"
