#!/usr/bin/env bash
#SBATCH --job-name=magset-field-linear
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-19%4
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_CODE_SHA:?Set MAGSET_CODE_SHA}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_I="${SLURM_ARRAY_TASK_ID}"
TASK_TRIAL=$((TASK_I % 5 + 1))
if (( TASK_I < 10 )); then TASK_ENCODER=phikon-v2; else TASK_ENCODER=resnet50; fi
TASK_EXTRA=()
if (( (TASK_I / 5) % 2 == 0 )); then
  TASK_METHOD=field-late
else
  TASK_METHOD=field-early-late
  TASK_EXTRA=(--pool-before-classifier)
fi
cd "$MAGSET_ROOT/repo"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.field_baseline \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" \
  --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
  --features "$MAGSET_ROOT/cache/features" --encoder "$TASK_ENCODER" \
  --trial "$TASK_TRIAL" --seed 812 --code-sha "$MAGSET_CODE_SHA" \
  --output "$MAGSET_ROOT/runs/field-linear-v1/$TASK_ENCODER-$TASK_METHOD-trial$TASK_TRIAL-seed812" "${TASK_EXTRA[@]}"
