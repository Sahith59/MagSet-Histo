#!/usr/bin/env bash
#SBATCH --job-name=magset-validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-29%6
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_CODE_SHA:?Set MAGSET_CODE_SHA}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
TASK_METHODS=(mean concat late)
TASK_I="${SLURM_ARRAY_TASK_ID}"
TASK_TRIAL=$((TASK_I % 5 + 1))
TASK_METHOD="${TASK_METHODS[$(((TASK_I / 5) % 3))]}"
if (( TASK_I < 15 )); then TASK_ENCODER=phikon-v2; else TASK_ENCODER=resnet50; fi
cd "$MAGSET_ROOT/repo"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.baseline \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" \
  --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
  --features "$MAGSET_ROOT/cache/features" --encoder "$TASK_ENCODER" --method "$TASK_METHOD" \
  --trial "$TASK_TRIAL" --seed 812 --code-sha "$MAGSET_CODE_SHA" \
  --output "$MAGSET_ROOT/runs/validation-v1/$TASK_ENCODER-$TASK_METHOD-trial$TASK_TRIAL-seed812"
# Outer-test predictions are deliberately disabled in this initial pilot.
