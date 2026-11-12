#!/usr/bin/env bash
#SBATCH --job-name=magset-fusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-29%4
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_CODE_SHA:?Set MAGSET_CODE_SHA}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_I="${SLURM_ARRAY_TASK_ID}"
TASK_TRIAL=$((TASK_I % 5 + 1))
TASK_VIEWS=(mean gated attention)
TASK_VIEW="${TASK_VIEWS[$(((TASK_I / 5) % 3))]}"
if (( TASK_I < 15 )); then TASK_FIELD=mean; else TASK_FIELD=attention; fi
cd "$MAGSET_ROOT/repo"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.train_fusion \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" \
  --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
  --features "$MAGSET_ROOT/cache/features" --encoder phikon-v2 \
  --field-pool "$TASK_FIELD" --view-fusion "$TASK_VIEW" \
  --config configs/fusion-pilot.json --trial "$TASK_TRIAL" --seed 812 --training-seed 812 \
  --code-sha "$MAGSET_CODE_SHA" \
  --output "$MAGSET_ROOT/runs/fusion-v1/phikon-v2-field-$TASK_FIELD-view-$TASK_VIEW-trial$TASK_TRIAL-seed812"
