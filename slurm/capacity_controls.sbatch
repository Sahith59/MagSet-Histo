#!/usr/bin/env bash
#SBATCH --job-name=magset-capacity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-39%4
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_CODE_SHA:?Set MAGSET_CODE_SHA}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_I="${SLURM_ARRAY_TASK_ID}"
TASK_TRIAL=$((TASK_I % 5 + 1))
if (( (TASK_I / 5) % 2 == 0 )); then TASK_METHOD=mean; else TASK_METHOD=attention; fi
if (( (TASK_I / 10) % 2 == 0 )); then TASK_SPLIT=812; else TASK_SPLIT=813; fi
if (( TASK_I < 20 )); then TASK_POLICY=small; else TASK_POLICY=dropout; fi
cd "$MAGSET_ROOT/repo"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.train_fusion \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" \
  --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
  --features "$MAGSET_ROOT/cache/features" --encoder phikon-v2 \
  --field-pool "$TASK_METHOD" --view-fusion "$TASK_METHOD" --config "configs/fusion-$TASK_POLICY.json" \
  --trial "$TASK_TRIAL" --seed "$TASK_SPLIT" --training-seed 812 --code-sha "$MAGSET_CODE_SHA" \
  --output "$MAGSET_ROOT/runs/capacity-v1/$TASK_POLICY/split$TASK_SPLIT/phikon-v2-field-$TASK_METHOD-view-$TASK_METHOD-trial$TASK_TRIAL-seed$TASK_SPLIT"
