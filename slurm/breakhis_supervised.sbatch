#!/usr/bin/env bash
#SBATCH --job-name=magset-breakhis-supervised
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-9%4
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source}"
: "${MAGSET_CODE_SHA:?Set full source SHA}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export TORCH_HOME="$MAGSET_ROOT/cache/torch" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
TASK_TRIAL=$((SLURM_ARRAY_TASK_ID % 5 + 1))
if (( SLURM_ARRAY_TASK_ID < 5 )); then TASK_SEED=812; else TASK_SEED=813; fi
TASK_OUTPUT="$MAGSET_ROOT/runs/breakhis-supervised-v1/split$TASK_SEED/trial$TASK_TRIAL"
TASK_OPTIONS=()
if [[ "${MAGSET_SMOKE:-0}" == 1 ]]; then
  [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]
  TASK_OUTPUT="$MAGSET_ROOT/runs/breakhis-supervised-v1/technical-smoke"
  if [[ "${MAGSET_SMOKE_ATTEMPT:-0}" == 1 ]]; then TASK_OUTPUT="$MAGSET_ROOT/runs/breakhis-supervised-v1/technical-smoke-retry1"; fi
  TASK_OPTIONS+=(--smoke)
fi
if [[ "${MAGSET_RESUME:-0}" == 1 ]]; then TASK_OPTIONS+=(--resume); fi
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.supervised_breakhis \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
  --reference-root "$MAGSET_ROOT/runs/matched-mean-v1" --images-root "$MAGSET_ROOT/data/raw" \
  --weights "$MAGSET_ROOT/cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth" \
  --config configs/breakhis-supervised-v1.json --trial "$TASK_TRIAL" --seed "$TASK_SEED" \
  --code-sha "$MAGSET_CODE_SHA" --workers 3 --output "$TASK_OUTPUT" "${TASK_OPTIONS[@]}"
