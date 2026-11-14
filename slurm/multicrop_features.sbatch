#!/usr/bin/env bash
#SBATCH --job-name=magset-regional-features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-3%4
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source directory}"
: "${MAGSET_CODE_SHA:?Set immutable source SHA}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export HF_HOME="$MAGSET_ROOT/cache/huggingface" TORCH_HOME="$MAGSET_ROOT/cache/torch"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
TASK_MAGS=(40 100 200 400)
TASK_MAG="${TASK_MAGS[$SLURM_ARRAY_TASK_ID]}"
TASK_OUTPUT="$MAGSET_ROOT/cache/breakhis-multicrop-v1/mag$TASK_MAG"
TASK_OPTIONS=()
if [[ "${MAGSET_SMOKE:-0}" == 1 ]]; then
  [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]
  TASK_OUTPUT="$MAGSET_ROOT/runs/breakhis-multicrop-v1/technical-smoke"
  TASK_OPTIONS+=(--smoke)
fi
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.multicrop_features \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
  --images-root "$MAGSET_ROOT/data/raw" --thumbnail-cache "$MAGSET_ROOT/cache/features" \
  --config configs/breakhis-multicrop-v1.json --output "$TASK_OUTPUT" \
  --magnification "$TASK_MAG" --code-sha "$MAGSET_CODE_SHA" --batch-size 16 --workers 2 "${TASK_OPTIONS[@]}"
