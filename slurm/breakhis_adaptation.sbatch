#!/usr/bin/env bash
#SBATCH --job-name=magset-breakhis-adapt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-9%4
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_SOURCE:?Set immutable MAGSET_SOURCE}"
: "${MAGSET_CODE_SHA:?Set full MAGSET_CODE_SHA}"
: "${MAGSET_PROTOCOL_SHA:?Set protocol checksum}"
: "${MAGSET_CONFIG_SHA:?Set registered configuration checksum}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export HF_HOME="$MAGSET_ROOT/cache/huggingface" TORCH_HOME="$MAGSET_ROOT/cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_TRIAL=$((SLURM_ARRAY_TASK_ID % 5 + 1))
if (( SLURM_ARRAY_TASK_ID < 5 )); then TASK_SPLIT=812; else TASK_SPLIT=813; fi
TASK_OUTPUT="$MAGSET_ROOT/runs/breakhis-adaptation-v1/split$TASK_SPLIT/trial$TASK_TRIAL"
TASK_OPTIONS=()
if [[ "${MAGSET_SMOKE:-0}" == 1 ]]; then
  [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]
  TASK_OUTPUT="$MAGSET_ROOT/runs/breakhis-adaptation-v1/technical-smoke"
  TASK_OPTIONS+=(--smoke)
fi
if [[ "${MAGSET_RESUME:-0}" == 1 ]]; then TASK_OPTIONS+=(--resume); fi
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.adapt_breakhis \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
  --images-root "$MAGSET_ROOT/data/raw" --frozen-cache "$MAGSET_ROOT/cache/features" \
  --reference-root "$MAGSET_ROOT/runs/matched-mean-v1" --config configs/breakhis-adaptation-v1.json \
  --trial "$TASK_TRIAL" --seed "$TASK_SPLIT" --code-sha "$MAGSET_CODE_SHA" \
  --expected-protocol-sha "$MAGSET_PROTOCOL_SHA" --expected-config-sha "$MAGSET_CONFIG_SHA" \
  --workers 3 --output "$TASK_OUTPUT" "${TASK_OPTIONS[@]}"
