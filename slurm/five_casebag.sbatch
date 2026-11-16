#!/usr/bin/env bash
#SBATCH --job-name=magset-casebag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source}"
: "${MAGSET_CODE_SHA:?Set immutable SHA}"
: "${MAGSET_CASEBAG_ACTION:?Use audit, smoke or run}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME="$MAGSET_ROOT/cache/huggingface-casebag" TORCH_HOME="$MAGSET_ROOT/cache/torch"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
TASK_RUN_ROOT="$MAGSET_ROOT/runs/breakhis-casebag-v1"
TASK_ARGS=(--manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json"
 --images-root "$MAGSET_ROOT/data/raw" --weights "$MAGSET_ROOT/cache/casebag-effnetv2b0/model.safetensors"
 --config "$MAGSET_SOURCE/configs/breakhis-casebag-v1.json" --code-sha "$MAGSET_CODE_SHA" --workers 0)
case "$MAGSET_CASEBAG_ACTION" in
 audit) TASK_ARGS+=(--seed 812 --trial 1 --arm direct --device cpu --output "$TASK_RUN_ROOT/input-audit") ;;
 smoke)
  TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Need two-arm smoke array}"
  [[ "$TASK_INDEX" == 0 || "$TASK_INDEX" == 1 ]]
  TASK_ARM=direct; if [[ "$TASK_INDEX" == 1 ]]; then TASK_ARM=dual; fi
  TASK_ARGS+=(--seed 812 --trial 1 --arm "$TASK_ARM" --output "$TASK_RUN_ROOT/technical/$TASK_ARM") ;;
 run)
  TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Need production array}"
  [[ "$TASK_INDEX" =~ ^[0-9]+$ ]] && (( TASK_INDEX >= 0 && TASK_INDEX < 20 ))
  TASK_ROLE=$((TASK_INDEX % 10)); TASK_SEED=$((812 + TASK_ROLE / 5)); TASK_TRIAL=$((TASK_ROLE % 5 + 1))
  TASK_ARM=direct; if (( TASK_INDEX >= 10 )); then TASK_ARM=dual; fi
  TASK_ARGS+=(--seed "$TASK_SEED" --trial "$TASK_TRIAL" --arm "$TASK_ARM"
   --smoke-acceptance "$TASK_RUN_ROOT/technical-acceptance.json"
   --output "$TASK_RUN_ROOT/production/$TASK_ARM/split$TASK_SEED/trial$TASK_TRIAL") ;;
 *) exit 2 ;;
esac
if [[ "${MAGSET_CASEBAG_RESUME:-0}" == 1 ]]; then TASK_ARGS+=(--resume); fi
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.five_casebag "$MAGSET_CASEBAG_ACTION" "${TASK_ARGS[@]}"
