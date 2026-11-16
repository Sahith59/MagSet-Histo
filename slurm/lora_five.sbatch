#!/usr/bin/env bash
#SBATCH --job-name=magset-lora-five
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source}"
: "${MAGSET_CODE_SHA:?Set immutable source SHA}"
: "${MAGSET_LORA_ACTION:?Use audit, smoke, run or collect}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_RUN_ROOT="$MAGSET_ROOT/runs/breakhis-midnight-lora-v1"
TASK_ARGS=(--manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json"
 --cache "$MAGSET_ROOT/cache/features-midnight-v1" --config "$MAGSET_SOURCE/configs/breakhis-midnight-lora-v1.json"
 --weights "$MAGSET_ROOT/cache/midnight-12k" --initial-runs "$MAGSET_ROOT/runs/breakhis-midnight-five-v1"
 --images-root "$MAGSET_ROOT/data/raw" --code-sha "$MAGSET_CODE_SHA")
case "$MAGSET_LORA_ACTION" in
 audit) TASK_ARGS+=(--output "$TASK_RUN_ROOT/input-audit") ;;
 smoke)
  TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Use two-task technical array}"
  [[ "$TASK_INDEX" == 0 || "$TASK_INDEX" == 1 ]]
  TASK_ARM=direct; [[ "$TASK_INDEX" == 0 ]] || TASK_ARM=dual
  TASK_ARGS+=(--arm "$TASK_ARM" --seed 812 --trial 1 --audit "$TASK_RUN_ROOT/input-audit"
   --output "$MAGSET_ROOT/runs/breakhis-midnight-lora-smoke-v1/$TASK_ARM") ;;
 run)
  TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Use twenty-task production array}"
  [[ "$TASK_INDEX" =~ ^[0-9]+$ ]] && (( TASK_INDEX >= 0 && TASK_INDEX < 20 ))
  TASK_ARM=direct; (( TASK_INDEX < 10 )) || TASK_ARM=dual
  TASK_ROLE=$((TASK_INDEX % 10)); TASK_SEED=$((812 + TASK_ROLE / 5)); TASK_TRIAL=$((TASK_ROLE % 5 + 1))
  TASK_ARGS+=(--arm "$TASK_ARM" --seed "$TASK_SEED" --trial "$TASK_TRIAL" --audit "$TASK_RUN_ROOT/input-audit"
   --output "$TASK_RUN_ROOT/$TASK_ARM/split$TASK_SEED/trial$TASK_TRIAL") ;;
 collect) TASK_ARGS+=(--output "$TASK_RUN_ROOT/checked" --runs "$TASK_RUN_ROOT" --audit "$TASK_RUN_ROOT/input-audit") ;;
 *) exit 2 ;;
esac
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.lora_five "$MAGSET_LORA_ACTION" "${TASK_ARGS[@]}"
