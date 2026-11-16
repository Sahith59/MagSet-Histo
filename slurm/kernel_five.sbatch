#!/usr/bin/env bash
#SBATCH --job-name=magset-kernel-five
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source}"
: "${MAGSET_CODE_SHA:?Set immutable SHA}"
: "${MAGSET_KERNEL_ACTION:?Use audit, run or collect}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_RUN_ROOT="$MAGSET_ROOT/runs/breakhis-midnight-kernel-v1"
TASK_ARGS=(--manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json"
 --cache "$MAGSET_ROOT/cache/features-midnight-v1" --config "$MAGSET_SOURCE/configs/breakhis-midnight-kernel-v1.json" --code-sha "$MAGSET_CODE_SHA")
case "$MAGSET_KERNEL_ACTION" in
 audit) TASK_ARGS+=(--output "$TASK_RUN_ROOT/input-audit") ;;
 run)
  TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Need ten-task array}"
  [[ "$TASK_INDEX" =~ ^[0-9]+$ ]] && (( TASK_INDEX >= 0 && TASK_INDEX < 10 ))
  TASK_SEED=$((812 + TASK_INDEX / 5)); TASK_TRIAL=$((TASK_INDEX % 5 + 1))
  TASK_ARGS+=(--seed "$TASK_SEED" --trial "$TASK_TRIAL" --output "$TASK_RUN_ROOT/split$TASK_SEED/trial$TASK_TRIAL" --audit "$TASK_RUN_ROOT/input-audit") ;;
 collect) TASK_ARGS+=(--output "$TASK_RUN_ROOT/checked" --runs "$TASK_RUN_ROOT" --audit "$TASK_RUN_ROOT/input-audit") ;;
 *) exit 2 ;;
esac
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.kernel_five "$MAGSET_KERNEL_ACTION" "${TASK_ARGS[@]}"
