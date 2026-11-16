#!/usr/bin/env bash
#SBATCH --job-name=magset-native-five
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source}"
: "${MAGSET_CODE_SHA:?Set immutable source SHA}"
: "${MAGSET_NATIVE_ACTION:?Use audit, run or collect}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_ROOT="$MAGSET_ROOT/runs/breakhis-native-kernel-v1"
TASK_ARGS=(--manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json"
 --cache "$MAGSET_ROOT/cache/features-midnight-native-v1" --center-cache "$MAGSET_ROOT/cache/features-midnight-v1"
 --feature-audit "$MAGSET_ROOT/cache/features-midnight-native-v1/feature-audit.json"
 --config "$MAGSET_SOURCE/configs/breakhis-native-kernel-v1.json" --code-sha "$MAGSET_CODE_SHA")
case "$MAGSET_NATIVE_ACTION" in
 audit) TASK_ARGS+=(--output "$TASK_ROOT/input-audit") ;;
 run)
  TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Use ten-task array}"
  [[ "$TASK_INDEX" =~ ^[0-9]+$ ]] && (( TASK_INDEX >= 0 && TASK_INDEX < 10 ))
  TASK_SEED=$((812 + TASK_INDEX / 5)); TASK_TRIAL=$((TASK_INDEX % 5 + 1))
  TASK_ARGS+=(--output "$TASK_ROOT/split$TASK_SEED/trial$TASK_TRIAL" --seed "$TASK_SEED" --trial "$TASK_TRIAL" --audit "$TASK_ROOT/input-audit") ;;
 collect) TASK_ARGS+=(--output "$TASK_ROOT/checked" --runs "$TASK_ROOT" --audit "$TASK_ROOT/input-audit") ;;
 *) exit 2 ;;
esac
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.native_five "$MAGSET_NATIVE_ACTION" "${TASK_ARGS[@]}"
