#!/usr/bin/env bash
#SBATCH --job-name=magset-midnight-five
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable classifier source}"
: "${MAGSET_CODE_SHA:?Set classifier source SHA}"
: "${MAGSET_MIDNIGHT_FIVE_ACTION:?Use audit, run or collect}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_RUN_ROOT="$MAGSET_ROOT/runs/breakhis-midnight-five-v1"
TASK_ARGS=(--manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json"
 --cache "$MAGSET_ROOT/cache/features-midnight-v1" --config "$MAGSET_SOURCE/configs/breakhis-midnight-five-v1.json" --code-sha "$MAGSET_CODE_SHA")
case "$MAGSET_MIDNIGHT_FIVE_ACTION" in
 audit) TASK_ARGS+=(--output "$TASK_RUN_ROOT/input-audit") ;;
 run)
  TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Need twenty-task array}"
  [[ "$TASK_INDEX" =~ ^[0-9]+$ ]] && (( TASK_INDEX >= 0 && TASK_INDEX < 20 ))
  TASK_ROLE=$((TASK_INDEX % 10))
  TASK_READOUT=case
  if (( TASK_INDEX >= 10 )); then TASK_READOUT=field; fi
  TASK_SEED=$((812 + TASK_ROLE / 5))
  TASK_TRIAL=$((TASK_ROLE % 5 + 1))
  TASK_ARGS+=(--readout "$TASK_READOUT" --seed "$TASK_SEED" --trial "$TASK_TRIAL"
   --output "$TASK_RUN_ROOT/$TASK_READOUT/split$TASK_SEED/trial$TASK_TRIAL" --audit "$TASK_RUN_ROOT/input-audit") ;;
 collect) TASK_ARGS+=(--output "$TASK_RUN_ROOT/checked" --runs "$TASK_RUN_ROOT" --audit "$TASK_RUN_ROOT/input-audit") ;;
 *) exit 2 ;;
esac
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.midnight_five "$MAGSET_MIDNIGHT_FIVE_ACTION" "${TASK_ARGS[@]}"
