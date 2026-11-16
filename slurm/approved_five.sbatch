#!/usr/bin/env bash
#SBATCH --job-name=magset-approved-five
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source}"
: "${MAGSET_CODE_SHA:?Set immutable source SHA}"
: "${MAGSET_APPROVED_ACTION:?Use audit, run or collect}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
TASK_MODELS=(uni2-h-v1 virchow2-v1 h-optimus-0-v1)
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Use the model or role array}"
[[ "$TASK_INDEX" =~ ^[0-9]+$ ]]
if [[ "$MAGSET_APPROVED_ACTION" == run ]]; then
 (( TASK_INDEX>=0 && TASK_INDEX<30 ))
 TASK_MODEL="${TASK_MODELS[$((TASK_INDEX/10))]}"
 TASK_ROLE=$((TASK_INDEX%10))
else
 (( TASK_INDEX>=0 && TASK_INDEX<3 ))
 TASK_MODEL="${TASK_MODELS[$TASK_INDEX]}"
fi
TASK_ROOT="$MAGSET_ROOT/runs/breakhis-$TASK_MODEL-kernel"
TASK_ARGS=(--manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" --cache "$MAGSET_ROOT/cache/features-$TASK_MODEL" --center-cache "$MAGSET_ROOT/cache/features-midnight-v1" --feature-audit "$MAGSET_ROOT/cache/approved-audits/$TASK_MODEL.json" --config "$MAGSET_SOURCE/configs/breakhis-$TASK_MODEL-kernel.json" --code-sha "$MAGSET_CODE_SHA")
case "$MAGSET_APPROVED_ACTION" in
 audit) TASK_ARGS+=(--output "$TASK_ROOT/input-audit") ;;
 run)
  TASK_SEED=$((812+TASK_ROLE/5)); TASK_TRIAL=$((TASK_ROLE%5+1))
  TASK_ARGS+=(--output "$TASK_ROOT/split$TASK_SEED/trial$TASK_TRIAL" --seed "$TASK_SEED" --trial "$TASK_TRIAL" --audit "$TASK_ROOT/input-audit") ;;
 collect) TASK_ARGS+=(--output "$TASK_ROOT/checked" --runs "$TASK_ROOT" --audit "$TASK_ROOT/input-audit") ;;
 *) exit 2 ;;
esac
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.approved_five "$MAGSET_APPROVED_ACTION" "${TASK_ARGS[@]}"
