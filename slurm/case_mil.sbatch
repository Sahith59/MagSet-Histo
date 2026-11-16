#!/usr/bin/env bash
#SBATCH --job-name=magset-case-mil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source}"
: "${MAGSET_CODE_SHA:?Set immutable SHA}"
: "${MAGSET_MIL_ACTION:?Use audit, smoke, verify-smokes, run or collect}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_ROOT="$MAGSET_ROOT/runs/breakhis-case-mil-v1"
TASK_SMOKES="$MAGSET_ROOT/runs/breakhis-case-mil-smokes-v1"
TASK_ARGS=(--manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" --cache "$MAGSET_ROOT/cache/features-midnight-coverage-v1" --center-cache "$MAGSET_ROOT/cache/features-midnight-v1" --feature-audit "$MAGSET_ROOT/cache/features-midnight-coverage-v1/feature-audit.json" --config "$MAGSET_SOURCE/configs/breakhis-case-mil-v1.json" --code-sha "$MAGSET_CODE_SHA")
case "$MAGSET_MIL_ACTION" in
 audit) TASK_ARGS+=(--output "$TASK_ROOT/input-audit") ;;
 smoke|run)
  TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Use registered array}"
  [[ "$TASK_INDEX" =~ ^[0-9]+$ ]]
  if [[ "$MAGSET_MIL_ACTION" == smoke ]]; then
   ((TASK_INDEX>=0 && TASK_INDEX<4)); TASK_ROLE=0; TASK_METHOD=$TASK_INDEX
  else
   ((TASK_INDEX>=0 && TASK_INDEX<40)); TASK_ROLE=$((TASK_INDEX%10)); TASK_METHOD=$((TASK_INDEX/10))
  fi
  TASK_POOL=mean; (( TASK_METHOD<2 )) || TASK_POOL=attention
  TASK_ARM=direct; (( TASK_METHOD%2==0 )) || TASK_ARM=dual
  TASK_SEED=$((812+TASK_ROLE/5)); TASK_TRIAL=$((TASK_ROLE%5+1))
  TASK_OUTPUT="$TASK_ROOT/$TASK_POOL/$TASK_ARM/split$TASK_SEED/trial$TASK_TRIAL"
  [[ "$MAGSET_MIL_ACTION" != smoke ]] || TASK_OUTPUT="$TASK_SMOKES/$TASK_POOL/$TASK_ARM"
  TASK_ARGS+=(--pool "$TASK_POOL" --arm "$TASK_ARM" --seed "$TASK_SEED" --trial "$TASK_TRIAL" --audit "$TASK_ROOT/input-audit" --smokes "$TASK_SMOKES" --output "$TASK_OUTPUT") ;;
 verify-smokes) TASK_ARGS+=(--output "$TASK_ROOT/smoke-acceptance" --audit "$TASK_ROOT/input-audit" --smokes "$TASK_SMOKES") ;;
 collect) TASK_ARGS+=(--output "$TASK_ROOT/checked" --runs "$TASK_ROOT" --audit "$TASK_ROOT/input-audit" --smokes "$TASK_SMOKES") ;;
 *) exit 2 ;;
esac
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.case_mil_five "$MAGSET_MIL_ACTION" "${TASK_ARGS[@]}"
