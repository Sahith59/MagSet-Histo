#!/usr/bin/env bash
#SBATCH --job-name=magset-uni-linear
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source}"
: "${MAGSET_CODE_SHA:?Set immutable source SHA}"
: "${MAGSET_UNI_LINEAR_ACTION:?Use audit, smoke, verify-smoke, run or collect}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
TASK_ROOT="$MAGSET_ROOT/runs/breakhis-uni-linear-v1"
TASK_SMOKES="$MAGSET_ROOT/runs/breakhis-uni-linear-smoke-v1"
TASK_ARGS=(--manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" --cache "$MAGSET_ROOT/cache/features-uni2-h-v1" --center-cache "$MAGSET_ROOT/cache/features-midnight-v1" --feature-audit "$MAGSET_ROOT/cache/approved-audits/uni2-h-v1.json" --config "$MAGSET_SOURCE/configs/breakhis-uni-linear-v1.json" --code-sha "$MAGSET_CODE_SHA")
case "$MAGSET_UNI_LINEAR_ACTION" in
 audit) TASK_ARGS+=(--output "$TASK_ROOT/input-audit") ;;
 smoke) TASK_ARGS+=(--output "$TASK_SMOKES" --seed 812 --trial 1 --audit "$TASK_ROOT/input-audit") ;;
 verify-smoke) TASK_ARGS+=(--output "$TASK_ROOT/smoke-acceptance" --smokes "$TASK_SMOKES" --audit "$TASK_ROOT/input-audit") ;;
 run)
  TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Use original ten-role array}"
  [[ "$TASK_INDEX" =~ ^[0-9]+$ ]]; ((TASK_INDEX>=0 && TASK_INDEX<10))
  TASK_SEED=$((812+TASK_INDEX/5)); TASK_TRIAL=$((TASK_INDEX%5+1))
  TASK_ARGS+=(--output "$TASK_ROOT/split$TASK_SEED/trial$TASK_TRIAL" --seed "$TASK_SEED" --trial "$TASK_TRIAL" --audit "$TASK_ROOT/input-audit" --smokes "$TASK_SMOKES" --smoke-acceptance "$TASK_ROOT/smoke-acceptance/acceptance.json") ;;
 collect) TASK_ARGS+=(--output "$TASK_ROOT/checked" --runs "$TASK_ROOT" --audit "$TASK_ROOT/input-audit" --smokes "$TASK_SMOKES" --smoke-acceptance "$TASK_ROOT/smoke-acceptance/acceptance.json") ;;
 *) exit 2 ;;
esac
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.uni_linear_five "$MAGSET_UNI_LINEAR_ACTION" "${TASK_ARGS[@]}"
