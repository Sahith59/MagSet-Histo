#!/usr/bin/env bash
#SBATCH --job-name=magset-approved-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source}"
: "${MAGSET_CODE_SHA:?Set immutable source SHA}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
TASK_MODELS=(uni2-h-v1 virchow2-v1 h-optimus-0-v1)
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Use three-model array}"
[[ "$TASK_INDEX" =~ ^[0-9]+$ ]] && (( TASK_INDEX>=0 && TASK_INDEX<3 ))
TASK_MODEL="${TASK_MODELS[$TASK_INDEX]}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u scripts/audit_approved_features.py --config "configs/$TASK_MODEL.json" --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --cache "$MAGSET_ROOT/cache/features-$TASK_MODEL" --code-sha "$MAGSET_CODE_SHA" --output "$MAGSET_ROOT/cache/approved-audits/$TASK_MODEL.json"
