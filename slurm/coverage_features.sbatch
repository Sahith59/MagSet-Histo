#!/usr/bin/env bash
#SBATCH --job-name=magset-coverage-features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source}"
: "${MAGSET_CODE_SHA:?Set immutable source SHA}"
: "${MAGSET_COVERAGE_ACTION:?Use smoke or extract}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_LIMIT=0
case "$MAGSET_COVERAGE_ACTION" in
 smoke) TASK_MAG=40; TASK_LIMIT=4; TASK_OUTPUT="$MAGSET_ROOT/cache/midnight-coverage-smoke-v1" ;;
 extract)
  TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Use four-task shard array}"
  [[ "$TASK_INDEX" =~ ^[0-9]+$ ]] && (( TASK_INDEX >= 0 && TASK_INDEX < 4 ))
  TASK_MAGS=(40 100 200 400); TASK_MAG="${TASK_MAGS[$TASK_INDEX]}"
  TASK_OUTPUT="$MAGSET_ROOT/cache/features-midnight-coverage-v1" ;;
 *) exit 2 ;;
esac
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.coverage_features --config "$MAGSET_SOURCE/configs/midnight-coverage-v1.json" \
 --cache-root "$MAGSET_ROOT/cache/midnight-12k" --center-cache "$MAGSET_ROOT/cache/features-midnight-v1" \
 --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --images-root "$MAGSET_ROOT/data/raw" --output-root "$TASK_OUTPUT" \
 --magnification "$TASK_MAG" --limit "$TASK_LIMIT" --batch-size 8 --workers 0 --code-sha "$MAGSET_CODE_SHA"
