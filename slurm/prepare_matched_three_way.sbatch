#!/usr/bin/env bash
#SBATCH --job-name=magset-matched-inputs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}" "${MAGSET_MATCHED_ACTION:?}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_RUNS="$MAGSET_ROOT/runs/breakhis-matched-three-way-v1"
mkdir -p "$TASK_RUNS"
TASK_ARGS=("$MAGSET_MATCHED_ACTION" --config "$MAGSET_SOURCE/configs/breakhis-matched-three-way-v1.json" --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --raw-root "$MAGSET_ROOT/data/raw" --cache "$MAGSET_ROOT/cache/features-midnight-coverage-v1" --code-sha "$MAGSET_CODE_SHA")
case "$MAGSET_MATCHED_ACTION" in
 audit) TASK_ARGS+=(--output "$TASK_RUNS/input-audit") ;;
 prepare) TASK_ARGS+=(--audit "$TASK_RUNS/input-audit" --output "$TASK_RUNS/pixels" --workers 4) ;;
 *) exit 2 ;;
esac
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u scripts/prepare_matched_three_way.py "${TASK_ARGS[@]}"
