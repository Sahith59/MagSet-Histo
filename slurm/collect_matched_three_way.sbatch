#!/usr/bin/env bash
#SBATCH --job-name=magset-matched-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_RUNS="$MAGSET_ROOT/runs/breakhis-matched-three-way-v1"
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u scripts/collect_matched_three_way.py --root "$TASK_RUNS" --audit "$TASK_RUNS/input-audit" --config "$MAGSET_SOURCE/configs/breakhis-matched-three-way-v1.json" --code-sha "$MAGSET_CODE_SHA" --output "$TASK_RUNS/checked"
