#!/usr/bin/env bash
#SBATCH --job-name=magset-specialist-reader
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}" "${MAGSET_SPECIALIST_ACTION:?}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
TASK_STUDY="$MAGSET_ROOT/runs/breakhis-specialist-reader-v1"
TASK_PARENT="$MAGSET_ROOT/runs/breakhis-matched-three-way-v1"
cd "$MAGSET_SOURCE"
if [[ "$MAGSET_SPECIALIST_ACTION" == run ]]; then
 "$MAGSET_ROOT/envs/magset/bin/python" -u scripts/run_specialist_reader.py --parent "$TASK_PARENT" --p-cache "$MAGSET_ROOT/cache/features-midnight-coverage-v1" --uni-cache "$MAGSET_ROOT/cache/features-uni2-h-v1" --config "$MAGSET_SOURCE/configs/breakhis-specialist-reader-v1.json" --output "$TASK_STUDY" --code-sha "$MAGSET_CODE_SHA"
elif [[ "$MAGSET_SPECIALIST_ACTION" == collect ]]; then
 "$MAGSET_ROOT/envs/magset/bin/python" -u scripts/collect_specialist_reader.py --root "$TASK_STUDY" --parent "$TASK_PARENT" --config "$MAGSET_SOURCE/configs/breakhis-specialist-reader-v1.json" --code-sha "$MAGSET_CODE_SHA" --output "$TASK_STUDY/checked"
else exit 2; fi
