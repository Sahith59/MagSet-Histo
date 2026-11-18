#!/usr/bin/env bash
#SBATCH --job-name=magset-wide-views
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:35:00
set -euo pipefail
cd "${MAGSET_SOURCE:?}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "${MAGSET_ACTION:?}" == extract ]]; then
  zooms=(40 100 200 400)
  args=(--zoom "${zooms[${SLURM_ARRAY_TASK_ID:?}]}")
  run_id="${SLURM_ARRAY_JOB_ID:?}"
elif [[ "$MAGSET_ACTION" == collect ]]; then
  args=()
  run_id="${MAGSET_FEATURE_JOB:?}"
else
  exit 2
fi
"${MAGSET_ROOT:?}/envs/magset/bin/python" -u -m scripts.extract_wide_uni_views "${args[@]}" \
  --audit "$MAGSET_ROOT/runs/breakhis-matched-three-way-v1/input-audit/private.json" \
  --views "$MAGSET_ROOT/runs/breakhis-uni-case-svm-v1/preprocessing.npz" \
  --weights "$MAGSET_ROOT/cache/uni2-h-v1" --cache "$MAGSET_ROOT/cache/features-uni2-h-v1" \
  --images "$MAGSET_ROOT/data/raw" --output "$MAGSET_ROOT/runs/breakhis-wide-uni-v1/$run_id"
