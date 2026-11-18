#!/usr/bin/env bash
#SBATCH --job-name=magset-field-hierarchy
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}" "${SLURM_JOB_ID:?}"
cd "$MAGSET_SOURCE"
[[ "$(cat .source-sha)" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
"$MAGSET_ROOT/envs/magset/bin/python" -m scripts.run_field_hierarchy \
  --config configs/breakhis-field-hierarchy-v1.json \
  --audit "$MAGSET_ROOT/runs/breakhis-matched-three-way-v1/input-audit/private.json" \
  --views "$MAGSET_ROOT/runs/breakhis-uni-case-svm-v1/preprocessing.npz" \
  --field-cache "$MAGSET_ROOT/cache/features-uni2-h-v1" \
  --output "$MAGSET_ROOT/runs/breakhis-field-hierarchy-v1/$SLURM_JOB_ID" \
  --code-sha "$MAGSET_CODE_SHA"
