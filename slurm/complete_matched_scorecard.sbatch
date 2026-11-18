#!/usr/bin/env bash
#SBATCH --job-name=magset-comparison
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:12:00
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}" "${SLURM_JOB_ID:?}"
cd "$MAGSET_SOURCE"
[[ "$(cat .source-sha)" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
"$MAGSET_ROOT/envs/magset/bin/python" -m scripts.complete_matched_scorecard \
  --audit "$MAGSET_ROOT/runs/breakhis-matched-three-way-v1/input-audit/private.json" \
  --views "$MAGSET_ROOT/runs/breakhis-uni-case-svm-v1/preprocessing.npz" \
  --field-cache "$MAGSET_ROOT/cache/features-uni2-h-v1" \
  --old-run "$MAGSET_ROOT/runs/breakhis-field-hierarchy-v1/${MAGSET_FIELD_JOB:?}" \
  --prior-group "$MAGSET_ROOT/runs/breakhis-single-output-group-v1/${MAGSET_PRIOR_GROUP_JOB:?}/check/predictions.npz" \
  --output "$MAGSET_ROOT/runs/breakhis-matched-scorecard-v1/$SLURM_JOB_ID"
