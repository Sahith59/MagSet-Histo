#!/usr/bin/env bash
#SBATCH --job-name=magset-weight-val
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:12:00
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}" "${SLURM_JOB_ID:?}"
cd "$MAGSET_SOURCE"
[[ "$(cat .source-sha)" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
"$MAGSET_ROOT/envs/magset/bin/python" -m scripts.validate_branch_balanced_output \
  --audit "$MAGSET_ROOT/runs/breakhis-matched-three-way-v1/input-audit/private.json" \
  --views "$MAGSET_ROOT/runs/breakhis-uni-case-svm-v1/preprocessing.npz" \
  --output "$MAGSET_ROOT/runs/breakhis-weight-validation-v1/$SLURM_JOB_ID"
