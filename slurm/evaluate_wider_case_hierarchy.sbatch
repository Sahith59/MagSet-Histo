#!/usr/bin/env bash
#SBATCH --job-name=magset-wider-fit
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
cd "${MAGSET_SOURCE:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
"${MAGSET_ROOT:?}/envs/magset/bin/python" -m scripts.evaluate_wider_case_hierarchy \
  --features "${MAGSET_FEATURE_BUNDLE:?}" \
  --audit "$MAGSET_ROOT/runs/breakhis-matched-three-way-v1/input-audit/private.json" \
  --original-views "$MAGSET_ROOT/runs/breakhis-uni-case-svm-v1/preprocessing.npz" \
  --previous-runs "$MAGSET_ROOT/runs/breakhis-subtype-initialization-v1" \
  --output "$MAGSET_ROOT/runs/breakhis-wider-hierarchy-v1/${SLURM_JOB_ID:?}"
