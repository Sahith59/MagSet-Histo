#!/usr/bin/env bash
#SBATCH --job-name=magset-subtype-init
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:12:00
set -euo pipefail
cd "${MAGSET_SOURCE:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
args=()
if [[ "${MAGSET_ACTION:-primary}" == stability ]]; then
  args+=(--stability-check)
fi
"${MAGSET_ROOT:?}/envs/magset/bin/python" -m scripts.run_subtype_initialization \
  "${args[@]}" \
  --audit "$MAGSET_ROOT/runs/breakhis-matched-three-way-v1/input-audit/private.json" \
  --views "$MAGSET_ROOT/runs/breakhis-uni-case-svm-v1/preprocessing.npz" \
  --output "$MAGSET_ROOT/runs/breakhis-subtype-initialization-v1/${SLURM_JOB_ID:?}"
