#!/usr/bin/env bash
#SBATCH --job-name=magset-final-bench
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:10:00
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}" "${MAGSET_FIT_JOB:?}" "${SLURM_JOB_ID:?}"
cd "$MAGSET_SOURCE"
[[ "$(cat .source-sha)" == "$MAGSET_CODE_SHA" && "$MAGSET_FIT_JOB" =~ ^[0-9]+$ ]]
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
matched="$MAGSET_ROOT/runs/breakhis-matched-three-way-v1"
"$MAGSET_ROOT/envs/magset/bin/python" -m scripts.benchmark_single_output \
  --audit "$matched/input-audit/private.json" --comparators "$matched/models" \
  --views "$MAGSET_ROOT/runs/breakhis-uni-case-svm-v1/preprocessing.npz" \
  --run "$MAGSET_ROOT/runs/breakhis-single-output-center-v1/$MAGSET_FIT_JOB" \
  --output "$MAGSET_ROOT/runs/breakhis-single-output-benchmark-v1/$SLURM_JOB_ID"
