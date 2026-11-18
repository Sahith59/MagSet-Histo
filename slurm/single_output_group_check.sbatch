#!/usr/bin/env bash
#SBATCH --job-name=magset-one-check
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:12:00
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}" "${MAGSET_FIT_JOB:?}" "${SLURM_JOB_ID:?}"
cd "$MAGSET_SOURCE"
[[ "$(cat .source-sha)" == "$MAGSET_CODE_SHA" && "$MAGSET_FIT_JOB" =~ ^[0-9]+$ ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
python_bin="$MAGSET_ROOT/envs/magset/bin/python"
study="$MAGSET_ROOT/runs/breakhis-single-output-center-v1"
output="$MAGSET_ROOT/runs/breakhis-single-output-group-v1/$SLURM_JOB_ID"
mkdir -p "$output"
inputs=(--config configs/breakhis-single-output-center-v1.json
        --audit "$MAGSET_ROOT/runs/breakhis-matched-three-way-v1/input-audit/private.json"
        --views "$MAGSET_ROOT/runs/breakhis-uni-case-svm-v1/preprocessing.npz")
"$python_bin" -m scripts.verify_single_output_coverage "${inputs[@]}" --run "$study/$MAGSET_FIT_JOB" --output "$output/replay.json"
"$python_bin" -m scripts.crossfit_single_output "${inputs[@]}" --output "$output/check"
