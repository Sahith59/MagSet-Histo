#!/usr/bin/env bash
#SBATCH --job-name=magset-one-output
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}" "${SLURM_JOB_ID:?}"
cd "$MAGSET_SOURCE"
[[ "$(cat .source-sha)" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
python_bin="$MAGSET_ROOT/envs/magset/bin/python"
run_dir="$MAGSET_ROOT/runs/breakhis-single-output-center-v1/$SLURM_JOB_ID"
inputs=(--config configs/breakhis-single-output-center-v1.json
        --audit "$MAGSET_ROOT/runs/breakhis-matched-three-way-v1/input-audit/private.json"
        --views "$MAGSET_ROOT/runs/breakhis-uni-case-svm-v1/preprocessing.npz")
"$python_bin" -m scripts.train_single_output_coverage "${inputs[@]}" --output "$run_dir" --code-sha "$MAGSET_CODE_SHA"
"$python_bin" scripts/verify_single_output_coverage.py "${inputs[@]}" --run "$run_dir" --output "$run_dir/replay.json"
