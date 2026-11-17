#!/usr/bin/env bash
#SBATCH --job-name=magset-swin-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/swin-tf215-v1/bin/python" -u scripts/collect_supplied_swin.py --run "$MAGSET_ROOT/runs/supplied-swin-v1/attempt-3/dual-split0" --audit "$MAGSET_ROOT/runs/supplied-swin-v1/attempt-3/input-audit" --config "$MAGSET_SOURCE/configs/supplied-swin-v1.json" --code-sha "$MAGSET_CODE_SHA" --output "$MAGSET_ROOT/runs/supplied-swin-v1/attempt-3/checked"
