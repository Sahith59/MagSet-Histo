#!/usr/bin/env bash
#SBATCH --job-name=magset-notebook-review
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source}"
: "${MAGSET_CODE_SHA:?Set immutable source SHA}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
TASK_ROOT="$MAGSET_ROOT/runs/supplied-cnn-transformer-v1"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/notebook-tf215-v1/bin/python" -u scripts/collect_supplied_cnn_transformer.py \
 --audit "$TASK_ROOT/input-audit" --runs "$TASK_ROOT" \
 --notebook "$MAGSET_ROOT/inputs/supplied-cnn-transformer-v1/original.ipynb" \
 --config "$MAGSET_SOURCE/configs/supplied-cnn-transformer-v1.json" \
 --code-sha "$MAGSET_CODE_SHA" --output "$TASK_ROOT/checked"
