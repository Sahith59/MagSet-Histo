#!/usr/bin/env bash
#SBATCH --job-name=magset-router-diagnosis
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u scripts/diagnose_router_groups.py --producer-source "$MAGSET_ROOT/sources/a6e6a54e4fac090051837df17b0dea14efca3bb3" --parent "$MAGSET_ROOT/runs/breakhis-matched-three-way-v1" --config "$MAGSET_SOURCE/configs/breakhis-router-diagnosis-v1.json" --output "$MAGSET_ROOT/runs/breakhis-router-diagnosis-v1"
