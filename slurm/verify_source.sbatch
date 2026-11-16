#!/usr/bin/env bash
#SBATCH --job-name=magset-source-checks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source}"
: "${MAGSET_CODE_SHA:?Set immutable SHA}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -m unittest discover -s tests -v
