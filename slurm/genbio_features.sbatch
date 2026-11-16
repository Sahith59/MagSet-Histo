#!/usr/bin/env bash
#SBATCH --job-name=magset-genbio-features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source}"
: "${MAGSET_CODE_SHA:?Set immutable SHA}"
: "${MAGSET_GENBIO_ACTION:?Use smoke or run}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_LIMIT=0
TASK_OUTPUT="$MAGSET_ROOT/cache/features-genbio-v1"
if [[ "$MAGSET_GENBIO_ACTION" == smoke ]]; then
 TASK_MAG=40; TASK_LIMIT=4; TASK_OUTPUT="$MAGSET_ROOT/runs/genbio-feature-smoke-v1"
elif [[ "$MAGSET_GENBIO_ACTION" == run ]]; then
 TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Use four magnification tasks}"
 [[ "$TASK_INDEX" =~ ^[0-9]+$ ]] && (( TASK_INDEX>=0 && TASK_INDEX<4 ))
 TASK_MAGS=(40 100 200 400); TASK_MAG="${TASK_MAGS[$TASK_INDEX]}"
else exit 2; fi
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.genbio_features --config configs/genbio-pathfm-v1.json --cache-root "$MAGSET_ROOT/cache/genbio-pathfm-v1" --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --images-root "$MAGSET_ROOT/data/raw" --output-root "$TASK_OUTPUT" --magnification "$TASK_MAG" --code-sha "$MAGSET_CODE_SHA" --limit "$TASK_LIMIT" --batch-size 4 --workers 0
