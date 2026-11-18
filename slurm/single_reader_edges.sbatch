#!/usr/bin/env bash
#SBATCH --job-name=magset-crop-probe
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:12:00
#SBATCH --gres=gpu:A40:1
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}" "${SLURM_JOB_ID:?}"
cd "$MAGSET_SOURCE"
[[ "$(cat .source-sha)" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
packet_dir="$MAGSET_ROOT/repo/.local/shared-hierarchy-v1"
"$MAGSET_ROOT/envs/magset/bin/python" -m scripts.probe_single_reader_edges \
  --fields "$packet_dir/private-review/fields" \
  --centers "$MAGSET_ROOT/repo/.local/field-hierarchy-v1/crop-review/centers" \
  --key "$packet_dir/review-key-private.csv" \
  --config configs/uni2-h-v1.json --weights "$MAGSET_ROOT/cache/uni2-h-v1" \
  --cache "$MAGSET_ROOT/cache/features-uni2-h-v1" \
  --output "$MAGSET_ROOT/runs/breakhis-crop-probe-v1/$SLURM_JOB_ID"
