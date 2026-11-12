#!/usr/bin/env bash
#SBATCH --job-name=magset-tcga-features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%4
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_SOURCE:?Set immutable MAGSET_SOURCE}"
: "${MAGSET_CODE_SHA:?Set MAGSET_CODE_SHA}"
: "${MAGSET_PROTOCOL_SHA:?Set immutable protocol hash}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export TORCH_HOME="$MAGSET_ROOT/cache/torch" TMPDIR="$MAGSET_ROOT/cache/tmp"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
TASK_RESOLUTIONS=(0.25 0.5 1.0 2.0)
TASK_LIMIT=0
TASK_OUTPUT="$MAGSET_ROOT/cache/tcga-quarantine-resnet-v1"
if [[ "${MAGSET_SMOKE:-0}" == 1 ]]; then TASK_LIMIT=16; TASK_OUTPUT="$MAGSET_ROOT/cache/tcga-quarantine-smoke-v1"; fi
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.tcga_features \
  --manifest "$MAGSET_ROOT/data/tcga-ms/quarantine-v1/manifest.csv" \
  --protocol "$MAGSET_ROOT/data/tcga-ms/quarantine-v1/protocol.json" \
  --images-root "$MAGSET_ROOT/data/tcga-ms/raw" --output-root "$TASK_OUTPUT" \
  --expected-protocol-sha "$MAGSET_PROTOCOL_SHA" \
  --resolution "${TASK_RESOLUTIONS[$SLURM_ARRAY_TASK_ID]}" --limit "$TASK_LIMIT" --code-sha "$MAGSET_CODE_SHA"
