#!/usr/bin/env bash
#SBATCH --job-name=magset-regional-compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source directory}"
: "${MAGSET_CODE_SHA:?Set immutable source SHA}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
TASK_CACHE="$MAGSET_ROOT/cache/breakhis-multicrop-v1/assembled"
mkdir "$TASK_CACHE"
for TASK_MAG in 40 100 200 400; do
  cp "$MAGSET_ROOT/cache/breakhis-multicrop-v1/mag$TASK_MAG/phikon-v2-$TASK_MAG.npz" "$TASK_CACHE/"
  cp "$MAGSET_ROOT/cache/breakhis-multicrop-v1/mag$TASK_MAG/phikon-v2-$TASK_MAG.json" "$TASK_CACHE/"
done
# The evaluator verifies every copied shard, provenance chain and blend before fitting.
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.multicrop_compare \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
  --thumbnail-cache "$MAGSET_ROOT/cache/features" --multicrop-cache "$TASK_CACHE" \
  --reference-root "$MAGSET_ROOT/runs/matched-mean-v1" --config configs/breakhis-multicrop-v1.json \
  --output "$MAGSET_ROOT/runs/breakhis-multicrop-v1/comparison" --code-sha "$MAGSET_CODE_SHA"
