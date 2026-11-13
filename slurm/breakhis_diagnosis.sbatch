#!/usr/bin/env bash
#SBATCH --job-name=magset-breakhis-diagnose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable diagnosis source}"
: "${MAGSET_CODE_SHA:?Set full diagnosis source SHA}"
: "${MAGSET_TRAINING_SOURCE:?Set original immutable training source}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.diagnose_breakhis \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
  --reference-root "$MAGSET_ROOT/runs/matched-mean-v1" --runs "$MAGSET_ROOT/runs/breakhis-adaptation-v1" \
  --frozen-cache "$MAGSET_ROOT/cache/features" --training-source "$MAGSET_TRAINING_SOURCE" \
  --config configs/breakhis-diagnosis-v1.json --code-sha "$MAGSET_CODE_SHA" \
  --output "$MAGSET_ROOT/runs/breakhis-diagnosis-v1"
