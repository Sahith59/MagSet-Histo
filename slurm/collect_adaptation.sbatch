#!/usr/bin/env bash
#SBATCH --job-name=magset-adaptation-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_SOURCE:?Set immutable collector source}"
: "${MAGSET_CODE_SHA:?Set full collector source SHA}"
: "${MAGSET_TRAINING_SOURCE:?Set immutable training source}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.collect_adaptation \
  --runs "$MAGSET_ROOT/runs/breakhis-adaptation-v1" \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
  --reference-root "$MAGSET_ROOT/runs/matched-mean-v1" --training-source "$MAGSET_TRAINING_SOURCE" \
  --output "$MAGSET_ROOT/runs/breakhis-adaptation-v1/paired-comparison.json"
