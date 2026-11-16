#!/usr/bin/env bash
#SBATCH --job-name=magset-casebag-review
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable independent reviewer source}"
: "${MAGSET_REVIEW_CODE_SHA:?Set independent reviewer full commit}"
: "${MAGSET_TRAINING_SOURCE:?Set immutable b5 training source}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_REVIEW_CODE_SHA" ]]
[[ "$(cat "$MAGSET_TRAINING_SOURCE/.source-sha")" == b5f0df18a47c1607be492bdaab7ab6cd2171226b ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
TASK_RUN_ROOT="$MAGSET_ROOT/runs/breakhis-casebag-v1"
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u scripts/review_casebag_results.py \
  --runs "$TASK_RUN_ROOT/production" \
  --comparison "$TASK_RUN_ROOT/complete-comparison/comparison.json" \
  --verification "$TASK_RUN_ROOT/complete-comparison/verification.json" \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" \
  --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
  --training-source-root "$MAGSET_TRAINING_SOURCE" \
  --training-source-archive "$MAGSET_ROOT/cache/tmp/b5f0df18a47c1607be492bdaab7ab6cd2171226b.tar" \
  --output "${MAGSET_CASEBAG_REVIEW_OUTPUT:-$TASK_RUN_ROOT/independent-review.json}"
