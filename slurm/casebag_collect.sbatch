#!/usr/bin/env bash
#SBATCH --job-name=magset-casebag-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable collector source}"
: "${MAGSET_TRAINING_SOURCE:?Set frozen b5 training source}"
: "${MAGSET_CODE_SHA:?Set frozen producer SHA}"
: "${MAGSET_COLLECTOR_CODE_SHA:?Set immutable collector SHA}"
[[ "$(cat "$MAGSET_TRAINING_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_COLLECTOR_CODE_SHA" ]]
[[ "$MAGSET_CODE_SHA" == b5f0df18a47c1607be492bdaab7ab6cd2171226b ]]
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
TASK_RUN_ROOT="$MAGSET_ROOT/runs/breakhis-casebag-v1"
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.casebag_collect \
  --runs "$TASK_RUN_ROOT/production" \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" \
  --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
  --weights "$MAGSET_ROOT/cache/casebag-effnetv2b0/model.safetensors" \
  --config "$MAGSET_TRAINING_SOURCE/configs/breakhis-casebag-v1.json" \
  --training-source-root "$MAGSET_TRAINING_SOURCE" \
  --training-source-archive "$MAGSET_ROOT/cache/tmp/$MAGSET_CODE_SHA.tar" \
  --expected-code-sha "$MAGSET_CODE_SHA" \
  --smoke-acceptance "$TASK_RUN_ROOT/technical-acceptance.json" \
  --reference-comparison "$MAGSET_TRAINING_SOURCE/docs/evidence/BREAKHIS_FIVE_CLASS_RESULTS.json" \
  --output "${MAGSET_CASEBAG_COLLECT_OUTPUT:-$TASK_RUN_ROOT/complete-comparison}"
