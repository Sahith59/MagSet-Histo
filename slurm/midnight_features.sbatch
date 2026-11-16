#!/usr/bin/env bash
#SBATCH --job-name=magset-midnight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source}"
: "${MAGSET_CODE_SHA:?Set source SHA}"
: "${MAGSET_MIDNIGHT_ACTION:?Use prepare, smoke or extract}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
TASK_CONFIG="$MAGSET_SOURCE/configs/midnight-12k-v1.json"
TASK_CACHE="$MAGSET_ROOT/cache/midnight-12k"
cd "$MAGSET_SOURCE"
if [[ "$MAGSET_MIDNIGHT_ACTION" == prepare ]]; then
 "$MAGSET_ROOT/envs/magset/bin/python" -u scripts/prepare_midnight.py --config "$TASK_CONFIG" --cache-root "$TASK_CACHE" --code-sha "$MAGSET_CODE_SHA"
 exit
fi
TASK_ARGS=(--config "$TASK_CONFIG" --cache-root "$TASK_CACHE" --manifest "$MAGSET_ROOT/data/processed/manifest.csv"
 --images-root "$MAGSET_ROOT/data/raw" --code-sha "$MAGSET_CODE_SHA" --batch-size 8 --workers 0 --device cuda)
case "$MAGSET_MIDNIGHT_ACTION" in
 smoke) TASK_ARGS+=(--magnification 40 --limit 4 --output-root "$MAGSET_ROOT/cache/midnight-smoke-v1") ;;
 extract)
  TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Need four magnification shards}"
  [[ "$TASK_INDEX" =~ ^[0-3]$ ]]
  TASK_VIEWS=(40 100 200 400)
  TASK_ARGS+=(--magnification "${TASK_VIEWS[$TASK_INDEX]}" --output-root "$MAGSET_ROOT/cache/features-midnight-v1") ;;
 *) exit 2 ;;
esac
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.midnight_features "${TASK_ARGS[@]}"
