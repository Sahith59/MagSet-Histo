#!/usr/bin/env bash
#SBATCH --job-name=magset-field-crossfit
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:24:00
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}" "${MAGSET_PRIMARY_JOB:?}"
cd "$MAGSET_SOURCE"
[[ "$(cat .source-sha)" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
study="$MAGSET_ROOT/runs/breakhis-field-hierarchy-v1/$MAGSET_PRIMARY_JOB"
args=(--config configs/breakhis-field-hierarchy-v1.json --plan configs/breakhis-field-crossfit-v1.json
      --audit "$MAGSET_ROOT/runs/breakhis-matched-three-way-v1/input-audit/private.json"
      --input-receipt "$study/train-input-receipt.json" --train-fields "$study/train_fields.npz" --code-sha "$MAGSET_CODE_SHA")
if [[ "${MAGSET_ACTION:-}" == collect ]]; then
  "$MAGSET_ROOT/envs/magset/bin/python" -m scripts.collect_field_crossfit "${args[@]}" \
    --shards "$study/crossfit" --output "$study/crossfit-result.json"
else
  : "${SLURM_ARRAY_TASK_ID:?}"
  "$MAGSET_ROOT/envs/magset/bin/python" -m scripts.crossfit_field_hierarchy "${args[@]}" \
    --shard "$SLURM_ARRAY_TASK_ID" --output "$study/crossfit/$SLURM_ARRAY_TASK_ID"
fi
