#!/usr/bin/env bash
#SBATCH --job-name=magset-fusion-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_CODE_SHA:?Set MAGSET_CODE_SHA}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$MAGSET_ROOT/repo"
for TASK_FIELD in mean attention; do
  for TASK_VIEW in mean gated attention; do
    "$MAGSET_ROOT/envs/magset/bin/python" -m magset.train_fusion \
      --manifest "$MAGSET_ROOT/data/processed/manifest.csv" \
      --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
      --features "$MAGSET_ROOT/cache/features" --encoder phikon-v2 \
      --field-pool "$TASK_FIELD" --view-fusion "$TASK_VIEW" \
      --config configs/fusion-smoke.json --trial 1 --seed 812 --training-seed 812 \
      --code-sha "$MAGSET_CODE_SHA" \
      --output "$MAGSET_ROOT/runs/fusion-smoke-v1/field-$TASK_FIELD-view-$TASK_VIEW"
  done
done
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.collect \
  --runs "$MAGSET_ROOT/runs/fusion-smoke-v1" --encoders phikon-v2 \
  --methods field-mean-view-mean field-mean-view-gated field-mean-view-attention \
    field-attention-view-mean field-attention-view-gated field-attention-view-attention \
  --trials 1 --output "$MAGSET_ROOT/runs/fusion-smoke-v1/verified-summary.json"
# Two epochs verify the execution path only; these scores are not benchmark evidence.
