#!/usr/bin/env bash
#SBATCH --job-name=magset-collect-seeds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
cd "$MAGSET_ROOT/repo"
TASK_PYTHON="$MAGSET_ROOT/envs/magset/bin/python"
"$TASK_PYTHON" -m magset.collect --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" --runs "$MAGSET_ROOT/runs/fusion-sensitivity-v1/split813-train812" \
  --encoders phikon-v2 --seed 813 \
  --methods field-mean-view-mean field-mean-view-gated field-mean-view-attention \
    field-attention-view-mean field-attention-view-gated field-attention-view-attention \
  --output "$MAGSET_ROOT/runs/fusion-sensitivity-v1/split813-train812/verified-summary.json"
"$TASK_PYTHON" -m magset.collect --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" --runs "$MAGSET_ROOT/runs/fusion-sensitivity-v1/split812-train813" \
  --encoders phikon-v2 --seed 812 --methods field-mean-view-mean field-attention-view-attention \
  --output "$MAGSET_ROOT/runs/fusion-sensitivity-v1/split812-train813/verified-summary.json"
"$TASK_PYTHON" -m magset.collect --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" --runs "$MAGSET_ROOT/runs/linear-sensitivity-v1" \
  --encoders phikon-v2 --seed 813 --methods mean \
  --output "$MAGSET_ROOT/runs/linear-sensitivity-v1/verified-summary.json"
