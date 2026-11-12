#!/usr/bin/env bash
#SBATCH --job-name=magset-collect-round2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
cd "$MAGSET_ROOT/repo"
case "${1:-}" in
  field)
    "$MAGSET_ROOT/envs/magset/bin/python" -m magset.collect \
      --runs "$MAGSET_ROOT/runs/field-linear-v1" --methods field-late field-early-late \
      --output "$MAGSET_ROOT/runs/field-linear-v1/verified-summary.json"
    ;;
  fusion)
    "$MAGSET_ROOT/envs/magset/bin/python" -m magset.collect \
      --runs "$MAGSET_ROOT/runs/fusion-v1" --encoders phikon-v2 \
      --methods field-mean-view-mean field-mean-view-gated field-mean-view-attention \
        field-attention-view-mean field-attention-view-gated field-attention-view-attention \
      --output "$MAGSET_ROOT/runs/fusion-v1/verified-summary.json"
    ;;
  *) echo 'Expected field or fusion collection stage' >&2; exit 2 ;;
esac
