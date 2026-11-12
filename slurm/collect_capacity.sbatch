#!/usr/bin/env bash
#SBATCH --job-name=magset-collect-capacity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$MAGSET_ROOT/repo"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.capacity_report \
  --runs-root "$MAGSET_ROOT/runs" --output "$MAGSET_ROOT/runs/capacity-v1/verified-comparison.json" \
  --markdown "$MAGSET_ROOT/runs/capacity-v1/report.md"
