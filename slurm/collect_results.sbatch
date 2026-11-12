#!/usr/bin/env bash
#SBATCH --job-name=magset-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
cd "$MAGSET_ROOT/repo"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.collect \
  --runs "$MAGSET_ROOT/runs/validation-v1" \
  --output "$MAGSET_ROOT/runs/validation-v1/verified-summary.json"
