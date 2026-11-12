#!/usr/bin/env bash
#SBATCH --job-name=magset-decode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-3%4
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
TASK_MAGS=(40 100 200 400)
cd "$MAGSET_ROOT/repo"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.validate_images \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --images-root "$MAGSET_ROOT/data/raw" \
  --magnification "${TASK_MAGS[$SLURM_ARRAY_TASK_ID]}" \
  --output "$MAGSET_ROOT/data/processed/decode-${TASK_MAGS[$SLURM_ARRAY_TASK_ID]}.json"
