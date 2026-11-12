#!/usr/bin/env bash
#SBATCH --job-name=magset-tcga-decode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --array=0-6%4
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_CODE_SHA:?Set MAGSET_CODE_SHA}"
cd "$MAGSET_ROOT/repo"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.prepare_tcga --stage decode \
  --data-root "$MAGSET_ROOT/data/tcga-ms" --source configs/tcga-ms-source.json \
  --rank "$SLURM_ARRAY_TASK_ID" --code-sha "$MAGSET_CODE_SHA"
