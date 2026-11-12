#!/usr/bin/env bash
#SBATCH --job-name=magset-tcga-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_CODE_SHA:?Set MAGSET_CODE_SHA}"
cd "$MAGSET_ROOT/repo"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.archive_audit \
  --source configs/tcga-ms-source.json --data-root "$MAGSET_ROOT/data/tcga-ms" \
  --code-sha "$MAGSET_CODE_SHA"
