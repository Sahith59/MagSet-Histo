#!/usr/bin/env bash
#SBATCH --job-name=magset-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT to the shared project directory}"
export TMPDIR="$MAGSET_ROOT/cache/tmp"
mkdir -p "$TMPDIR"
cd "$MAGSET_ROOT/repo"
python3 -u scripts/prepare_breakhis.py --data-root "$MAGSET_ROOT/data" --download
