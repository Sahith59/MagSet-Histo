#!/usr/bin/env bash
#SBATCH --job-name=magset-notebook-summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable report source}"
: "${MAGSET_CODE_SHA:?Set immutable report source SHA}"
: "${MAGSET_NOTEBOOK_PRODUCER:?Set original notebook producer SHA}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
"$MAGSET_ROOT/envs/magset/bin/python" "$MAGSET_SOURCE/scripts/summarize_supplied_notebook.py" \
 --comparison "$MAGSET_ROOT/runs/supplied-cnn-transformer-v1/checked/comparison.json" \
 --output "$MAGSET_ROOT/runs/supplied-cnn-transformer-v1/summary.md" \
 --expected-producer "$MAGSET_NOTEBOOK_PRODUCER"
