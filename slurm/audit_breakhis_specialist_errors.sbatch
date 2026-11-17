#!/usr/bin/env bash
#SBATCH --job-name=magset-specialist-errors
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable audit source}"
: "${MAGSET_CODE_SHA:?Set immutable audit source SHA}"
: "${MAGSET_METRIC_COMPARISON_SHA256:?Set accepted metric comparison hash}"
: "${MAGSET_LINEAR_COMPARISON_SHA256:?Set accepted linear comparison hash}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u scripts/audit_breakhis_specialist_errors.py \
 --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
 --metric-comparison "$MAGSET_ROOT/runs/breakhis-specialist-metric-v1/checked/comparison.json" \
 --metric-comparison-sha256 "$MAGSET_METRIC_COMPARISON_SHA256" \
 --metric-runs "$MAGSET_ROOT/runs/breakhis-specialist-metric-v1" \
 --linear-comparison "$MAGSET_ROOT/runs/breakhis-uni-linear-v1/checked/comparison.json" \
 --linear-comparison-sha256 "$MAGSET_LINEAR_COMPARISON_SHA256" \
 --linear-runs "$MAGSET_ROOT/runs/breakhis-uni-linear-v1" \
 --code-sha "$MAGSET_CODE_SHA" --output "$MAGSET_ROOT/runs/breakhis-specialist-errors-v1"
