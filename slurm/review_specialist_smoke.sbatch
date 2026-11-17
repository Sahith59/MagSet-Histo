#!/usr/bin/env bash
#SBATCH --job-name=magset-specialist-review
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source}"
: "${MAGSET_CODE_SHA:?Set immutable source SHA}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u scripts/review_specialist_smoke.py --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" --cache "$MAGSET_ROOT/cache/features-uni2-h-v1" --center-cache "$MAGSET_ROOT/cache/features-midnight-v1" --feature-audit "$MAGSET_ROOT/cache/approved-audits/uni2-h-v1.json" --config "$MAGSET_SOURCE/configs/breakhis-specialist-metric-v1.json" --reference-runs "$MAGSET_ROOT/runs/breakhis-uni2-h-v1-kernel" --code-sha "$MAGSET_CODE_SHA" --audit "$MAGSET_ROOT/runs/breakhis-specialist-metric-v1/input-audit" --smoke-root "$MAGSET_ROOT/runs/breakhis-specialist-metric-smoke-v1" --out "$MAGSET_ROOT/runs/breakhis-specialist-metric-smoke-review-v1.json"
