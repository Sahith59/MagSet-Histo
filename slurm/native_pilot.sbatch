#!/usr/bin/env bash
#SBATCH --job-name=magset-native-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_SOURCE:?Set immutable MAGSET_SOURCE}"
: "${MAGSET_CODE_SHA:?Set MAGSET_CODE_SHA}"
: "${MAGSET_PROTOCOL_SHA:?Set registered protocol SHA}"
: "${MAGSET_AUDIT_SHA:?Set independently verified cache audit SHA}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.native_pilot \
 --manifest "$MAGSET_ROOT/data/tcga-ms/quarantine-v1/manifest.csv" \
 --protocol "$MAGSET_ROOT/data/tcga-ms/quarantine-v1/protocol.json" \
 --cache "$MAGSET_ROOT/cache/tcga-quarantine-resnet-v1" \
 --audit "$MAGSET_ROOT/cache/tcga-quarantine-resnet-v1/verified-cache-audit.json" \
 --expected-protocol-sha "$MAGSET_PROTOCOL_SHA" --expected-audit-sha "$MAGSET_AUDIT_SHA" \
 --output "$MAGSET_ROOT/runs/native-development-v1" --code-sha "$MAGSET_CODE_SHA"
