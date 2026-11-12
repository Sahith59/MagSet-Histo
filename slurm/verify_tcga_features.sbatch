#!/usr/bin/env bash
#SBATCH --job-name=magset-tcga-cache-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_SOURCE:?Set immutable MAGSET_SOURCE}"
: "${MAGSET_CODE_SHA:?Set MAGSET_CODE_SHA}"
: "${MAGSET_PROTOCOL_SHA:?Set fixed protocol SHA}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.verify_native \
 --manifest "$MAGSET_ROOT/data/tcga-ms/quarantine-v1/manifest.csv" \
 --protocol "$MAGSET_ROOT/data/tcga-ms/quarantine-v1/protocol.json" \
 --cache "$MAGSET_ROOT/cache/tcga-quarantine-resnet-v1" \
 --expected-protocol-sha "$MAGSET_PROTOCOL_SHA" \
 --output "$MAGSET_ROOT/cache/tcga-quarantine-resnet-v1/verified-cache-audit.json"
