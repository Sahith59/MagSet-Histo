#!/usr/bin/env bash
#SBATCH --job-name=magset-breakhis-demo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_SOURCE:?Set immutable MAGSET_SOURCE}"
: "${MAGSET_CODE_SHA:?Set MAGSET_CODE_SHA}"
: "${MAGSET_MODEL_SHA:?Set saved head checksum}"
: "${MAGSET_PROTOCOL_SHA:?Set BreakHis protocol checksum}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export HF_HOME="$MAGSET_ROOT/cache/huggingface" TORCH_HOME="$MAGSET_ROOT/cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.breakhis_predict \
 --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
 --run-dir "$MAGSET_ROOT/runs/matched-mean-v1/split812/phikon-v2-mean-trial1-seed812" \
 --cache "$MAGSET_ROOT/cache/features" --images-root "$MAGSET_ROOT/data/raw" \
 --expected-model-sha "$MAGSET_MODEL_SHA" --expected-protocol-sha "$MAGSET_PROTOCOL_SHA" \
 --output "$MAGSET_ROOT/runs/breakhis-milestone-v1/raw-image-demo.json" --code-sha "$MAGSET_CODE_SHA"
