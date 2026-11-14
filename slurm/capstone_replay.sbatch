#!/usr/bin/env bash
# Reuse the retained head; this is inference/replay, never model fitting.
#SBATCH --job-name=magset-capstone-replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set the immutable prediction source}"
: "${MAGSET_CODE_SHA:?Set the prediction source SHA}"
: "${MAGSET_DEMO_OUTPUT:?Set a new output JSON path}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export HF_HOME="$MAGSET_ROOT/cache/huggingface" TORCH_HOME="$MAGSET_ROOT/cache/torch"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" -u -m magset.breakhis_predict \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --protocol "$MAGSET_ROOT/data/processed/protocol.json" \
  --run-dir "$MAGSET_ROOT/runs/matched-mean-v1/split812/phikon-v2-mean-trial1-seed812" \
  --cache "$MAGSET_ROOT/cache/features" --images-root "$MAGSET_ROOT/data/raw" \
  --expected-model-sha 318a38488b84845b0731be38fe7d8d55e46739526f0ef196a4ba480152db8f21 \
  --expected-protocol-sha dca9a93bb570fd7cfec9fe72e5cad9ac76f838725b03c7a7a593824fecd51134 \
  --code-sha "$MAGSET_CODE_SHA" --output "$MAGSET_DEMO_OUTPUT"
