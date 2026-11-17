#!/usr/bin/env bash
#SBATCH --job-name=magset-midnight-raw-proof
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}" "${MAGSET_RAW_PROOF_ACTION:?}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TASK_ROOT="$MAGSET_ROOT/runs/breakhis-midnight-raw-proof-v1"
TASK_PRODUCER="$MAGSET_ROOT/sources/a6e6a54e4fac090051837df17b0dea14efca3bb3"
cd "$MAGSET_SOURCE"
if [[ "$MAGSET_RAW_PROOF_ACTION" == extract ]]; then
 TASK_MAGS=(40 100 200 400); TASK_MAG="${TASK_MAGS[${SLURM_ARRAY_TASK_ID:?}]}"
 "$MAGSET_ROOT/envs/magset/bin/python" -u scripts/replay_midnight_raw_features.py --producer-source "$TASK_PRODUCER" --study "$MAGSET_ROOT/runs/breakhis-matched-three-way-v1" --model-cache "$MAGSET_ROOT/cache/midnight-12k" --magnification "$TASK_MAG" --output "$TASK_ROOT/$TASK_MAG"
elif [[ "$MAGSET_RAW_PROOF_ACTION" == collect ]]; then
 export CUDA_VISIBLE_DEVICES=""
 "$MAGSET_ROOT/envs/magset/bin/python" -u scripts/collect_midnight_raw_proof.py --producer-source "$TASK_PRODUCER" --study "$MAGSET_ROOT/runs/breakhis-matched-three-way-v1" --features-root "$TASK_ROOT" --output "$TASK_ROOT/checked"
else exit 2; fi
