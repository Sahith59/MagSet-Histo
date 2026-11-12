#!/usr/bin/env bash
#SBATCH --job-name=magset-features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-7%4
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_CODE_SHA:?Set MAGSET_CODE_SHA to the source commit}"
export HF_HOME="$MAGSET_ROOT/cache/huggingface"
export TORCH_HOME="$MAGSET_ROOT/cache/torch"
export TMPDIR="$MAGSET_ROOT/cache/tmp"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
TASK_MAGS=(40 100 200 400)
TASK_INDEX="${SLURM_ARRAY_TASK_ID}"
if (( TASK_INDEX < 4 )); then TASK_ENCODER=phikon-v2; else TASK_ENCODER=resnet50; fi
TASK_MAG="${TASK_MAGS[$((TASK_INDEX % 4))]}"
cd "$MAGSET_ROOT/repo"
"$MAGSET_ROOT/envs/magset/bin/python" -m magset.features \
  --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --images-root "$MAGSET_ROOT/data/raw" \
  --output-root "$MAGSET_ROOT/cache/features" --encoder "$TASK_ENCODER" --magnification "$TASK_MAG" \
  --batch-size 16 --workers 4 --code-sha "$MAGSET_CODE_SHA"
