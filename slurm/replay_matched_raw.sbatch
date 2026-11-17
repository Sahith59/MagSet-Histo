#!/usr/bin/env bash
#SBATCH --job-name=magset-raw-proof
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
TASK_MODELS=(cnn swin); TASK_MODEL="${TASK_MODELS[${SLURM_ARRAY_TASK_ID:?}]}"
TASK_ENV="$MAGSET_ROOT/envs/swin-tf215-v1"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
export TF_USE_LEGACY_KERAS=1 TF_FORCE_GPU_ALLOW_GROWTH=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$MAGSET_ROOT/cache/hf-supplied-swin" MPLCONFIGDIR="$MAGSET_ROOT/cache/matplotlib-matched-proof"
TASK_SITE="$("$TASK_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
TASK_LIBS="$("$TASK_ENV/bin/python" -c 'import glob,sys; print(":".join(sorted(glob.glob(sys.argv[1]+"/nvidia/*/lib"))))' "$TASK_SITE")"
export LD_LIBRARY_PATH="$TASK_LIBS:${LD_LIBRARY_PATH:-}" PATH="$TASK_SITE/nvidia/cuda_nvcc/bin:$PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$TASK_SITE/nvidia/cuda_nvcc"
cd "$MAGSET_SOURCE"
"$TASK_ENV/bin/python" -u scripts/replay_matched_raw.py --model "$TASK_MODEL" --producer-source "$MAGSET_ROOT/sources/a6e6a54e4fac090051837df17b0dea14efca3bb3" --study "$MAGSET_ROOT/runs/breakhis-matched-three-way-v1" --cluster-root "$MAGSET_ROOT" --output "$MAGSET_ROOT/runs/breakhis-matched-raw-proof-v1/$TASK_MODEL"
