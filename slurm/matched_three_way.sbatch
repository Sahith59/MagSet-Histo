#!/usr/bin/env bash
#SBATCH --job-name=magset-matched-model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}" "${MAGSET_MATCHED_ACTION:?}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
TASK_RUNS="$MAGSET_ROOT/runs/breakhis-matched-three-way-v1"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
export TF_USE_LEGACY_KERAS=1 TF_FORCE_GPU_ALLOW_GROWTH=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$MAGSET_ROOT/cache/hf-supplied-swin" MPLCONFIGDIR="$MAGSET_ROOT/cache/matplotlib-matched-three-way"
TASK_ENV="$MAGSET_ROOT/envs/swin-tf215-v1"
TASK_MODEL=cnn
case "$MAGSET_MATCHED_ACTION" in
 smoke|train|test)
  TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Use the two-model array}"
  case "$TASK_INDEX" in 0) TASK_MODEL=cnn ;; 1) TASK_MODEL=swin ;; *) exit 2 ;; esac
  TASK_SITE="$("$TASK_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  TASK_LIBS="$("$TASK_ENV/bin/python" -c 'import glob,sys; print(":".join(sorted(glob.glob(sys.argv[1]+"/nvidia/*/lib"))))' "$TASK_SITE")"
  export LD_LIBRARY_PATH="$TASK_LIBS:${LD_LIBRARY_PATH:-}" PATH="$TASK_SITE/nvidia/cuda_nvcc/bin:$PATH"
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=$TASK_SITE/nvidia/cuda_nvcc"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  case "$MAGSET_MATCHED_ACTION" in
   smoke) TASK_OUTPUT="$TASK_RUNS/technical/$TASK_MODEL" ;;
   train) TASK_OUTPUT="$TASK_RUNS/models/$TASK_MODEL" ;;
   test) TASK_OUTPUT="$TASK_RUNS/models/$TASK_MODEL/test" ;;
  esac ;;
 gate|p-select|p-test)
  export CUDA_VISIBLE_DEVICES=""
  TASK_ENV="$MAGSET_ROOT/envs/magset"
  case "$MAGSET_MATCHED_ACTION" in
   gate) TASK_OUTPUT="$TASK_RUNS/gate-operation" ;;
   p-select) TASK_OUTPUT="$TASK_RUNS/models/midnight_p" ;;
   p-test) TASK_OUTPUT="$TASK_RUNS/models/midnight_p/test" ;;
  esac ;;
 *) exit 2 ;;
esac
cd "$MAGSET_SOURCE"
"$TASK_ENV/bin/python" -u scripts/run_matched_three_way.py "$MAGSET_MATCHED_ACTION" --root "$TASK_RUNS" --audit "$TASK_RUNS/input-audit" --pixels "$TASK_RUNS/pixels" --cache "$MAGSET_ROOT/cache/features-midnight-coverage-v1" --config "$MAGSET_SOURCE/configs/breakhis-matched-three-way-v1.json" --output "$TASK_OUTPUT" --model "$TASK_MODEL" --code-sha "$MAGSET_CODE_SHA" --cnn-weights "$MAGSET_ROOT/cache/keras-notebook-tf215/models/efficientnetv2-b0_notop.h5" --swin-prepared "$MAGSET_ROOT/envs/swin-tf215-v1/prepared.json"
