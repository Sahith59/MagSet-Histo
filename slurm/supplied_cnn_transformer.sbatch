#!/usr/bin/env bash
#SBATCH --job-name=magset-supplied-notebook
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set immutable source}"
: "${MAGSET_CODE_SHA:?Set immutable source SHA}"
: "${MAGSET_NOTEBOOK_ACTION:?Use audit, smoke or run}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
TASK_ENV="$MAGSET_ROOT/envs/notebook-tf215-v1"
TASK_RUNS="$MAGSET_ROOT/runs/supplied-cnn-transformer-v1"
test -f "$TASK_ENV/prepared.json"
export KERAS_HOME="$MAGSET_ROOT/cache/keras-notebook-tf215"
export MPLCONFIGDIR="$MAGSET_ROOT/cache/matplotlib-notebook-tf215"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 TF_FORCE_GPU_ALLOW_GROWTH=true
TASK_SITE="$("$TASK_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
TASK_CUDA_LIBS="$("$TASK_ENV/bin/python" -c 'import glob,sys; print(":".join(sorted(glob.glob(sys.argv[1]+"/nvidia/*/lib"))))' "$TASK_SITE")"
export LD_LIBRARY_PATH="$TASK_CUDA_LIBS:${LD_LIBRARY_PATH:-}"
export PATH="$TASK_SITE/nvidia/cuda_nvcc/bin:$PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$TASK_SITE/nvidia/cuda_nvcc"
export JUPYTER_RUNTIME_DIR="$MAGSET_ROOT/cache/notebook-runtime/$SLURM_JOB_ID"
export JUPYTER_CONFIG_DIR="$MAGSET_ROOT/cache/notebook-jupyter-config"
export IPYTHONDIR="$MAGSET_ROOT/cache/notebook-ipython/$SLURM_JOB_ID"
mkdir -p "$JUPYTER_RUNTIME_DIR" "$JUPYTER_CONFIG_DIR" "$IPYTHONDIR"
TASK_ARGS=(--action "$MAGSET_NOTEBOOK_ACTION" --notebook "$MAGSET_ROOT/inputs/supplied-cnn-transformer-v1/original.ipynb" --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --raw-root "$MAGSET_ROOT/data/raw" --dataset-root "$MAGSET_ROOT/data/raw/BreaKHis_v1/histology_slides/breast" --code-sha "$MAGSET_CODE_SHA" --config "$MAGSET_SOURCE/configs/supplied-cnn-transformer-v1.json")
case "$MAGSET_NOTEBOOK_ACTION" in
 audit)
  export CUDA_VISIBLE_DEVICES=""
  TASK_ARGS+=(--output "$TASK_RUNS/input-audit") ;;
 smoke|run)
  TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Use the two-variant array}"
  case "$TASK_INDEX" in 0) TASK_VARIANT=faithful ;; 1) TASK_VARIANT=input_fixed ;; *) exit 2 ;; esac
  TASK_ARGS+=(--variant "$TASK_VARIANT")
  if [[ "$MAGSET_NOTEBOOK_ACTION" == smoke ]]; then
   export CUDA_VISIBLE_DEVICES=""
   TASK_ARGS+=(--output "$TASK_RUNS/technical/$TASK_VARIANT")
  else
   "$TASK_ENV/bin/python" - "$TASK_RUNS" "$MAGSET_CODE_SHA" "$MAGSET_SOURCE/configs/supplied-cnn-transformer-v1.json" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);sha=sys.argv[2]
config_sha=hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest()
for variant in ('faithful','input_fixed'):
 folder=root/'technical'/variant
 result=json.loads((folder/'result.json').read_text())
 assert result['status']=='complete notebook execution' and result['code_sha']==sha and result['config_sha256']==config_sha and result['variant']==variant
 smoke=json.loads((folder/'smoke.json').read_text())
 assert smoke['status']=='accepted synthetic eight-image train and serialization smoke'
 assert smoke['train_batches']==1 and smoke['real_validation_predictions']==0
 assert max(smoke['weight_reload_max_abs'],smoke['full_model_reload_max_abs'])<=1e-5
 assert hashlib.sha256((folder/'smoke.json').read_bytes()).hexdigest()==result['artifact_sha256']['smoke.json']
CHECK
   TASK_ARGS+=(--audit "$TASK_RUNS/input-audit" --output "$TASK_RUNS/$TASK_VARIANT")
   nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  fi ;;
 *) exit 2 ;;
esac
cd "$MAGSET_SOURCE"
"$TASK_ENV/bin/python" -u scripts/run_supplied_cnn_transformer.py "${TASK_ARGS[@]}"
