#!/usr/bin/env bash
#SBATCH --job-name=magset-supplied-swin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
: "${MAGSET_ROOT:?}" "${MAGSET_SOURCE:?}" "${MAGSET_CODE_SHA:?}" "${MAGSET_SWIN_ACTION:?}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
TASK_ENV="$MAGSET_ROOT/envs/swin-tf215-v1"
TASK_RUNS="$MAGSET_ROOT/runs/supplied-swin-v1/attempt-2"
test -f "$TASK_ENV/prepared.json"
export HF_HOME="$MAGSET_ROOT/cache/hf-supplied-swin" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TF_USE_LEGACY_KERAS=1
export MPLCONFIGDIR="$MAGSET_ROOT/cache/matplotlib-swin-tf215"
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
mkdir -p "$JUPYTER_RUNTIME_DIR" "$JUPYTER_CONFIG_DIR" "$IPYTHONDIR" "$TASK_RUNS"
TASK_ARGS=(--action "$MAGSET_SWIN_ACTION" --notebook "$MAGSET_ROOT/inputs/supplied-swin-v1/original.ipynb" --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --raw-root "$MAGSET_ROOT/data/raw" --dataset-root "$MAGSET_ROOT/data/raw/BreaKHis_v1/histology_slides/breast" --code-sha "$MAGSET_CODE_SHA" --config "$MAGSET_SOURCE/configs/supplied-swin-v1.json" --prepared "$TASK_ENV/prepared.json")
case "$MAGSET_SWIN_ACTION" in
 audit) export CUDA_VISIBLE_DEVICES=""; TASK_ARGS+=(--output "$TASK_RUNS/input-audit") ;;
 smoke) export CUDA_VISIBLE_DEVICES=""; TASK_ARGS+=(--output "$TASK_RUNS/technical") ;;
 run)
  "$TASK_ENV/bin/python" - "$TASK_RUNS" "$MAGSET_CODE_SHA" "$MAGSET_SOURCE/configs/supplied-swin-v1.json" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); result=json.loads((root/'technical/result.json').read_text()); smoke=json.loads((root/'technical/smoke.json').read_text())
assert result['status']=='complete notebook execution' and result['action']=='smoke' and result['code_sha']==sys.argv[2]
assert result['config_sha256']==hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest()
assert result['artifact_sha256']['smoke.json']==hashlib.sha256((root/'technical/smoke.json').read_bytes()).hexdigest()
assert smoke['status']=='accepted synthetic Swin train and weight reload'
assert smoke['train_batches']==1 and smoke['real_test_predictions']==0 and smoke['masked_benign_loss']==0
assert smoke['weight_reload_max_abs']<=1e-5 and smoke['encoder_stage_trainable']==[False,False,True,True]
CHECK
  TASK_ARGS+=(--audit "$TASK_RUNS/input-audit" --output "$TASK_RUNS/dual-split0")
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader ;;
 *) exit 2 ;;
esac
cd "$MAGSET_SOURCE"
"$TASK_ENV/bin/python" -u scripts/run_supplied_swin.py "${TASK_ARGS[@]}"
