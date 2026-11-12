#!/usr/bin/env bash
#SBATCH --job-name=magset-runtime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_CONDA_EXE:?Set MAGSET_CONDA_EXE to the site conda executable}"
export TMPDIR="$MAGSET_ROOT/cache/tmp"
export CONDA_PKGS_DIRS="$MAGSET_ROOT/cache/conda-pkgs"
export PIP_CACHE_DIR="$MAGSET_ROOT/cache/pip"
export HF_HOME="$MAGSET_ROOT/cache/huggingface"
export TORCH_HOME="$MAGSET_ROOT/cache/torch"
export XDG_CACHE_HOME="$MAGSET_ROOT/cache/xdg"
export HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$TMPDIR" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"
exec 9>"$MAGSET_ROOT/envs/.bootstrap.lock"
flock -n 9 || { echo 'Another runtime bootstrap holds the project lock'; exit 3; }
TASK_ENV="$MAGSET_ROOT/envs/magset"
if [[ ! -x "$TASK_ENV/bin/python" ]]; then
  "$MAGSET_CONDA_EXE" create --yes --prefix "$TASK_ENV" --override-channels -c conda-forge python=3.11 pip
fi
cd "$MAGSET_ROOT/repo"
"$TASK_ENV/bin/python" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$TASK_ENV/bin/python" -m pip install -r requirements.txt
"$TASK_ENV/bin/python" -m pip check
"$TASK_ENV/bin/python" -m pip freeze > "$MAGSET_ROOT/envs/requirements.lock.txt"
"$TASK_ENV/bin/python" - <<'PY'
import json, os, platform
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(os.environ['MAGSET_ROOT'])
config = json.loads((root/'repo/configs/encoders.json').read_text())['phikon-v2']
# Download a pinned public checkpoint in a CPU job before reserving GPUs.
snapshot_download(repo_id=config['repository'], revision=config['revision'],
                  allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors'])
import torch, torchvision, transformers
record = {'python': platform.python_version(), 'torch': torch.__version__,
          'torchvision': torchvision.__version__, 'transformers': transformers.__version__,
          'encoder_revision': config['revision'],
          'gpu_smoke_status': 'pending; runtime installed on a CPU allocation'}
(root/'envs/runtime-ready.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
PY
