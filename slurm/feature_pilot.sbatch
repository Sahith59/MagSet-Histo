#!/usr/bin/env bash
#SBATCH --job-name=magset-feature-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -euo pipefail
: "${MAGSET_ROOT:?Set MAGSET_ROOT}"
: "${MAGSET_CODE_SHA:?Set MAGSET_CODE_SHA}"
export HF_HOME="$MAGSET_ROOT/cache/huggingface"
export TORCH_HOME="$MAGSET_ROOT/cache/torch"
export TMPDIR="$MAGSET_ROOT/cache/tmp"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
cd "$MAGSET_ROOT/repo"
"$MAGSET_ROOT/envs/magset/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA missing'
x=torch.randn(32,32,device='cuda',requires_grad=True)
x.square().mean().backward()
assert torch.isfinite(x.grad).all(), 'Nonfinite GPU gradient'
print({'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'gradient_smoke':'passed'})
PY
for TASK_ENCODER in phikon-v2 resnet50; do
  "$MAGSET_ROOT/envs/magset/bin/python" -m magset.features \
    --manifest "$MAGSET_ROOT/data/processed/manifest.csv" --images-root "$MAGSET_ROOT/data/raw" \
    --output-root "$MAGSET_ROOT/cache/features-pilot" --encoder "$TASK_ENCODER" --magnification 40 \
    --batch-size 8 --workers 0 --limit 16 --code-sha "$MAGSET_CODE_SHA"
done
