# Running on a SLURM cluster

Set a shared project root and your allocation before submission. The scripts read configuration from the environment; they do not contain a university username or account.

```bash
export MAGSET_ROOT=/shared/project/magset-histo
export MAGSET_ACCOUNT=your-allocation
export MAGSET_CPU_PARTITION=cpu-partition
export MAGSET_GPU_PARTITION=gpu-partition
mkdir -p "$MAGSET_ROOT"/{repo,data,cache/tmp,envs,runs,logs}
```

Place a copy of the source in `$MAGSET_ROOT/repo`, inspect your queue, then submit:

```bash
sbatch --account="$MAGSET_ACCOUNT" --partition="$MAGSET_CPU_PARTITION" \
  --output="$MAGSET_ROOT/logs/data-%j.log" slurm/prepare_data.sbatch
sbatch --account="$MAGSET_ACCOUNT" --partition="$MAGSET_GPU_PARTITION" \
  --gres=gpu:1 --output="$MAGSET_ROOT/logs/gpu-probe-%j.log" slurm/gpu_probe.sbatch
```

Your site may require a typed GPU request such as `gpu:A40:1`. Follow its policy. A successful `nvidia-smi` probe establishes device access; a separate framework smoke test is required before training.

For source synchronization, set `MAGSET_SSH_HOST` to an existing configured SSH alias and `MAGSET_REMOTE_ROOT` to the destination. `scripts/sync_cluster.sh` uses rsync without deletion. It separately mirrors optional private running notes; those files are ignored by Git. Do not commit SSH configuration, credentials, personal paths, raw datasets, or model weights.

Use job arrays for independent magnifications, encoders, folds, and seeds once pilots establish runtime/memory needs. Apply a concurrency limit and avoid duplicate runs. Store large caches and checkpoints on shared project storage, with explicit retention limits.

## Runtime and experiment dependency chain

`slurm/bootstrap.sbatch` creates a Python 3.11 environment under shared storage, installs the pinned CUDA 12.4 PyTorch 2.6/torchvision 0.21 pair and `requirements.txt`, checks dependencies, saves an environment lock, and downloads the pinned public Phikon-v2 checkpoint. Set `MAGSET_CONDA_EXE` to your site's conda executable. This is a CPU scheduler task.

After it succeeds, run a `feature_pilot.sbatch` GPU job. It performs a CUDA gradient check and encodes 16 real images with each encoder in a separate smoke-cache directory. Only after that succeeds should `extract_features.sbatch` run its eight tasks with concurrency four. `MAGSET_CODE_SHA` must identify the source revision.

Image verification uses `audit_images.sbatch` with four CPU shards. `validation_baselines.sbatch` has 30 tasks capped at six CPU workers. Submit it only after the feature array and protocol audit succeed. This array does not request GPUs: cached-feature linear models benefit from independent CPU runs rather than reserving idle GPUs. Outer-test predictions remain disabled during these initial pilots.
