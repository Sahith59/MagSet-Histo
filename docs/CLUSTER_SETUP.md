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
