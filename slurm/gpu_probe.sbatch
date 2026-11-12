#!/usr/bin/env bash
#SBATCH --job-name=magset-gpu-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
set -euo pipefail
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 --version
# Pass the allocation, partition, GPU request, and output path to sbatch.
# This checks scheduler/device access, not a framework or training run.
