#!/usr/bin/env bash
# Run docs/theory/run_cases.py studies on one A100 (gmerlin7); HP-MR studies only use the GPU.
#   sbatch docs/theory/run_cases_gpu.sbatch [STUDY ...]
#SBATCH --job-name=garnet-theory-gpu
#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=/data/scratch/shared/poncet_m/garnet-slurm-debug/garnet-theory-%j.out

set -euo pipefail
repo="${GARNET_REPO:-/data/user/poncet_m/GARNET}"
export PYTHONPATH="$repo"
export GARNET_THEORY_DEVICE=gpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$repo"
echo "start $(date -Is) on $(hostname) ($(nproc) cores)"
/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python -u docs/theory/run_cases.py "$@"
echo "end $(date -Is)"
