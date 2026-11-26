#!/usr/bin/env bash
# Run docs/theory/run_cases.py on one idle merlin7 CPU core.
#   sbatch docs/theory/run_cases.sbatch [STUDY ...]
#SBATCH --job-name=garnet-theory
#SBATCH --cluster=merlin7
#SBATCH --partition=daily
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --output=/data/scratch/shared/poncet_m/garnet-slurm-debug/garnet-theory-%j.out

set -euo pipefail
repo="${GARNET_REPO:-/data/user/poncet_m/GARNET}"
export PYTHONPATH="$repo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$repo"
echo "start $(date -Is) on $(hostname) ($(nproc) cores)"
/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python -u docs/theory/run_cases.py "$@"
echo "end $(date -Is)"
