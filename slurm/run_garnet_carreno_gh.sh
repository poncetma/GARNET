#!/usr/bin/env bash
#
# One method of the C5G7 pin-resolved mesh study of the Carreno et al. (2024)
# V&V report (audit item O10) on a Merlin7 GH200 node. Normally submitted by
# slurm/stage_and_submit_carreno_gh.sh, which sets the method on the sbatch
# command line.
#
# Runs tools/assess_carreno2024.py c5g7 for GARNET_VV_METHOD with the coupled
# Marshak vacuum at GARNET_VV_CELLS (default "8 16 24" cells per pin; 8 is the
# CPU/GPU agreement check), then with the per-moment Robin vacuum at
# GARNET_VV_ROBIN_CELLS (default "16"; empty skips it). Every outer iteration
# is logged. Optional overrides: GARNET_REPO, GARNET_PYTHON_BIN,
# GARNET_VV_RESULTS.

#SBATCH --job-name=garnet-carreno-vv
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-daily
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --mem=46G
#SBATCH --chdir=/data/scratch/shared/poncet_m/garnet-carreno-vv
#SBATCH --output=/data/scratch/shared/poncet_m/garnet-carreno-vv/garnet-carreno-vv-%j.out
#SBATCH --error=/data/scratch/shared/poncet_m/garnet-carreno-vv/garnet-carreno-vv-%j.out

set -euo pipefail

method="${GARNET_VV_METHOD:?set GARNET_VV_METHOD (SP1 SP3 SP5 SP7 SDP1 SDP2 SDP3)}"
read -r -a cells <<<"${GARNET_VV_CELLS:-8 16 24}"
read -r -a robin_cells <<<"${GARNET_VV_ROBIN_CELLS-16}"

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "error: this job expects a GH200 (aarch64) node, got $(uname -m)" >&2
    exit 1
fi
python_bin="${GARNET_PYTHON_BIN:-/data/scratch/shared/poncet_m/garnet-gh-py313-v1/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "error: GH200 Python environment not found: $python_bin" >&2
    echo "build it on a login node with slurm/setup_gh_environment.sh" >&2
    exit 1
fi
# The CUDA 12.6 runtime libraries come from the environment's nvidia-* wheels.
for lib in "$(dirname "$(dirname "$python_bin")")"/lib/python3.13/site-packages/nvidia/*/lib; do
    LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH

repo="${GARNET_REPO:-}"
if [[ -z "$repo" ]]; then
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo="$(cd "$here/.." && pwd)"
fi
results="${GARNET_VV_RESULTS:-/data/scratch/shared/poncet_m/garnet-carreno-vv/results}/${SLURM_JOB_ID:-manual}-$method"
mkdir -p "$results"
cd "$repo"

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

printf '=== GARNET Carreno V&V, C5G7 %s, started at %s on %s ===\n' "$method" "$(date -Is)" "$(hostname)"
printf 'job=%s repo=%s python=%s results=%s\n' "${SLURM_JOB_ID:-manual}" "$repo" "$python_bin" "$results"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

common=(c5g7 --methods "$method" --device gpu --tol-k 1e-9 --tol-source 1e-8 --verbose)
status=0
for s in "${cells[@]}"; do
    "$python_bin" tools/assess_carreno2024.py "${common[@]}" --vacuum marshak \
        --cells-per-pin "$s" --output "$results/c5g7-pin-$method-marshak-s$s.json" || status=$?
done
if [[ "$method" != SP1 ]]; then
    for s in "${robin_cells[@]}"; do
        "$python_bin" tools/assess_carreno2024.py "${common[@]}" --vacuum robin \
            --cells-per-pin "$s" --output "$results/c5g7-pin-$method-robin-s$s.json" || status=$?
    done
fi
printf '=== finished at %s, status %s ===\n' "$(date -Is)" "$status"
exit "$status"
