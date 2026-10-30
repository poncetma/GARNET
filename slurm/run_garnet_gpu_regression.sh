#!/usr/bin/env bash
#
# Run the complete GARNET regression suite on one Merlin GPU (A100 or GH200).
# The project normally excludes tests marked "slow"; this job clears the
# configured addopts so every collected regression is included.
# GARNET_PYTEST_TARGETS (paths relative to the repo) narrows the run;
# GARNET_PYTEST_ARGS replaces the pytest options.

#SBATCH --job-name=garnet-gpu-regression
#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-daily
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --mem=46G
#SBATCH --chdir=/data/scratch/shared/poncet_m/garnet-regression-logs
#SBATCH --output=/data/scratch/shared/poncet_m/garnet-regression-logs/garnet-gpu-regression-%j.out
#SBATCH --error=/data/scratch/shared/poncet_m/garnet-regression-logs/garnet-gpu-regression-%j.err

set -euo pipefail

if [[ "$(uname -m)" == "aarch64" ]]; then
    default_python="/data/scratch/shared/poncet_m/garnet-gh-py313-v1/bin/python"
    # The CUDA 12.6 runtime libraries come from the environment's nvidia-* wheels.
    for lib in "$(dirname "$(dirname "${GARNET_PYTHON_BIN:-$default_python}")")"/lib/python3.13/site-packages/nvidia/*/lib; do
        LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
    export LD_LIBRARY_PATH
else
    default_python="/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python"
fi
python_bin="${GARNET_PYTHON_BIN:-$default_python}"
if [[ ! -x "$python_bin" ]]; then
    if [[ -n "${MODULESHOME:-}" ]]; then
        module load "${GARNET_PYTHON_MODULE:-Python/3.11.11}"
    fi
    python_bin="${PYTHON_BIN:-}"
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    python_bin="$(command -v python3.12 || command -v python3.11 || command -v python3 || true)"
fi

repo="${GARNET_REPO:-}"
if [[ -z "$repo" ]]; then
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo="$(cd "$here/.." && pwd)"
fi

test_deps="${GARNET_TEST_DEPS:-/data/scratch/shared/poncet_m/garnet-test-deps/pytest8}"
logdir="${GARNET_REGRESSION_LOG_DIR:-/data/scratch/shared/poncet_m/garnet-regression-logs}"
mkdir -p "$logdir"
logfile="$logdir/garnet-gpu-regression-${SLURM_JOB_ID:-manual}.log"
touch "$logfile"
exec >>"$logfile" 2>&1
cd "$repo"

printf '=== GARNET full GPU regression started at %s on %s ===\n' "$(date -Is)" "$(hostname)"
printf 'job=%s repo=%s python=%s test_deps=%s\n' \
    "${SLURM_JOB_ID:-manual}" "$repo" "$python_bin" "$test_deps"

if [[ -z "$python_bin" ]]; then
    echo "error: no usable Python interpreter found" >&2
    exit 127
fi
# The GH200 environment ships pytest; the x86 conda Python needs test_deps.
if [[ -d "$test_deps" ]]; then
    export PYTHONPATH="$test_deps:$repo${PYTHONPATH:+:$PYTHONPATH}"
else
    export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

if ! "$python_bin" - <<'PY'
import cupy
import numpy
import pytest
import scipy
from cupy.cuda import runtime

from garnet import TransientSolver
from garnet.multigroup import MultigroupStepOperator

device_count = runtime.getDeviceCount()
if device_count < 1:
    raise RuntimeError("CuPy cannot see an allocated CUDA device")
props = runtime.getDeviceProperties(0)
name = props["name"]
if isinstance(name, bytes):
    name = name.decode()
print(f"preflight: pytest={pytest.__version__} numpy={numpy.__version__} "
      f"scipy={scipy.__version__} cupy={cupy.__version__}")
print(f"preflight: cuda_devices={device_count} device0={name}")
print(f"preflight: {TransientSolver.__name__}, {MultigroupStepOperator.__name__}")
PY
then
    echo "error: GPU regression environment preflight failed" >&2
    exit 1
fi

if [[ -n "${GARNET_PYTEST_ARGS:-}" ]]; then
    read -r -a pytest_args <<<"$GARNET_PYTEST_ARGS"
else
    pytest_args=(-ra -q -o addopts= --durations=25)
fi

if [[ -f "$repo/PROVENANCE.txt" ]]; then cat "$repo/PROVENANCE.txt"; fi
read -r -a targets <<<"${GARNET_PYTEST_TARGETS:-tests}"

printf 'pytest arguments:'
printf ' %q' "${pytest_args[@]}" "${targets[@]}"
printf '\n'

set +e
"$python_bin" -u -m pytest "${pytest_args[@]}" "${targets[@]}"
status=$?
set -e

printf '=== GARNET full GPU regression finished at %s with status %d ===\n' \
    "$(date -Is)" "$status"
exit "$status"
