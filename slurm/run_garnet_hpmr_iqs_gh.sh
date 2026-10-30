#!/usr/bin/env bash
#
# One case of the HP-MR IQS guard re-check (audit item O3) on a Merlin7 GH200
# node. Normally submitted by slurm/stage_and_submit_hpmr_iqs_gh.sh, which sets
# the case, time limit and partition on the sbatch command line.
#
# Every case runs examples/hpmr_coupled_transient.py on the calibrated 3-D
# HP-MR drum transient of benchmark-results/hpmr-iqs-gh200 (refine 4, nz 20,
# 11 groups, +0.248 $ over 4 s from t = 1 s, followed to 200 s).
# GARNET_IQS_CASE selects the treatment:
#   reference      full transient diffusion (progress written every step)
#   guarded-0.005  guarded adaptive IQS, iqs_predictor_tol 0.005
#   guarded-0.01   guarded adaptive IQS, iqs_predictor_tol 0.01
#   guarded-0.02   guarded adaptive IQS, iqs_predictor_tol 0.02 (calibrated)
#   fixed          fixed-cadence IQS (--unguarded-iqs)
# Guarded cases keep residual_tol 0.002 and fallback_residual 0.01.
# Optional overrides: GARNET_REPO, GARNET_PYTHON_BIN, GARNET_IQS_RESULTS,
# GARNET_IQS_ARGS (replaces the common transient arguments).

#SBATCH --job-name=garnet-hpmr-iqs
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-daily
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --mem=46G
#SBATCH --chdir=/data/scratch/shared/poncet_m/garnet-hpmr-iqs
#SBATCH --output=/data/scratch/shared/poncet_m/garnet-hpmr-iqs/garnet-hpmr-iqs-%j.out
#SBATCH --error=/data/scratch/shared/poncet_m/garnet-hpmr-iqs/garnet-hpmr-iqs-%j.out

set -euo pipefail

case_name="${GARNET_IQS_CASE:-guarded-0.02}"
case "$case_name" in
    reference) case_args=() ;;
    guarded-*)
        case_args=(--quasistatic-shape-dt 2 --shape-method iqs --adjoint-every 5
                   --residual-tol 0.002 --fallback-residual 0.01
                   --iqs-predictor-tol "${case_name#guarded-}") ;;
    fixed)
        case_args=(--quasistatic-shape-dt 2 --shape-method iqs --adjoint-every 5
                   --unguarded-iqs) ;;
    *) echo "error: unknown GARNET_IQS_CASE=$case_name" >&2; exit 2 ;;
esac

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
results="${GARNET_IQS_RESULTS:-/data/scratch/shared/poncet_m/garnet-hpmr-iqs/results}/${SLURM_JOB_ID:-manual}-$case_name"
mkdir -p "$results"
cd "$repo"

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

printf '=== GARNET HP-MR IQS case %s started at %s on %s ===\n' "$case_name" "$(date -Is)" "$(hostname)"
printf 'job=%s partition=%s repo=%s python=%s results=%s\n' "${SLURM_JOB_ID:-manual}" \
    "${SLURM_JOB_PARTITION:-unknown}" "$repo" "$python_bin" "$results"
if [[ -f "$repo/PROVENANCE.txt" ]]; then cp "$repo/PROVENANCE.txt" "$results/"; fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

"$python_bin" - <<'PY'
import cupy, numpy, scipy
from cupy.cuda import runtime
name = runtime.getDeviceProperties(0)["name"]
name = name.decode() if isinstance(name, bytes) else name
print(f"preflight: numpy={numpy.__version__} scipy={scipy.__version__} cupy={cupy.__version__} "
      f"devices={runtime.getDeviceCount()} device0={name}")
PY

read -r -a common <<<"${GARNET_IQS_ARGS:---refine 4 --nz 20 --groups 11 --t-end 200 --dt 0.05 --dt-thermal 0.5 --drum-from 90 --dollars 0.3076923076923077 --t-start 1 --t-ramp 4 --n-angles 13 --device gpu --profile}"
output=(--json "$results/record.json")
if [[ "$case_name" == reference ]]; then
    output+=(--progress "$results/progress.txt")
fi

status=0
"$python_bin" -u examples/hpmr_coupled_transient.py "${common[@]}" "${case_args[@]}" \
    "${output[@]}" || status=$?

printf '=== GARNET HP-MR IQS case %s finished at %s with status %d ===\n' "$case_name" "$(date -Is)" "$status"
exit "$status"
