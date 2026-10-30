#!/usr/bin/env bash
# Stage an independent GARNET snapshot in shared scratch and submit the complete
# GPU regression suite. The separate stage avoids mutating queued benchmarks.
#
# Overrides: GARNET_REGRESSION_PARTITION (default a100-daily; gh-* selects the
# GH200 Python), GARNET_REGRESSION_TIME (default 04:00:00), GARNET_PYTEST_TARGETS
# and GARNET_PYTEST_ARGS (see run_garnet_gpu_regression.sh), SBATCH ("echo" for
# a dry run).

set -euo pipefail

src_root="${GARNET_SRC_ROOT:-$HOME/GARNET}"
stage_root="${GARNET_REGRESSION_STAGE:-/data/scratch/shared/poncet_m/garnet-regression-stage}"
logdir="${GARNET_REGRESSION_LOG_DIR:-/data/scratch/shared/poncet_m/garnet-regression-logs}"

mkdir -p "$stage_root" "$logdir"
rsync -a --delete \
    --exclude '.git' \
    --exclude '.agents' \
    --exclude '.codex' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude 'results' \
    "$src_root/" "$stage_root/repo/"

# .git is not staged, so record exactly which source state the job ran.
{
    printf 'staged: %s by %s on %s\n' "$(date -Is)" "$(id -un)" "$(hostname)"
    printf 'source: %s\n' "$src_root"
    git -C "$src_root" log -1 --format='commit: %H %ad %s' --date=iso
    printf '\n# git status --short\n'
    git -C "$src_root" status --short
} >"$stage_root/repo/PROVENANCE.txt" 2>&1 || true

partition="${GARNET_REGRESSION_PARTITION:-a100-daily}"
if [[ "$partition" == gh-* ]]; then
    default_python=/data/scratch/shared/poncet_m/garnet-gh-py313-v1/bin/python
else
    default_python=/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python
fi
export GARNET_REPO="$stage_root/repo"
export GARNET_PYTHON_BIN="${GARNET_PYTHON_BIN:-$default_python}"
export GARNET_TEST_DEPS="${GARNET_TEST_DEPS:-/data/scratch/shared/poncet_m/garnet-test-deps/pytest8}"
export GARNET_REGRESSION_LOG_DIR="$logdir"
"${SBATCH:-sbatch}" --parsable --partition="$partition" \
    --time="${GARNET_REGRESSION_TIME:-04:00:00}" \
    "$stage_root/repo/slurm/run_garnet_gpu_regression.sh"
