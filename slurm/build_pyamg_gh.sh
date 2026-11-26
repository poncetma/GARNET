#!/usr/bin/env bash
#SBATCH --job-name=pyamg-build
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-hourly
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --output=/data/scratch/shared/poncet_m/pyamg-build/build-%j.out
#SBATCH --error=/data/scratch/shared/poncet_m/pyamg-build/build-%j.out
#
# Build pyamg (cmfd_solver="mg", the O(N) 3-D CMFD solve) from source on a
# GH200 node and install it into the GH environment: PyPI ships no Linux
# aarch64 wheel. Staging on the login node first (x86 pip, --target):
#   B=/data/scratch/shared/poncet_m/pyamg-build; P=<host python 3.13 with pip>
#   curl the pyamg-5.4.1.tar.gz sdist into $B/src and check its sha256
#   $P -m pip install --target $B/deps --only-binary=:all: --python-version 3.13 \
#       --implementation py --platform any setuptools setuptools-scm pybind11 \
#       wheel packaging meson meson-python versioningit pyproject-metadata
#   $P -m pip install --target $B/deps-arch --no-deps --only-binary=:all: \
#       --python-version 3.13 --platform manylinux2014_aarch64 ninja
#   $B/gh-bin/meson and $B/gh-bin/pybind11-config: wrappers running
#   "<GH python> -m mesonbuild.mesonmain" and "-m pybind11" with PYTHONPATH=$B/deps
# Built 2026-11-21: pyamg 5.4.1, gcc 7.5.0 (system), smoke solve residual 8e-11.
set -euo pipefail
B=/data/scratch/shared/poncet_m/pyamg-build
ENV=/data/scratch/shared/poncet_m/garnet-gh-py313-v1
PY=$ENV/bin/python3
SITE=$ENV/lib/python3.13/site-packages
echo "host=$(hostname) arch=$(uname -m)"
if ! command -v g++ >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module avail gcc 2>&1 | tail -5 || true
    module load gcc 2>/dev/null || module load GCC 2>/dev/null || true
fi
command -v g++ && g++ --version | head -1
export PYTHONPATH=$B/deps
export PATH=$B/gh-bin:$B/deps-arch/bin:$PATH
export PKG_CONFIG_PATH=$B/deps/pybind11/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}
pybind11-config --version
ninja --version; meson --version
rm -rf $B/work/pyamg-5.4.1
tar xzf $B/src/pyamg-5.4.1.tar.gz -C $B/work
$PY -m pip wheel --disable-pip-version-check --no-build-isolation --no-deps \
    -w $B/wheels $B/work/pyamg-5.4.1
unset PYTHONPATH
$PY -m pip install --disable-pip-version-check --no-deps --no-index --target $SITE \
    $B/wheels/pyamg-5.4.1-*.whl
$PY - <<'PY'
import numpy as np, pyamg, scipy.sparse as sp
A = pyamg.gallery.poisson((60, 60, 60), format="csr")
b = np.ones(A.shape[0])
ml = pyamg.smoothed_aggregation_solver(A)
x = ml.solve(b, tol=1e-10)
print("pyamg", pyamg.__version__, "residual", np.linalg.norm(b - A @ x) / np.linalg.norm(b))
PY
