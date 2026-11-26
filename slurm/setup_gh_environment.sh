#!/usr/bin/env bash
#
# Build GARNET's native aarch64 Python environment for the GH200 nodes. Run on
# an x86_64 Merlin login node (it needs network access, the GH200 nodes do not
# get it):
#
#   bash slurm/setup_gh_environment.sh
#
# The interpreter is a relocatable python-build-standalone CPython; packages are
# aarch64 wheels installed with the login node's pip (--platform/--target), so
# no module system or CUDA toolkit is needed on the GPU node. The CUDA 12.6
# runtime comes from nvidia-* wheels; the run scripts put their lib directories
# on LD_LIBRARY_PATH. Validate on a GPU with slurm/probe_gh_environment.sh.
#
# Overrides: GARNET_GH_ENV (target), GARNET_GH_DOWNLOADS (tarball cache),
# HOST_PYTHON (x86_64 Python >= 3.10 with pip).

set -euo pipefail

env_root="${GARNET_GH_ENV:-/data/scratch/shared/poncet_m/garnet-gh-py313-v1}"
downloads="${GARNET_GH_DOWNLOADS:-/data/scratch/shared/poncet_m/garnet-gh-downloads}"
host_python="${HOST_PYTHON:-/opt/psi/Programming/Python/3.13.2/bin/python3}"

pbs_release=20250918
pbs_file="cpython-3.13.7+${pbs_release}-aarch64-unknown-linux-gnu-install_only.tar.gz"
pbs_sha256=7d4448fe9b6fa37d8d16635d52d1d8ffb42576d85e23331ae345a07cfa60bc61

packages=(numpy==2.3.2 scipy==1.16.3 cupy-cuda12x==13.6.0 pytest==8.4.2)
# CUDA 12.6 components. Installed with --no-deps: the newest cuSPARSE in the
# 12.5 series is a CUDA 12.8 build that would pull a newer nvJitLink.
cuda_packages=(
    nvidia-cuda-runtime-cu12==12.6.77
    nvidia-cuda-nvrtc-cu12==12.6.85
    nvidia-nvjitlink-cu12==12.6.85
    nvidia-cublas-cu12==12.6.3.3
    nvidia-cusparse-cu12==12.5.4.2
    nvidia-cusolver-cu12==11.7.1.2
    nvidia-curand-cu12==10.3.7.77
    nvidia-cufft-cu12==11.3.0.4
)
platform_args=(--only-binary=:all: --python-version 3.13 --implementation cp --abi cp313
               --platform manylinux_2_28_aarch64 --platform manylinux_2_17_aarch64
               --platform manylinux2014_aarch64 --platform linux_aarch64)

printf '=== GH200 environment setup started: %s ===\n' "$(date -Is)"
printf 'host=%s arch=%s env=%s\n' "$(hostname)" "$(uname -m)" "$env_root"
if [[ -e "$env_root" ]]; then
    echo "error: $env_root exists; remove it first to rebuild" >&2
    exit 1
fi

mkdir -p "$downloads"
if [[ ! -f "$downloads/$pbs_file" ]]; then
    curl -fsSL -o "$downloads/$pbs_file" \
        "https://github.com/astral-sh/python-build-standalone/releases/download/$pbs_release/$pbs_file"
fi
echo "$pbs_sha256  $downloads/$pbs_file" | sha256sum -c -

mkdir -p "$env_root"
tar -xzf "$downloads/$pbs_file" -C "$env_root" --strip-components=1
site="$env_root/lib/python3.13/site-packages"

"$host_python" -m pip install --disable-pip-version-check "${platform_args[@]}" \
    --target "$site" --upgrade "${packages[@]}"
# A second --target install skips the existing nvidia/ namespace directory, so
# all CUDA wheels go in together.
"$host_python" -m pip install --disable-pip-version-check "${platform_args[@]}" \
    --target "$site" --no-deps "${cuda_packages[@]}"

file -L "$env_root/bin/python3.13" "$site"/numpy/_core/_multiarray_umath*.so \
    "$site/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
printf '=== GH200 environment setup finished: %s ===\n' "$(date -Is)"
printf 'next: sbatch slurm/probe_gh_environment.sh; for cmfd_solver="mg" build pyamg with slurm/build_pyamg_gh.sh (no aarch64 wheel)\n'
