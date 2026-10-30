#!/usr/bin/env bash
# Best-of-3 CPU timings for the cost tables, one process at a time: every
# Brantley-Larsen method and vacuum treatment at n = 160, and every C5G7
# pin-resolved method with Marshak vacuum at 4 cells per pin. Run on an
# otherwise idle machine, after run_sweep.sh.
#
#   bash benchmark-results/carreno2024-sdpn-vv/run_timing.sh
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PY=${PYTHON:-python}
mkdir -p "$HERE/data/timing"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PY" "$ROOT/tools/assess_carreno2024.py" bl --vacuum marshak robin --meshes 160 \
    --repeats 3 --output "$HERE/data/timing/bl-n160.json"
"$PY" "$ROOT/tools/assess_carreno2024.py" c5g7 --vacuum marshak --cells-per-pin 4 \
    --tol-k 1e-9 --tol-source 1e-8 --no-lines --repeats 3 \
    --output "$HERE/data/timing/c5g7-pin-s4.json"
