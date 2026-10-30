#!/usr/bin/env bash
# Runs every configuration of the Carreno et al. (2024) V&V study, one process
# per (case, method, vacuum treatment, mesh), most expensive first, JOBS at a
# time. Each run writes data/runs/<tag>.json and logs/<tag>.log (one line per
# outer iteration); progress.log gets a DONE or FAIL line per run.
#
#   JOBS=16 bash benchmark-results/carreno2024-sdpn-vv/run_sweep.sh [bl|c5g7|all]
#
# Run from the repository root with garnet importable (PYTHONPATH=.).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PY=${PYTHON:-python}
JOBS=${JOBS:-16}
WHICH=${1:-all}
mkdir -p "$HERE/data/runs" "$HERE/logs"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# One job per line: "<cost> <tag> <tool arguments>", sorted by cost below.
joblist() {
  local SN="SP1 SP3 SP5 SP7 SDP1 SDP2 SDP3"
  if [[ $WHICH == bl || $WHICH == all ]]; then
    for n in 40 80 160 320 640; do
      for m in $SN SP3-SPN SDP1-SDPN; do
        for v in marshak robin; do
          [[ $m == SP1 && $v == robin ]] && continue
          echo "$(( n * n / 100 )) bl-$m-$v-n$n bl --methods $m --vacuum $v --meshes $n"
        done
      done
    done
  fi
  if [[ $WHICH == c5g7 || $WHICH == all ]]; then
    local base="c5g7 --tol-k 1e-9 --tol-source 1e-8"
    for s in 4 6 8 12 16; do
      for m in $SN; do
        echo "$(( s * s * 300 )) c5g7-pin-$m-marshak-s$s $base --methods $m --vacuum marshak --cells-per-pin $s"
      done
    done
    for s in 4 8 16; do
      for m in SP3 SP5 SP7 SDP1 SDP2 SDP3; do
        echo "$(( s * s * 300 )) c5g7-pin-$m-robin-s$s $base --methods $m --vacuum robin --cells-per-pin $s"
      done
    done
    for m in $SN; do
      echo "19200 c5g7-pin-$m-zeroflux-s8 $base --methods $m --vacuum zero-flux --cells-per-pin 8"
    done
    for s in 4 8 16; do
      echo "$(( s * s * 300 )) c5g7-pin-SP1-sampled-s$s $base --methods SP1 --vacuum marshak --cells-per-pin $s --area sampled"
    done
    for m in SP3-SPN SDP1-SDPN; do
      echo "4800 c5g7-pin-$m-marshak-s4 $base --methods $m --vacuum marshak --cells-per-pin 4"
    done
    for s in 1 2 4 8; do
      for m in $SN; do
        for v in marshak zero-flux; do
          echo "$(( s * s * 200 )) c5g7-hom-$m-${v/-/}-s$s $base --geometry homogenized --methods $m --vacuum $v --cells-per-pin $s"
        done
      done
    done
  fi
}

run() {
  local tag=$1; shift
  local out="$HERE/data/runs/$tag.json" log="$HERE/logs/$tag.log"
  if [[ -s $out ]]; then echo "SKIP $tag" >> "$HERE/progress.log"; return 0; fi
  if "$PY" "$ROOT/tools/assess_carreno2024.py" "$@" --verbose --output "$out" > "$log" 2>&1; then
    echo "DONE $tag $(grep -h '^RESULT' "$log" | tail -1)" >> "$HERE/progress.log"
  else
    echo "FAIL $tag (exit $?, see logs/$tag.log)" >> "$HERE/progress.log"
  fi
}
export -f run
export HERE ROOT PY

joblist | sort -rn | cut -d' ' -f2- | xargs -P "$JOBS" -L 1 bash -c 'run "$@"' _
echo "SWEEP FINISHED $WHICH" >> "$HERE/progress.log"
