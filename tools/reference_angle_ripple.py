"""How linear is the HP-MR S_N reference itself in drum angle?

At each test angle, the reference k, assembly powers and pin powers are compared
with the linear interpolation of the same reference between its two
neighbouring table angles on the given spacing. A table model cannot do better
than this where the reference is not smooth, so it bounds the interpolation
error that is the reference's own (e.g. mesh-alignment ripple of the drum arc).

Usage: python tools/reference_angle_ripple.py refine spacing angles [method] [cache]
       angles: comma-separated test angles (off the table grid)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hpmr_pin_reference import DEFAULT_CACHE  # noqa: E402
from sph_hpmr_nodes import pin_reference  # noqa: E402

refine, spacing = int(sys.argv[1]), float(sys.argv[2])
tests = [float(a) for a in sys.argv[3].split(",")]
method = sys.argv[4] if len(sys.argv) > 4 else "sn-p2:4:16"
cache = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_CACHE
_memo = {}


def ref(a):
    if a not in _memo:
        r = pin_reference(refine, method, a, cache=cache, device="cpu")
        pin = r["rate"].sum(axis=1)
        asm = np.bincount(r["pins"].assembly[r["pins"].fuel], weights=pin)
        _memo[a] = (r["k"], pin / pin.mean(), asm / asm.mean())
    return _memo[a]


print(f"{method} refine {refine}: reference at the test angle against its own linear "
      f"interpolation on {spacing:g} deg\n angle  dk pcm | assembly max % | pin RMS %  worst pin %",
      flush=True)
t0 = time.perf_counter()
out = []
for a in tests:
    lo = spacing * np.floor(a / spacing)
    hi = lo + spacing
    w = (a - lo) / spacing
    (kl, pl, al), (kh, ph, ah), (k, p, s) = ref(lo), ref(hi), ref(a)
    ki, pi, ai = (1 - w) * kl + w * kh, (1 - w) * pl + w * ph, (1 - w) * al + w * ah
    pi, ai = pi / pi.mean(), ai / ai.mean()
    ep, ea = pi / p - 1.0, ai / s - 1.0
    j = int(np.argmax(np.abs(ep)))
    out.append((a, 1e5 * (ki - k), 100 * np.abs(ea).max(), 100 * np.sqrt((ep ** 2).mean()),
                100 * ep[j]))
    print(f"{a:6.2f}  {out[-1][1]:+7.1f} | {out[-1][2]:6.2f} | {out[-1][3]:6.2f}  "
          f"{out[-1][4]:+7.2f}  [{time.perf_counter() - t0:.0f} s]", flush=True)
o = np.array(out)
print(f"\nworst: |dk| {np.abs(o[:, 1]).max():.1f} pcm, assembly {o[:, 2].max():.2f} %, "
      f"pin RMS {o[:, 3].max():.2f} %, worst pin {np.abs(o[:, 4]).max():.2f} %")
