"""Mesh convergence of the cached HP-MR S_N references across drum angle.

Compares two refinements of the same method (default ``sn-p2:4:16``, refine 16
against 32) at every angle both caches hold: k, drum worth relative to 0 deg,
the smoothness of k(theta) (second differences, midpoint interpolation error on
the 15 deg grid), fuel-assembly powers and pin powers (1890 fuel compacts).

Usage: python tools/compare_reference_refinement.py [coarse] [fine] [method] [cache]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hpmr_pin_reference import DEFAULT_CACHE, cache_path  # noqa: E402
from sph_hpmr_nodes import pin_reference  # noqa: E402

coarse = int(sys.argv[1]) if len(sys.argv) > 1 else 16
fine = int(sys.argv[2]) if len(sys.argv) > 2 else 32
method = sys.argv[3] if len(sys.argv) > 3 else "sn-p2:4:16"
cache = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_CACHE
only = [float(a) for a in os.environ.get("REF_ANGLES", "").split(",") if a.strip()]

angles = [a for a in np.arange(0.0, 180.01, 2.5)
          if os.path.exists(cache_path(cache, method, coarse, a))
          and os.path.exists(cache_path(cache, method, fine, a))]
if only:
    angles = [a for a in angles if any(np.isclose(a, o) for o in only)]
print(f"{method}: refine {coarse} against refine {fine}, angles {angles}\n", flush=True)


def powers(ref):
    pin = ref["rate"].sum(axis=1)
    asm = np.bincount(ref["pins"].assembly[ref["pins"].fuel], weights=pin)
    return pin / pin.mean(), asm / asm.mean()


rows = []
t0 = time.perf_counter()
print(" angle   k coarse   k fine    dk pcm | assembly max % | pin RMS %  worst pin % "
      "(fine as reference)", flush=True)
for a in angles:
    rc = pin_reference(coarse, method, a, cache=cache, device="cpu")
    rf = pin_reference(fine, method, a, cache=cache, device="cpu")
    pc, ac = powers(rc)
    pf, af = powers(rf)
    e_pin, e_asm = pc / pf - 1.0, ac / af - 1.0
    w = int(np.argmax(np.abs(e_pin)))
    row = (a, rc["k"], rf["k"], 1e5 * (rc["k"] - rf["k"]), 100 * np.abs(e_asm).max(),
           100 * np.sqrt((e_pin ** 2).mean()), 100 * e_pin[w])
    rows.append(row)
    print(f"{a:6.1f}  {row[1]:.6f}  {row[2]:.6f}  {row[3]:+8.1f} | {row[4]:6.2f} | "
          f"{row[5]:6.2f}  {row[6]:+7.2f}  [{time.perf_counter() - t0:.0f} s]", flush=True)

r = np.array(rows)
a, kc, kf = r[:, 0], r[:, 1], r[:, 2]
print("\nDrum worth from the lowest angle (pcm, rho = 1 - 1/k):")
rho = lambda k: 1e5 * (1.0 - 1.0 / k)       # noqa: E731
print(f"  coarse {rho(kc[-1]) - rho(kc[0]):.0f}, fine {rho(kf[-1]) - rho(kf[0]):.0f}, "
      f"difference {(rho(kc[-1]) - rho(kc[0])) - (rho(kf[-1]) - rho(kf[0])):+.0f}")
d = 1e5 * (kc - kf)
print(f"\nk difference coarse - fine (pcm): min {d.min():+.1f} at {a[d.argmin()]:g}, "
      f"max {d.max():+.1f} at {a[d.argmax()]:g}")
uniform = np.allclose(np.diff(a), a[1] - a[0]) if a.size > 2 else False
if uniform:
    h = a[1] - a[0]
    for name, k in (("coarse", kc), ("fine", kf), ("difference", kc - kf)):
        dd = 1e5 * (k[2:] - 2 * k[1:-1] + k[:-2])
        print(f"second difference of {name} k over {h:g} deg (pcm): "
              f"{' '.join(f'{x:+.0f}' for x in dd)}")
    # midpoint interpolation of k on the 2h grid, each refinement against itself
    for name, k in (("coarse", kc), ("fine", kf)):
        mid = 1e5 * (0.5 * (k[:-2:2] + k[2::2]) - k[1:-1:2])
        print(f"linear interpolation of {name} k on {2 * h:g} deg, error at the "
              f"midpoints (pcm): {' '.join(f'{x:+.0f}' for x in mid)}")
print(f"\nworst over angles: |dk| {np.abs(d).max():.0f} pcm, assembly {r[:, 4].max():.2f} %, "
      f"pin RMS {r[:, 5].max():.2f} %, worst pin {np.abs(r[:, 6]).max():.2f} %")
print(f"total {time.perf_counter() - t0:.0f} s")
