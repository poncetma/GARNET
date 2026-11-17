"""Compare the cached pin-resolved HP-MR references (tools/hpmr_pin_reference.py).

    python tools/compare_hpmr_pin_references.py [CACHE] [ANGLES] [REFERENCE]

REFERENCE is a cache tag and refine, e.g. ``sn-p2416@32``; by default the
finest anisotropic S_N solve (``sn-p*``), else the finest S_N of any kind.

ANGLES is a comma list of drum angles (default 0,180); runs missing one are skipped.

For every cached solve: k at each drum angle, drum worth (180 -> 0 degrees,
in pcm of reactivity), and the errors against the reference -- the finest S_N
solve in the cache (highest refine, then most ordinates) -- in k, assembly
power (max) and pin power (RMS and max over the 1890 fuel compacts, powers
normalized to mean 1). Assembly power is kappaSigma_f phi summed over each
fuel assembly; pin power is kappaSigma_f phi of the fuel compact over its
exact footprint, so meshes of different refine compare pin for pin.
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from garnet.benchmarks.hpmr import hpmr_assembly_zones  # noqa: E402
from garnet.benchmarks.hpmr_assembly import FUEL as FUEL_COMPACT  # noqa: E402
from garnet.benchmarks.hpmr_core_pins import (PIN_OFFSET, R_FUEL,  # noqa: E402
                                              build_hpmr2d_pins)
from garnet.pin_power import footprint_flux, pin_footprints  # noqa: E402
from garnet.power import fission_energy_xs  # noqa: E402

cache = sys.argv[1] if len(sys.argv) > 1 else "/data/scratch/shared/poncet_m/garnet-sph-ref-net"
pattern = re.compile(r"(.+)_r(\d+)_a([\d.]+)\.npz$")
runs = {}
for path in glob.glob(os.path.join(cache, "*.npz")):
    m = pattern.search(os.path.basename(path))
    if m:
        runs[(m.group(1), int(m.group(2)), float(m.group(3)))] = path
if not runs:
    raise SystemExit(f"no cached references in {cache}")

geometry = {}


def edits(refine, angle, flux):
    key = (refine, angle)
    if key not in geometry:
        p, pins = build_hpmr2d_pins(refine=refine, drum_angle_deg=angle)
        z = hpmr_assembly_zones(p)
        table, _ = fission_energy_xs(p.materials)
        W = pin_footprints(p.raster, pins.xy[pins.fuel], R_FUEL)
        geometry[key] = (p, z, table, W)
    p, z, table, W = geometry[key]
    G = flux.shape[0]
    mm = p.material_map.reshape(-1)
    # the volume-mixed drum arc carries no fission, so the base material's
    # table is the cell's fission table everywhere
    dens = (table[mm].T * flux.reshape(G, -1)).sum(axis=0)
    zone = z.region_map.reshape(-1)
    assembly = np.bincount(zone, weights=dens * p.grid.cell_volume,
                           minlength=len(z.names))[z.fuel]
    pin = (table[PIN_OFFSET + FUEL_COMPACT] * footprint_flux(W, flux)).sum(axis=1)
    return assembly / assembly.mean(), pin / pin.mean()


def order(method):
    """Sort key: anisotropic S_N last (most trusted), then by tag."""
    return (method.startswith("sn-p"), method.startswith("sn"), method)


angles = [float(a) for a in (sys.argv[2] if len(sys.argv) > 2 else "0,180").split(",")]
complete = sorted({(m, r) for m, r, _ in runs
                   if all((m, r, a) in runs for a in angles)})
if len(sys.argv) > 3:
    tag, _, ref_r = sys.argv[3].partition("@")
    ref_method, ref_refine = tag, int(ref_r)
else:
    pool = ([t for t in complete if t[0].startswith("sn-p")]
            or [t for t in complete if t[0].startswith("sn")])
    ref_method, ref_refine = max(pool, key=lambda t: (t[1], order(t[0])))
print(f"cache {cache}; reference = {ref_method} at refine {ref_refine}\n")
ref = {}
for a in angles:
    d = np.load(runs[(ref_method, ref_refine, a)])
    ref[a] = (float(d["k"]),) + edits(ref_refine, a, d["flux"])

print(f"{'method':>10} {'refine':>6} " + " ".join(f"{'k(' + f'{a:g}' + ')':>10}" for a in angles)
      + f" {'worth':>8} " + " ".join(f"{'dk(' + f'{a:g}' + ')':>9} {'dPasm':>6} {'pinRMS':>6} {'pinMax':>6}"
                                       for a in angles) + f" {'t[s]':>6}")
for method, refine in sorted(complete, key=lambda t: (order(t[0]), t[1])):
    if not all((method, refine, a) in runs for a in angles):
        continue
    ks, cells, secs = [], [], 0.0
    for a in angles:
        d = np.load(runs[(method, refine, a)])
        k = float(d["k"])
        ks.append(k)
        secs += float(d["seconds"])
        asm, pin = edits(refine, a, d["flux"])
        kr, asm_r, pin_r = ref[a]
        dp = 100 * np.abs(asm / asm_r - 1).max()
        e = pin / pin_r - 1
        cells.append(f"{1e5 * (k - kr):+8.1f}p {dp:5.2f}% {100 * np.sqrt((e ** 2).mean()):5.2f}% "
                     f"{100 * np.abs(e).max():5.2f}%")
    worth = (1 / ks[0] - 1 / ks[-1]) * 1e5 if len(ks) > 1 else float("nan")
    print(f"{method:>10} {refine:6d} " + " ".join(f"{k:10.6f}" for k in ks)
          + f" {worth:7.0f}p " + " ".join(cells) + f" {secs:6.0f}")
