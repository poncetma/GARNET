"""Smoothness in drum angle of SPH-DF tables built with different drum zonings.

    python tools/compare_sphdf_zonings.py LABEL=TABLES [LABEL=TABLES ...]

Each TABLES directory holds the sphdf_<angle>.npz files written by
``tools/assess_sph_df_tables.py generate``. Tables without stored zone names
(written before the names were saved) are taken to use the per-arc zoning:
per-hex zones with a separate absorber-arc zone per drum and the inner/outer
boundary split.

For every zone the script measures how far a quantity departs from linear
interpolation between its neighbours on the table grid,

    defect(a) = |q(a) - (q(a - s) + q(a + s)) / 2| / |q(a)|,

with s the grid step. This is the error a table at spacing 2 s makes at angle a,
up to second order. It is reported for the thermal SPH factor mu, for the
corrected thermal absorption mu * Sigma_a (what the node model sees), and for
the boundary factors, as the worst over zones of each class and over the
interior angles. A table that interpolates well has small defects; an artefact
that changes irregularly with angle shows up as a large one.
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def per_arc_names():
    from sph_hpmr_nodes import split_boundary_layer

    from garnet.benchmarks.hpmr import build_hpmr2d, hpmr_assembly_zones
    q = build_hpmr2d(refine=8, drum_angle_deg=0.0, absorber="polar")
    z, _, _ = split_boundary_layer(hpmr_assembly_zones(q, per_hex=True), q)
    return list(z.names)


def zone_class(name):
    if name.endswith(":outer"):
        return "drum outer" if name.startswith("drum") else "Be outer"
    if name.startswith("fuel"):
        return "fuel"
    if name.startswith("be_reflector"):
        return "Be reflector"
    if name.startswith("drum_absorber"):
        return "drum absorber arc"
    if name.startswith("drum_be"):
        return "drum body"
    if re.search(r":core$", name):
        return "drum core"
    if re.search(r":s\d+$", name):
        return "drum ring sector"
    return "other"


def load(path):
    tables = {}
    for f in glob.glob(os.path.join(path, "sphdf_*.npz")):
        d = np.load(f)
        tables[float(d["angle"])] = {k: d[k] for k in d.files}
    return dict(sorted(tables.items()))


def defects(angles, values):
    """(n_interior, ...) relative midpoint defects and the interior angles."""
    a = np.asarray(angles)
    steps = np.diff(a)
    keep = [i for i in range(1, len(a) - 1) if np.isclose(steps[i - 1], steps[i])]
    v = np.asarray(values)
    out = np.array([np.abs(v[i] - 0.5 * (v[i - 1] + v[i + 1])) /
                    np.maximum(np.abs(v[i]), 1e-300) for i in keep])
    return out, a[keep]


SITES = ("(0, 3)", "(2, 2)")          # a corner and a mid-edge drum
CLASSES = ("fuel", "Be reflector", "Be outer", "drum body", "drum absorber arc", "drum core",
           "drum ring sector", "drum outer")

args = [a.split("=", 1) for a in sys.argv[1:]]
if not args:
    raise SystemExit(__doc__)
legacy = None
for label, path in args:
    tab = load(path)
    angles = list(tab)
    first = tab[angles[0]]
    if "names" in first:
        names = [str(n) for n in first["names"]]
    else:
        legacy = legacy or per_arc_names()
        names = legacy
    cls = np.array([zone_class(n) for n in names])
    print(f"== {label}: {path}\n   {len(names)} zones, angles {angles}")
    print(f"   {'angle':>6} {'conv':>5} {'solves':>6} {'residual':>9} {'dk pcm':>7} "
          f"{'max drum th mu':>15}  zone")
    drum = np.array([n.startswith("drum") for n in names])
    for a, t in tab.items():
        mu = t["mu"][:, -1]
        i = int(np.argmax(np.where(drum, mu, -np.inf)))
        print(f"   {a:6.1f} {str(bool(t['converged'])):>5} {int(t['iterations']):6d} "
              f"{float(t['residual']):9.1e} {1e5 * float(t['k'] - t['k_ref']):+7.2f} "
              f"{mu[i]:15.3g}  {names[i]}")
    mu = np.array([tab[a]["mu"][:, -1] for a in angles])                 # (A, Z)
    sa = np.array([tab[a]["sa"][:, -1] for a in angles])
    gam = np.array([tab[a]["gamma"] for a in angles])                   # (A, B, G)
    d_mu, interior = defects(angles, mu)
    d_rate, _ = defects(angles, mu * sa)
    d_gam, _ = defects(angles, gam)
    print(f"\n   Linear-interpolation defect at spacing {2 * (angles[1] - angles[0]):g} deg "
          f"(interior angles {list(interior)}), worst zone of each class, %:")
    print(f"   {'class':>18} {'zones':>5} {'thermal mu':>11} {'mu*Sigma_a':>11}  worst zone (mu) @ angle")
    for c in CLASSES:
        z = np.flatnonzero(cls == c)
        if z.size == 0:
            continue
        sub = d_mu[:, z]
        ia, iz = np.unravel_index(int(np.argmax(sub)), sub.shape)
        print(f"   {c:>18} {z.size:5d} {100 * sub.max():11.2f} {100 * d_rate[:, z].max():11.2f}"
              f"  {names[z[iz]]} @ {interior[ia]:g}")
    print(f"   {'boundary factors':>18} {gam.shape[1]:5d} {100 * d_gam.max():11.2f} "
          f"(all groups)\n")
    # thermal mu of the zones of one corner and one mid-edge drum against angle
    for site in SITES:
        z = [i for i, n in enumerate(names) if n.startswith("drum") and site in n]
        print(f"   thermal mu, zones of the drum at {site}:")
        print("   " + f"{'angle':>6} " + " ".join(
            f"{(n.split(':')[-1] if ':' in n else n.split('(')[0])[:13]:>13}"
            for n in (names[i] for i in z)))
        for j, a in enumerate(angles):
            print("   " + f"{a:6.1f} " + " ".join(f"{mu[j, i]:13.4g}" for i in z))
        print()
