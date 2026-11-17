"""How far is the homogenized HP-MR model from its reference before SPH?

    python tools/diagnose_hpmr_homogenization.py REFINE ANGLES CASES [NODES] [DEVICE] [CACHE]

    REFINE : refine of the cached pin-resolved reference (tools/hpmr_pin_reference.py)
    ANGLES : comma list of drum angles
    CASES  : comma list of METHOD@TC: the reference method (its flux and k) and
             the fine data the zone constants are collapsed from
             (auto, none, outscatter, diagonal; see ``pin_reference`` in
             tools/sph_hpmr_nodes.py). METHOD alone means METHOD@auto.
    NODES  : comma list of node specs (default raster:REFINE,raster:4); a
             raster at the reference refine isolates the homogenization from
             the node mesh
    DEVICE : auto (default), gpu or cpu
    CACHE  : reference cache (default that of tools/hpmr_pin_reference.py)

Each line: the zone model with no SPH, solved by diffusion on the node mesh,
against the reference k. The node solve sees only the zone constants, so the
difference between two TC choices on the same flux is the homogenized
diffusion coefficient alone (absorption, fission and transfers do not depend
on the transport correction).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sph_hpmr_nodes import node_model, pin_reference  # noqa: E402

refine = int(sys.argv[1])
angles = [float(a) for a in sys.argv[2].split(",")]
cases = [c if "@" in c else c + "@auto" for c in sys.argv[3].split(",")]
nodes = (sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4]
         else [f"raster:{refine}", "raster:4"])
device = sys.argv[5] if len(sys.argv) > 5 else "auto"
cache = sys.argv[6] if len(sys.argv) > 6 else None
print(f"HP-MR zone homogenization before SPH: reference refine {refine}, "
      f"device {device}\n", flush=True)
print(f"{'angle':>6} {'reference':>14} {'homog. TC':>10} {'k_ref':>9} "
      f"{'fuel D_thermal':>14} | " + " | ".join(f"{s:>16}" for s in nodes), flush=True)
for angle in angles:
    models = {s: node_model(s, angle, device) for s in nodes}
    for case in cases:
        method, tc = case.split("@")
        ref = pin_reference(refine, method, angle, False, device, cache,
                            homogenize_tc=None if tc == "none" else tc)
        fuel = ref["zones"].fuel
        d_th = np.array([ref["hm"][i].diffusion[-1] for i in np.flatnonzero(fuel)])
        cells = []
        for s in nodes:
            t0 = time.perf_counter()
            _, k = models[s]["solve"](ref["hm"])
            cells.append(f"{1e5 * (k - ref['k']):+8.0f}p {time.perf_counter() - t0:4.0f}s")
        print(f"{angle:6.1f} {method:>14} {str(ref['homogenize_tc']):>10} "
              f"{ref['k']:9.6f} {d_th.mean():14.3f} | " + " | ".join(f"{c:>16}" for c in cells),
              flush=True)
