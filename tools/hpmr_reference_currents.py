"""Cache the zone-surface partial currents of a pin-resolved HP-MR S_N reference.

    python tools/hpmr_reference_currents.py REFINE ANGLES METHOD [DEVICE] [CACHE] [ZONING]

ZONING: pooled (default) or hex (per-hex zones of the outer ring).

See ``load_or_solve_currents`` in tools/hpmr_pin_reference.py. The k of the
re-solve is printed next to the cached flux reference's, as a check that the
currents belong to the same solution.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hpmr_pin_reference import (DEFAULT_CACHE, cache_path,  # noqa: E402
                                load_or_solve_currents)

refine = int(sys.argv[1])
angles = [float(a) for a in sys.argv[2].split(",")]
method = sys.argv[3]
device = sys.argv[4] if len(sys.argv) > 4 else "auto"
cache = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else DEFAULT_CACHE
per_hex = (sys.argv[6] if len(sys.argv) > 6 else "pooled") == "hex"
for angle in angles:
    flux_file = cache_path(cache, method, refine, angle)
    k_cached = float(np.load(flux_file)["k"]) if os.path.exists(flux_file) else float("nan")
    c = load_or_solve_currents(refine, angle, method, device, cache, per_hex=per_hex)
    G = next(iter(c["boundaries"].values())).shape[0]
    total = sum(v[:, 0] - v[:, 1] for v in c["boundaries"].values())
    print(f"RESULT refine={refine} angle={angle:g} method={method} k={c['k']:.7f} "
          f"(cached flux k {k_cached:.7f}, diff {1e5 * (c['k'] - k_cached):+.2f} pcm); "
          f"boundary zones {sorted(c['boundaries'])}, {len(c['interfaces'])} interfaces, "
          f"G={G}, total boundary leakage {total.sum():.6e}", flush=True)
