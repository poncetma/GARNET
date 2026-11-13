"""HP-MR 2D (drums inserted, raster absorber, refine 4): SPH from a TriSP3
reference with D_SCALING = 'inverse' (garnet default) and 'direct' (literature),
plus the gauge test and the region reaction-rate self-consistency check.
Mirrors tests/validation/test_hpmr_sph.py. Read-only use of garnet."""
import sys
import time

import numpy as np

sys.path.insert(0, "<repo>")
import garnet.sph as S                                           # noqa: E402
from garnet import (TriDiffusionEigenSolver, TriSP3EigenSolver,   # noqa: E402
                    flux_weighted_homogenize, region_average, sph_correct)
from garnet.benchmarks.hpmr import build_hpmr2d, MATERIAL_NAMES  # noqa: E402

TIGHT = dict(tol_k=1e-9, tol_source=1e-8)
refine = int(sys.argv[1]) if len(sys.argv) > 1 else 4
angle = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

p = build_hpmr2d(refine=refine, drum_angle_deg=angle, absorber="raster")
dV = p.grid.cell_volume
common = dict(active=p.active, mask_bc=p.mask_bc, device="cpu")
region = p.material_map
R = int(region.max()) + 1

t0 = time.perf_counter()
ref = TriSP3EigenSolver(p.grid, p.materials, region, **common).solve(**TIGHT)
t_ref = time.perf_counter() - t0
dif = TriDiffusionEigenSolver(p.grid, p.materials, region, **common).solve(**TIGHT)
assert ref.converged and dif.converged
hmats, rflux, rvol = flux_weighted_homogenize(ref.flux_numpy, p.materials, region,
                                              region, cell_volume=dV)
print(f"HP-MR 2D refine={refine} angle={angle}: cells={region.size}, R={R} regions "
      f"(one per material), G={hmats[0].n_groups}")
print(f"k_ref(TriSP3) = {ref.k_eff:.7f} ({t_ref:.0f} s)   k_diff = {dif.k_eff:.7f} "
      f"({1e5 * (dif.k_eff - ref.k_eff):+.1f} pcm)")
print("region volumes:", np.array2string(rvol, precision=0))
print("reference region flux (R,G):\n", np.array2string(rflux, precision=4))


def coarse(materials):
    r = TriDiffusionEigenSolver(p.grid, materials, region, **common).solve(**TIGHT)
    return region_average(r.flux_numpy, region), r.k_eff


def rates(materials, Phi):
    sa = np.array([m.sigma_a for m in materials])
    nf = np.array([m.nu_sigma_f for m in materials])
    V = rvol[:, None]
    a, f = sa * Phi * V, nf * Phi * V
    return a / f.sum(), f / f.sum()


a_ref, f_ref = rates(hmats, rflux)
for d_scaling in ("inverse", "direct"):
    S.D_SCALING = d_scaling
    t0 = time.perf_counter()
    out = sph_correct(hmats, region, rflux, coarse, tol=1e-7, depth=6, max_iter=300)
    dt = time.perf_counter() - t0
    Phi, k = coarse(out.corrected_materials)
    a, f = rates(out.corrected_materials, Phi)
    live = rvol > 0
    err_a = np.max(np.abs(a - a_ref)) / np.max(np.abs(a_ref))
    err_f = np.max(np.abs(f - f_ref)) / np.max(np.abs(f_ref))
    mu = out.factors
    mu_g = mu.copy(); mu_g[:, 0] *= 1.3
    k_g = coarse([S._scale_material(hmats[i], mu_g[i]) for i in range(R)])[1]
    mu_a = mu * 1.3
    k_a = coarse([S._scale_material(hmats[i], mu_a[i]) for i in range(R)])[1]
    print(f"\n--- D_SCALING = {d_scaling!r}: conv={out.converged} its={out.iterations} "
          f"({dt:.0f} s)  k_sph = {k:.7f} ({1e5 * (k - ref.k_eff):+.1f} pcm vs SP3)")
    print(f"    self-consistency: absorption {err_a:.1e}, production {err_f:.1e} "
          f"(max rel err of normalized region rates)")
    print(f"    gauge: group-0 x1.3 -> dk = {1e5 * (k_g - k):+.1f} pcm; all x1.3 -> "
          f"dk = {1e5 * (k_a - k):+.1f} pcm")
    for i, name in enumerate(MATERIAL_NAMES):
        if i < R:
            print(f"    mu[{name:14s}] = {np.array2string(mu[i], precision=4)}")
S.D_SCALING = "inverse"
