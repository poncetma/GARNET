"""Numerical checks for the SPH audit (read-only use of garnet).

1. Reflective 3-ring assembly and leaky 3-column colorset (geometries copied
   from tests/verification/test_sph.py).
2. For D_SCALING in ("inverse", "direct"):
     - run garnet.sph.sph_correct (default joint-sum normalization, Anderson)
     - self-consistency: SPH-corrected coarse region reaction rates vs the
       reference region reaction rates, both normalized to unit total fission
       production
     - gauge test: multiply the converged factors of ONE group by 1.3 (per-group
       gauge) and ALL factors by 1.3 (global gauge); re-solve; report dk
     - per-group flux-volume-normalized SPH (standard Hebert normalization),
       solved independently with scipy root(hybr); compare k and factors
3. Leaky colorset: also garnet's norm_weight="none" (absolute) mode.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "<repo>")
import garnet.sph as S                                           # noqa: E402
from garnet import (DiffusionEigenSolver, Grid, Material, PWR_TWO_GROUP,   # noqa: E402
                    SP3EigenSolver, flux_weighted_homogenize, region_average,
                    sph_correct)

POISON = Material(name="poison", diffusion=[1.15, 0.55], sigma_a=[0.009, 0.12],
                  nu_sigma_f=[0, 0], sigma_s=[[0, 0.03], [0, 0]])


def reflective_assembly():
    mats = [PWR_TWO_GROUP, POISON]
    n = 32
    grid = Grid(shape=(n, n, 1), size=(40.0, 40.0, 1.0))
    dV = (40.0 / n) ** 2
    mmap = np.zeros((n, n, 1), dtype=np.int64)
    c = n // 2
    for di in range(-3, 4):
        for dj in range(-3, 4):
            if abs(di) + abs(dj) <= 3:
                mmap[c + di, c + dj, 0] = 1
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    rr = np.sqrt((ii - c + 0.5) ** 2 + (jj - c + 0.5) ** 2)
    region = np.digitize(rr, [6, 12]).reshape(n, n, 1).astype(np.int64)
    return grid, mats, mmap, region, dV, "reflective"


def leaky_colorset():
    mats = [PWR_TWO_GROUP, POISON]
    n = 36
    grid = Grid(shape=(n, 12, 1), size=(90.0, 30.0, 1.0))
    dV = (90.0 / n) * (30.0 / 12)
    mmap = np.zeros((n, 12, 1), dtype=np.int64)
    mmap[n // 3:2 * n // 3, 4:8, :] = 1
    region = np.zeros((n, 12, 1), dtype=np.int64)
    region[n // 3:2 * n // 3, :, :] = 1
    region[2 * n // 3:, :, :] = 2
    return grid, mats, mmap, region, dV, ("vacuum", "reflective", "reflective")


def rates(materials, region_flux, region_volume):
    sa = np.array([m.sigma_a for m in materials])
    nf = np.array([m.nu_sigma_f for m in materials])
    out = np.array([m.sigma_s.sum(axis=1) - np.diag(m.sigma_s) for m in materials])
    V = np.asarray(region_volume)[:, None]
    r = dict(absorption=sa * region_flux * V, production=nf * region_flux * V,
             outscatter=out * region_flux * V)
    tot = r["production"].sum()
    return {k: v / tot for k, v in r.items()}


def max_rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def per_group_fv_sph(hmats, rflux, rvol, coarse, tol=1e-12):
    """Standard SPH: mu_ig Phi_ig = phi_ref_ig with per-GROUP flux-volume
    normalization sum_i V_i Phi_ig = sum_i V_i phi_ref_ig. Solved with hybr."""
    from scipy.optimize import root
    R, G = rflux.shape
    V = np.asarray(rvol)[:, None]
    ref_n = rflux / (V * rflux).sum(axis=0, keepdims=True)
    nfev = [0]

    def resid(x):
        nfev[0] += 1
        mu = np.exp(x.reshape(R, G))
        Phi, k = coarse([S._scale_material(hmats[i], mu[i]) for i in range(R)])
        Phi = np.asarray(Phi)
        Phi_n = Phi / (V * Phi).sum(axis=0, keepdims=True)
        return (np.log(ref_n) - np.log(Phi_n) - x.reshape(R, G)).ravel()

    sol = root(resid, np.zeros(R * G), method="hybr", options=dict(xtol=1e-13))
    mu = np.exp(sol.x.reshape(R, G))
    Phi, k = coarse([S._scale_material(hmats[i], mu[i]) for i in range(R)])
    return mu, k, np.asarray(Phi), sol.success, nfev[0], float(np.abs(sol.fun).max())


def run(name, build):
    grid, mats, mmap, region, dV, bc = build()
    t0 = time.perf_counter()
    ref = SP3EigenSolver(grid, mats, material_map=mmap, bc=bc, device="cpu"
                         ).solve(tol_k=1e-11, tol_source=1e-10)
    assert ref.converged
    hmats, rflux, rvol = flux_weighted_homogenize(ref.flux_numpy, mats, mmap, region,
                                                  cell_volume=dV)
    R, G = rflux.shape

    def coarse(materials):
        res = DiffusionEigenSolver(grid, materials, material_map=region, bc=bc,
                                   device="cpu").solve(tol_k=1e-12, tol_source=1e-11)
        return region_average(res.flux_numpy, region), res.k_eff

    k_h = coarse(hmats)[1]
    r_ref = rates(hmats, rflux, rvol)
    print(f"\n=== {name}: R={R}, G={G}, bc={bc} ===")
    print(f"k_ref(SP3) = {ref.k_eff:.7f}   k_homog(diffusion, flux-volume XS) = {k_h:.7f} "
          f"({1e5 * (k_h - ref.k_eff):+.1f} pcm)   [{time.perf_counter() - t0:.1f} s]")

    for d_scaling in ("inverse", "direct"):
        S.D_SCALING = d_scaling
        print(f"\n--- D_SCALING = {d_scaling!r} ---")
        modes = [("garnet joint-sum (default)", dict()),
                 ("garnet hybr joint-sum", dict(method="hybr"))]
        if isinstance(bc, tuple):
            modes.append(("garnet absolute ('none')", dict(norm_weight="none", method="hybr")))
        for label, kw in modes:
            try:
                out = sph_correct(hmats, region, rflux, coarse, tol=1e-10, max_iter=400,
                                  region_volume=rvol, **kw)
            except Exception as e:                                    # noqa: BLE001
                print(f"  {label:32s} FAILED: {e}")
                continue
            Phi, k_s = coarse(out.corrected_materials)
            r_sph = rates(out.corrected_materials, Phi, rvol)
            errs = {k: max_rel(r_sph[k], r_ref[k]) for k in r_ref}
            mu = out.factors
            # gauge tests
            mu_g = mu.copy(); mu_g[:, 0] *= 1.3
            k_g = coarse([S._scale_material(hmats[i], mu_g[i]) for i in range(R)])[1]
            mu_a = mu * 1.3
            k_a = coarse([S._scale_material(hmats[i], mu_a[i]) for i in range(R)])[1]
            print(f"  {label:32s} conv={out.converged!s:5s} nfev={out.iterations:4d} "
                  f"k_sph={k_s:.7f} ({1e5 * (k_s - ref.k_eff):+8.1f} pcm vs ref)")
            print(f"      self-consistency (normalized rates, max rel err): "
                  + ", ".join(f"{k}={v:.1e}" for k, v in errs.items()))
            print(f"      gauge: group-0 factors x1.3 -> dk = {1e5 * (k_g - k_s):+8.2f} pcm; "
                  f"all factors x1.3 -> dk = {1e5 * (k_a - k_s):+8.2f} pcm")
            print(f"      mu = {np.array2string(mu, precision=4, suppress_small=True)}")
        # standard per-group flux-volume normalization, independent solve
        mu, k_s, Phi, ok, nfev, res = per_group_fv_sph(hmats, rflux, rvol, coarse)
        r_sph = rates([S._scale_material(hmats[i], mu[i]) for i in range(R)], Phi, rvol)
        errs = {k: max_rel(r_sph[k], r_ref[k]) for k in r_ref}
        print(f"  {'per-group flux-volume (hybr)':32s} conv={ok!s:5s} nfev={nfev:4d} "
              f"k_sph={k_s:.7f} ({1e5 * (k_s - ref.k_eff):+8.1f} pcm vs ref)  |res|={res:.1e}")
        print(f"      self-consistency (normalized rates, max rel err): "
              + ", ".join(f"{k}={v:.1e}" for k, v in errs.items()))
        print(f"      mu = {np.array2string(mu, precision=4, suppress_small=True)}")
    S.D_SCALING = "inverse"


if __name__ == "__main__":
    run("reflective 3-ring assembly", reflective_assembly)
    run("leaky 3-column colorset", leaky_colorset)
