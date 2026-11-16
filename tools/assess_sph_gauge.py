"""SPH diffusion-coefficient convention and normalization check (rebuilt ``sph_gauge.py``).

    python tools/assess_sph_gauge.py [--part A B C D E F G] [--marshak]

The October 2026 audit (item O5, docs/numerical_implementation_notes.md) quoted
an audit script ``sph_gauge.py`` that was not kept. Part A rebuilds it from the
notes: the leaky three-region PWR/poison colorset of tests/verification/
test_sph.py with UNEQUAL regions (x in [0, 30), [30, 45), [45, 90) cm), the SP3
reference with the per-moment Robin vacuum that was the default before the
audit (k_ref = 1.0992249, flux-volume homogenization +885 pcm), and
the pre-audit ``sph_correct(method="hybr")`` with ``norm_weight=None`` versus the
per-region volumes, under each ``D_SCALING``. That solver (one normalization
constant for all regions and groups) was removed from garnet.sph on branch
SPH-dev; ``sph_joint`` below is a line-for-line copy of it, so part A still
reproduces the historical table.

Parts B-D re-evaluate it (SPH audit 2026-11-13, docs/qa/sph_audit_2026-11-13.md):

B  Gauge check on reflective generation geometries. On a reflective problem the
   SPH factors are fixed only up to G constants, one per group (Labouré et al.
   2019, Sec. 3.1). With D~ = mu D a common rescaling of one group's factors is an
   exact symmetry of the corrected operator; with D~ = D / mu it is not. Each
   converged solution is perturbed by x1.3 in group 0 and in all groups.
C  The vacuum generation geometry with per-group flux-volume normalization
   (the standard, IGE-335 keyword STD) and with the fixed-k, no-normalization
   formulation of Labouré et al. 2019 Sec. 3.2 (k fixed at k_ref, mu Phi =
   phi_ref in absolute terms).
D  The production workflow: per-region factors generated on the REFLECTIVE
   colorset on a coarse node mesh (7.5 x 15 cm), then applied to the VACUUM
   colorset on the same coarse mesh. Reports how much the transferred eigenvalue
   and region powers depend on the D convention and the normalization.
E  Generation time of the SPH-corrected coarse model with v and with v/mu,
   against the fine heterogeneous diffusion model.

Parts F-G use the library solver garnet.sph_correct as changed on branch
SPH-dev (D~ = mu D, per-group flux-volume or fixed-k formulation):

F  Existence of the fixed-k solution against the coarse node size, on a
   water | fuel | stripe | fuel | water slab with vacuum x ends, all regions
   corrected or fuel only.
G  HP-MR 2-D (refine 4, raster absorber, TriSP3 reference, one region per
   material) at drums inserted and withdrawn: flux-volume, fixed-k all
   regions, fixed-k without the vacuum-facing Be reflector, fixed-k fuel only.

All coarse solves are Cartesian finite-volume diffusion. Reaction-rate errors are
the largest relative deviation of the region absorption / production rates,
each normalized to unit total production, from the reference.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.optimize import root

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import garnet.sph as S                                                 # noqa: E402
from garnet import (DiffusionEigenSolver, Grid, Material, PWR_TWO_GROUP,  # noqa: E402
                    SP3EigenSolver, flux_weighted_homogenize, region_average,
                    sph_correct)

POISON = Material(name="poison", diffusion=[1.15, 0.55], sigma_a=[0.009, 0.12],
                  nu_sigma_f=[0, 0], sigma_s=[[0, 0.03], [0, 0]])
MATS = [PWR_TWO_GROUP, POISON]
NX, NY = 36, 12                       # fine mesh, 2.5 x 2.5 cm
LX, LY = 90.0, 30.0
CUTS = (12, 18)                       # region boundaries in fine cells: 30 and 45 cm
COARSE = (12, 2)                      # coarse node mesh, 7.5 x 15 cm
VACUUM_X = ("vacuum", "reflective", "reflective")
REFLECTIVE = "reflective"
TIGHT = dict(tol_k=1e-12, tol_source=1e-11)


def fine_problem():
    grid = Grid(shape=(NX, NY, 1), size=(LX, LY, 1.0))
    mmap = np.zeros((NX, NY, 1), dtype=np.int64)
    mmap[NX // 3:2 * NX // 3, 4:8, :] = 1           # poison stripe
    region = np.zeros((NX, NY, 1), dtype=np.int64)
    region[CUTS[0]:CUTS[1]] = 1
    region[CUTS[1]:] = 2
    return grid, mmap, region, (LX / NX) * (LY / NY)


def coarse_problem():
    nx, ny = COARSE
    grid = Grid(shape=(nx, ny, 1), size=(LX, LY, 1.0))
    xc = (np.arange(nx) + 0.5) * LX / nx
    cut = np.array(CUTS) * LX / NX
    reg_x = np.digitize(xc, cut)
    region = np.broadcast_to(reg_x[:, None, None], (nx, ny, 1)).astype(np.int64).copy()
    return grid, region, (LX / nx) * (LY / ny)


def reference(bc, marshak):
    grid, mmap, region, dV = fine_problem()
    r = SP3EigenSolver(grid, MATS, material_map=mmap, bc=bc, device="cpu",
                       marshak_vacuum=marshak).solve(**TIGHT)
    assert r.converged
    hm, rflux, rvol = flux_weighted_homogenize(r.flux_numpy, MATS, mmap, region,
                                               cell_volume=dV)
    return r.k_eff, hm, rflux, rvol


def coarse_solver(bc, mesh):
    if mesh == "fine":
        grid, _, region, dV = fine_problem()
    else:
        grid, region, dV = coarse_problem()

    def solve(materials):
        res = DiffusionEigenSolver(grid, materials, material_map=region, bc=bc,
                                   device="cpu").solve(**TIGHT)
        return region_average(res.flux_numpy, region, cell_volume=dV), res.k_eff
    return solve


def corrected(hm, mu, d_scaling):
    return [S._scale_material(hm[i], mu[i], d_scaling=d_scaling) for i in range(len(hm))]


def rate_error(materials, Phi, ref_materials, rflux, V):
    def rates(mats, flux):
        sa = np.array([m.sigma_a for m in mats])
        nf = np.array([m.nu_sigma_f for m in mats])
        a, f = sa * flux * V[:, None], nf * flux * V[:, None]
        return a / f.sum(), f / f.sum()
    a1, f1 = rates(materials, Phi)
    a0, f0 = rates(ref_materials, rflux)
    return max(np.abs(a1 - a0).max() / np.abs(a0).max(),
               np.abs(f1 - f0).max() / np.abs(f0).max())


def region_power(materials, Phi, V):
    nf = np.array([m.nu_sigma_f for m in materials])
    p = (nf * Phi).sum(axis=1) * V
    return p / p.sum()


# --- SPH solves ------------------------------------------------------------------

def sph_joint(hm, rflux, solve, d_scaling, weight, tol=1e-10):
    """Pre-audit garnet.sph_correct(method="hybr", norm_weight=weight): one
    normalization constant for all regions and groups (weight None = plain
    sum, else the given (R, G) weights). Copied from garnet/sph.py at commit
    f70a090 (_sph_residual and the hybr branch of sph_correct)."""
    R, G = rflux.shape
    ref = np.asarray(rflux, dtype=float).reshape(-1)
    live = ref > ref.max() * 1e-12
    nw = np.ones(ref.size) if weight is None else np.asarray(weight, float).reshape(-1)
    ln_ref_n = np.zeros(ref.size)
    ln_ref_n[live] = np.log(ref[live] / (nw[live] * ref[live]).sum())

    def resid(x_live):
        x_live = np.clip(np.asarray(x_live, dtype=float), -8.0, 8.0)
        x = np.zeros(ref.size)
        x[live] = x_live
        Phi, _ = solve(corrected(hm, np.exp(x).reshape(R, G), d_scaling))
        live_phi = np.asarray(Phi, dtype=float).reshape(-1)[live]
        if not np.all(live_phi > 0.0) or not np.isfinite(live_phi).all():
            return 10.0 * x_live
        denom = live_phi.sum() if weight is None else (nw[live] * live_phi).sum()
        return (ln_ref_n[live] - np.log(live_phi / denom)) - x_live

    sol = root(resid, np.zeros(int(live.sum())), method="hybr",
               options=dict(xtol=max(tol * 1e-4, 1e-14)))
    x = np.zeros(ref.size)
    x[live] = sol.x
    return np.exp(x).reshape(R, G), bool(sol.success)


def sph_per_group(hm, rflux, rvol, solve, d_scaling):
    """Standard SPH: mu Phi = phi_ref with one flux-volume constant PER GROUP."""
    R, G = rflux.shape
    V = rvol[:, None]
    ref_n = rflux / (V * rflux).sum(axis=0, keepdims=True)

    def resid(x):
        mu = np.exp(x.reshape(R, G))
        Phi, _ = solve(corrected(hm, mu, d_scaling))
        Phi = np.asarray(Phi)
        Phi_n = Phi / (V * Phi).sum(axis=0, keepdims=True)
        return (np.log(ref_n) - np.log(Phi_n) - x.reshape(R, G)).ravel()

    sol = root(resid, np.zeros(R * G), method="hybr", options=dict(xtol=1e-13))
    return np.exp(sol.x.reshape(R, G)), bool(sol.success and np.abs(sol.fun).max() < 1e-9)


def sph_fixed_k(hm, rflux, rvol, solve, d_scaling, k_ref):
    """Labouré et al. 2019 Sec. 3.2: no normalization, k fixed at k_ref.

    Unknowns ln mu (R*G) and ln s (the absolute level of the eigenvector);
    conditions mu * s * Phi = phi_ref (R*G) and k(mu) = k_ref. Solved by
    Levenberg-Marquardt from the per-group flux-volume solution, so a case with
    no exact solution returns its least-squares floor instead of failing.

    A solution need not exist. With D~ = mu D a region-uniform mu leaves the
    flux shape inside each region unchanged and only moves the effective
    Marshak coefficient of a boundary region between alpha/mu -> 0 (reflective)
    and infinity (zero flux); a reference that leaks more than the zero-flux
    limit of the homogenized model cannot be matched."""
    from scipy.optimize import least_squares
    R, G = rflux.shape
    mu0, _ = sph_per_group(hm, rflux, rvol, solve, d_scaling)
    Phi0, _ = solve(corrected(hm, mu0, d_scaling))
    s0 = np.log(rflux.sum() / (mu0 * np.asarray(Phi0)).sum())

    def resid(z):
        mu = np.exp(np.clip(z[:-1], -6.0, 6.0).reshape(R, G))
        Phi, k = solve(corrected(hm, mu, d_scaling))
        r1 = np.log(mu * np.exp(z[-1]) * np.asarray(Phi) / rflux).ravel()
        return np.append(r1, np.log(k / k_ref))

    sol = least_squares(resid, np.append(np.log(mu0).ravel(), s0), method="lm",
                        xtol=1e-14, ftol=1e-14)
    mu = np.exp(np.clip(sol.x[:-1], -6.0, 6.0).reshape(R, G))
    return mu, bool(np.abs(sol.fun).max() < 1e-9)


def gauge(hm, mu, solve, d_scaling):
    k = solve(corrected(hm, mu, d_scaling))[1]
    m1 = mu.copy()
    m1[:, 0] *= 1.3
    k1 = solve(corrected(hm, m1, d_scaling))[1]
    k2 = solve(corrected(hm, mu * 1.3, d_scaling))[1]
    return 1e5 * (k1 - k), 1e5 * (k2 - k)


def fmt_mu(mu):
    return " ".join(f"({a:.4f},{b:.4f})" for a, b in mu)


# --- parts -------------------------------------------------------------------------

def part_a(marshak):
    print("\n=== A. Rebuilt sph_gauge.py: leaky colorset, unequal regions, method='hybr' ===")
    k_ref, hm, rflux, rvol = reference(VACUUM_X, marshak)
    solve = coarse_solver(VACUUM_X, "fine")
    k_h = solve(hm)[1]
    print(f"SP3 ({'Marshak' if marshak else 'Robin'} vacuum) k_ref = {k_ref:.7f}; "
          f"flux-volume homogenized {1e5 * (k_h - k_ref):+.1f} pcm")
    wvol = np.repeat(rvol[:, None], rflux.shape[1], axis=1)
    print(f"{'D_SCALING':10s} {'sum k':>10s} {'volume k':>10s} {'spread':>8s} "
          f"{'sum err':>9s} {'vol err':>9s}  scipy success  rate err (sum/vol)")
    for d in ("inverse", "direct"):
        mu_s, ok_s = sph_joint(hm, rflux, solve, d, None)
        mu_v, ok_v = sph_joint(hm, rflux, solve, d, wvol)
        Ps, ks = solve(corrected(hm, mu_s, d))
        Pv, kv = solve(corrected(hm, mu_v, d))
        es = rate_error(corrected(hm, mu_s, d), np.asarray(Ps), hm, rflux, rvol)
        ev = rate_error(corrected(hm, mu_v, d), np.asarray(Pv), hm, rflux, rvol)
        print(f"{d:10s} {ks:10.7f} {kv:10.7f} {1e5 * abs(ks - kv):7.0f}p "
              f"{1e5 * (ks - k_ref):+8.0f}p {1e5 * (kv - k_ref):+8.0f}p  {ok_s}/{ok_v}  "
              f"{es:.1e}/{ev:.1e}")


def part_b(marshak):
    print("\n=== B. Gauge check on REFLECTIVE generation geometries ===")
    k_ref, hm, rflux, rvol = reference(REFLECTIVE, marshak)
    wvol = np.repeat(rvol[:, None], rflux.shape[1], axis=1)
    for mesh in ("fine", "coarse"):
        solve = coarse_solver(REFLECTIVE, mesh)
        k_h = solve(hm)[1]
        print(f"\n[{mesh} mesh] reflective colorset: k_ref = {k_ref:.7f}, homogenized "
              f"{1e5 * (k_h - k_ref):+.1f} pcm")
        print(f"  {'D':7s} {'normalization':16s} {'conv':5s} {'k err':>7s} {'rate err':>9s} "
              f"{'g0 x1.3':>9s} {'all x1.3':>9s}  mu (fast,thermal) per region")
        for d in ("inverse", "direct"):
            for label, run in (("joint sum", lambda: sph_joint(hm, rflux, solve, d, None)),
                               ("joint volume", lambda: sph_joint(hm, rflux, solve, d, wvol)),
                               ("per-group FV", lambda: sph_per_group(hm, rflux, rvol, solve, d))):
                mu, ok = run()
                Phi, k = solve(corrected(hm, mu, d))
                err = rate_error(corrected(hm, mu, d), np.asarray(Phi), hm, rflux, rvol)
                g0, ga = gauge(hm, mu, solve, d)
                name = "D/mu" if d == "inverse" else "mu*D"
                print(f"  {name:7s} {label:16s} {str(ok):5s} {1e5 * (k - k_ref):+6.1f}p "
                      f"{err:9.1e} {g0:+8.2f}p {ga:+8.2f}p  {fmt_mu(mu)}")


def part_c(marshak):
    print("\n=== C. VACUUM generation geometry: standard and fixed-k formulations ===")
    k_ref, hm, rflux, rvol = reference(VACUUM_X, marshak)
    for mesh in ("fine", "coarse"):
        solve = coarse_solver(VACUUM_X, mesh)
        print(f"\n[{mesh} mesh] k_ref = {k_ref:.7f}, homogenized "
              f"{1e5 * (solve(hm)[1] - k_ref):+.1f} pcm")
        print(f"  {'D':7s} {'formulation':22s} {'conv':5s} {'k err':>8s} {'rate err':>9s} "
              f"{'g0 x1.3':>9s}  mu (fast,thermal) per region")
        for d in ("inverse", "direct"):
            for label, run in (("joint sum (default)", lambda: sph_joint(hm, rflux, solve, d, None)),
                               ("per-group FV", lambda: sph_per_group(hm, rflux, rvol, solve, d)),
                               ("fixed k, no norm.", lambda: sph_fixed_k(hm, rflux, rvol, solve, d, k_ref))):
                mu, ok = run()
                Phi, k = solve(corrected(hm, mu, d))
                err = rate_error(corrected(hm, mu, d), np.asarray(Phi), hm, rflux, rvol)
                g0, _ = gauge(hm, mu, solve, d)
                name = "D/mu" if d == "inverse" else "mu*D"
                print(f"  {name:7s} {label:22s} {str(ok):5s} {1e5 * (k - k_ref):+7.1f}p "
                      f"{err:9.1e} {g0:+8.2f}p  {fmt_mu(mu)}")


def part_d(marshak):
    print("\n=== D. Workflow (b): generate on reflective colorset, apply to vacuum colorset, "
          "coarse mesh ===")
    _, hm_r, rflux_r, rvol = reference(REFLECTIVE, marshak)
    k_vac, hm_v, rflux_v, _ = reference(VACUUM_X, marshak)
    gen = coarse_solver(REFLECTIVE, "coarse")
    app = coarse_solver(VACUUM_X, "coarse")
    p_ref = region_power(hm_v, rflux_v, rvol)
    wvol = np.repeat(rvol[:, None], rflux_r.shape[1], axis=1)
    Phi0, k0 = app(hm_r)
    print(f"vacuum reference k = {k_vac:.7f}; region power {np.array2string(p_ref, precision=4)}")
    print(f"  {'no SPH (reflective-lattice XS)':38s} k err {1e5 * (k0 - k_vac):+7.1f}p  "
          f"max power err {100 * np.abs(region_power(hm_r, Phi0, rvol) / p_ref - 1).max():.2f}%")
    for d in ("inverse", "direct"):
        ks = []
        for label, run in (("joint sum", lambda: sph_joint(hm_r, rflux_r, gen, d, None)),
                           ("joint volume", lambda: sph_joint(hm_r, rflux_r, gen, d, wvol)),
                           ("per-group FV", lambda: sph_per_group(hm_r, rflux_r, rvol, gen, d))):
            mu, ok = run()
            mats = corrected(hm_r, mu, d)
            Phi, k = app(mats)
            perr = 100 * np.abs(region_power(mats, np.asarray(Phi), rvol) / p_ref - 1).max()
            ks.append(k)
            name = "D/mu" if d == "inverse" else "mu*D"
            print(f"  {name:6s} {label:31s} conv={str(ok):5s} k err {1e5 * (k - k_vac):+7.1f}p  "
                  f"max power err {perr:.2f}%")
        print(f"  {'':6s} spread of transferred k over normalizations: "
              f"{1e5 * (max(ks) - min(ks)):.1f} pcm")


def part_e(marshak):
    """Generation time: fine heterogeneous diffusion vs SPH-corrected coarse
    diffusion with v and with v/mu. Under phi = mu Phi every term multiplying
    phi_g scales by mu, so 1/v becomes mu/v (Takasugi et al. 2023, Eq. 8, use
    the uncorrected 1/v in Griffin). The reference here is fine-mesh diffusion,
    so SPH folds only the spatial homogenization."""
    print("\n=== E. Generation time on the reflective colorset (two-group speeds 1e7, 3e5 cm/s) ===")
    v = np.array([1.0e7, 3.0e5])
    grid, mmap, region_f, dV = fine_problem()

    def lam(adj, fwd, inv_v, chi, nf, V):
        return (adj * inv_v * fwd * V).sum() / (((adj * chi).sum(0) * (nf * fwd).sum(0)) * V).sum()

    kw = dict(bc=REFLECTIVE, device="cpu")
    fwd = DiffusionEigenSolver(grid, MATS, material_map=mmap, **kw).solve(**TIGHT)
    adj = DiffusionEigenSolver(grid, MATS, material_map=mmap, **kw).solve(adjoint=True, **TIGHT)
    cells = mmap.reshape(-1)
    chi = np.array([m.chi for m in MATS])[cells].T
    nf = np.array([m.nu_sigma_f for m in MATS])[cells].T
    lam_ref = lam(adj.flux_numpy.reshape(2, -1), fwd.flux_numpy.reshape(2, -1),
                  (1 / v)[:, None], chi, nf, dV)
    hm, rflux, rvol = flux_weighted_homogenize(fwd.flux_numpy, MATS, mmap, region_f,
                                               cell_volume=dV)
    cgrid, cregion, cdV = coarse_problem()
    gen = coarse_solver(REFLECTIVE, "coarse")
    ci = cregion.reshape(-1)
    print(f"  fine heterogeneous diffusion: Lambda = {lam_ref:.4e} s")
    for d in ("direct", "inverse"):
        mu, ok = sph_per_group(hm, rflux, rvol, gen, d)
        mats = corrected(hm, mu, d)
        f = DiffusionEigenSolver(cgrid, mats, material_map=cregion, **kw).solve(**TIGHT)
        a = DiffusionEigenSolver(cgrid, mats, material_map=cregion, **kw).solve(adjoint=True, **TIGHT)
        cchi = np.array([m.chi for m in mats])[ci].T
        cnf = np.array([m.nu_sigma_f for m in mats])[ci].T
        F, Aj = f.flux_numpy.reshape(2, -1), a.flux_numpy.reshape(2, -1)
        l_v = lam(Aj, F, (1 / v)[:, None], cchi, cnf, cdV)
        l_m = lam(Aj, F, mu[ci].T / v[:, None], cchi, cnf, cdV)
        name = "D/mu" if d == "inverse" else "mu*D"
        print(f"  {name:5s} coarse SPH (conv={ok}): with v {l_v:.4e} ({100 * (l_v / lam_ref - 1):+.2f}%)"
              f" | with v/mu {l_m:.4e} ({100 * (l_m / lam_ref - 1):+.2f}%)")


WATER = Material(name="water", diffusion=[1.13, 0.16], sigma_a=[0.0004, 0.0197],
                 nu_sigma_f=[0, 0], sigma_s=[[0, 0.0494], [0, 0]])


def part_f(marshak):
    """Fixed-k existence against the coarse node size. Regions: water 0-15 cm,
    fuel 15-45, fuel with the poison stripe 45-75, fuel 75-105, water 105-120."""
    print("\n=== F. Fixed-k existence vs coarse node size (water-reflected slab, "
          "vacuum x) ===")
    nx, ny, h = 48, 12, 2.5
    mats = [PWR_TWO_GROUP, POISON, WATER]
    grid = Grid(shape=(nx, ny, 1), size=(nx * h, ny * h, 1.0))
    mmap = np.zeros((nx, ny, 1), dtype=np.int64)
    mmap[:6] = 2
    mmap[-6:] = 2
    mmap[18:30, 4:8] = 1
    cuts = [15.0, 45.0, 75.0, 105.0]

    def region_map(n_x, n_y, size):
        x = np.digitize((np.arange(n_x) + 0.5) * size, cuts)
        return np.broadcast_to(x[:, None, None], (n_x, n_y, 1)).astype(np.int64).copy()
    region = region_map(nx, ny, h)
    ref = SP3EigenSolver(grid, mats, material_map=mmap, bc=VACUUM_X, device="cpu",
                         marshak_vacuum=marshak).solve(**TIGHT)
    hm, rflux, rvol = flux_weighted_homogenize(ref.flux_numpy, mats, mmap, region,
                                               cell_volume=h * h)
    fuel = np.array([False, True, True, True, False])
    print(f"SP3 k_ref = {ref.k_eff:.7f}; water thermal diffusion length "
          f"{np.sqrt(WATER.diffusion[1] / WATER.sigma_a[1]):.2f} cm")
    print(f"  {'node':>6} {'homog':>8}  {'corrected':9s} {'fixed-k':>8} {'residual':>9} "
          f"{'k err':>9}  {'flux-volume k err':>17}")
    for size in (2.5, 3.75, 5.0, 7.5):
        cnx, cny = int(round(nx * h / size)), int(round(ny * h / size))
        cgrid = Grid(shape=(cnx, cny, 1), size=(nx * h, ny * h, 1.0))
        creg = region_map(cnx, cny, size)

        def solve(m, cgrid=cgrid, creg=creg, size=size):
            r = DiffusionEigenSolver(cgrid, m, material_map=creg, bc=VACUUM_X,
                                     device="cpu").solve(**TIGHT)
            return region_average(r.flux_numpy, creg, cell_volume=size * size), r.k_eff
        kh = solve(hm)[1]
        for label, mask in (("all", None), ("fuel only", fuel)):
            out = sph_correct(hm, None, rflux, solve, normalization="none",
                              k_ref=ref.k_eff, sph_regions=mask, tol=1e-9)
            fv = sph_correct(hm, None, rflux, solve, region_volume=rvol,
                             sph_regions=mask, tol=1e-9)
            print(f"  {size:5.2f}c {1e5 * (kh - ref.k_eff):+7.1f}p  {label:9s} "
                  f"{str(out.converged):>8} {out.residual:9.1e} "
                  f"{1e5 * (out.k_eff - ref.k_eff):+8.1f}p  "
                  f"{1e5 * (fv.k_eff - ref.k_eff):+8.1f}p (conv={fv.converged})")


def part_g(marshak):
    """HP-MR 2-D formulations (slow: two full-core TriSP3 solves)."""
    from garnet import TriDiffusionEigenSolver, TriSP3EigenSolver
    from garnet.benchmarks.hpmr import (BE_REFLECTOR, CENTRAL, DRUM_ABSORBER,
                                        DRUM_BE, FUEL, build_hpmr2d)
    print("\n=== G. HP-MR 2-D, refine 4, raster absorber, TriSP3 reference, one "
          "region per material ===")
    tight = dict(tol_k=1e-9, tol_source=1e-8)
    for angle in (0.0, 180.0):
        p = build_hpmr2d(refine=4, drum_angle_deg=angle, absorber="raster")
        common = dict(active=p.active, mask_bc=p.mask_bc)
        region = p.material_map
        ref = TriSP3EigenSolver(p.grid, p.materials, region, **common).solve(**tight)
        dif = TriDiffusionEigenSolver(p.grid, p.materials, region, **common).solve(**tight)
        hm, rflux, rvol = flux_weighted_homogenize(ref.flux_numpy, p.materials, region,
                                                   region, cell_volume=p.grid.cell_volume)

        def solve(m):
            r = TriDiffusionEigenSolver(p.grid, m, region, **common).solve(**tight)
            return region_average(r.flux_numpy, region), r.k_eff
        n = len(hm)
        interior = np.isin(np.arange(n), (FUEL, CENTRAL, DRUM_BE, DRUM_ABSORBER))
        print(f"\ndrum angle {angle:.0f} deg: TriSP3 k = {ref.k_eff:.7f}, diffusion "
              f"{1e5 * (dif.k_eff - ref.k_eff):+.1f} pcm")
        print(f"  {'formulation':28s} {'conv':5s} {'nfev':>4} {'k err':>8} {'all x1.3':>9}"
              f"  mu (fast, thermal): fuel | Be reflector | drum absorber")
        for label, kw in (("flux-volume, all", dict(region_volume=rvol, max_iter=300)),
                          ("fixed k, all", dict(normalization="none", k_ref=ref.k_eff)),
                          ("fixed k, no Be reflector", dict(normalization="none",
                                                            k_ref=ref.k_eff,
                                                            sph_regions=interior)),
                          ("fixed k, fuel only", dict(normalization="none", k_ref=ref.k_eff,
                                                      sph_regions=np.arange(n) == FUEL))):
            out = sph_correct(hm, region, rflux, solve, tol=1e-7, **kw)
            k13 = solve([S._scale_material(hm[i], 1.3 * out.factors[i]) for i in range(n)])[1]
            mu = out.factors
            print(f"  {label:28s} {str(out.converged):5s} {out.iterations:4d} "
                  f"{1e5 * (out.k_eff - ref.k_eff):+7.2f}p {1e5 * (k13 - out.k_eff):+8.2f}p  "
                  f"({mu[FUEL, 0]:.3f},{mu[FUEL, 1]:.3f}) | ({mu[BE_REFLECTOR, 0]:.3f},"
                  f"{mu[BE_REFLECTOR, 1]:.3f}) | ({mu[DRUM_ABSORBER, 0]:.3f},"
                  f"{mu[DRUM_ABSORBER, 1]:.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", nargs="*", default=["A", "B", "C", "D", "E"])
    ap.add_argument("--marshak", action="store_true",
                    help="SP3 reference with the coupled Marshak vacuum (current default); "
                         "the lost script used the per-moment Robin vacuum")
    args = ap.parse_args()
    for p in args.part:
        {"A": part_a, "B": part_b, "C": part_c, "D": part_d,
         "E": part_e, "F": part_f, "G": part_g}[p.upper()](args.marshak)


if __name__ == "__main__":
    main()
