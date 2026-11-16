"""Superhomogenization pipeline (garnet.sph).

Step 1 -- flux-weighted homogenization. Collapsing a fine reference solution to
one Material per coarse region must preserve that region's reaction rates
exactly: Sigma_hom * <phi>_region * V_region equals the sum of the fine
cell reaction rates. This holds by construction of the flux-and-volume weighting
and is the property the SPH factor solve relies on to recover the reference
eigenvalue (not just its flux shape).

Step 2 -- the factor solve. The checks follow the self-consistency test of the
2026-11-13 SPH audit (docs/qa/sph_audit_2026-11-13.md, Sec. 6): on the
generation geometry the corrected coarse solve reproduces the reference region
rates; on a reflective geometry k is invariant to the per-group normalization
gauge (which needs D~ = mu D); on a vacuum-bounded geometry k is reproduced only
by the fixed-k, no-normalization formulation; the coarse mesh may be a node mesh
with its own region map and volumes.
"""

import numpy as np

import pytest

from garnet import (DiffusionEigenSolver, Grid, Kinetics, Material, PWR_TWO_GROUP,
                   SP3EigenSolver, flux_weighted_homogenize, region_average,
                   sph_correct, sph_kinetics)
from garnet.sph import _scale_material

POISON = Material(name="poison", diffusion=[1.15, 0.55], sigma_a=[0.009, 0.12],
                  nu_sigma_f=[0, 0], sigma_s=[[0, 0.03], [0, 0]])
WATER = Material(name="water", diffusion=[1.13, 0.16], sigma_a=[0.0004, 0.0197],
                 nu_sigma_f=[0, 0], sigma_s=[[0, 0.0494], [0, 0]])
TIGHT = dict(tol_k=1e-12, tol_source=1e-11)
VACUUM_X = ("vacuum", "reflective", "reflective")


def test_sph_material_keeps_the_fission_energy_power_coefficient():
    """An SPH model handed to thermal coupling must not change power weighting
    from kappa*Sigma_f to the nu*Sigma_f fallback."""
    mat = Material(name="fuel", diffusion=[1.2, 0.3], sigma_a=[0.01, 0.1],
                   nu_sigma_f=[0.02, 0.2], kappa_fission=[0.015, 0.11])
    mu = np.array([1.1, 0.8])
    corrected = _scale_material(mat, mu)
    np.testing.assert_allclose(corrected.kappa_fission,
                               mat.kappa_fission * mu)


def test_unknown_d_scaling_is_rejected_not_defaulted():
    """A typo such as 'Direct' used to fall through silently to 'inverse'."""
    mat = Material(diffusion=[1.2, 0.3], sigma_a=[0.01, 0.1], nu_sigma_f=[0.0, 0.0])
    _scale_material(mat, np.array([1.1, 0.9]), d_scaling="direct")
    with pytest.raises(ValueError, match="d_scaling"):
        _scale_material(mat, np.array([1.1, 0.9]), d_scaling="Direct")


def test_per_cell_volume_array_matches_scalar_volume():
    """cell_volume may be a (*shape) array (documented); it used to be broadcast
    against the already-flattened map and crash on any multi-dimensional grid."""
    rng = np.random.default_rng(3)
    shape = (5, 4, 1)
    mats = [PWR_TWO_GROUP, Material(name="refl", diffusion=[1.2, 0.9],
                                    sigma_a=[2e-4, 5e-3], nu_sigma_f=[0, 0],
                                    sigma_s=[[0, 0.04], [0, 0]])]
    mmap = rng.integers(0, 2, size=shape)
    region = np.zeros(shape, dtype=np.int64)
    region[2:] = 1
    flux = rng.uniform(0.5, 2.0, size=(2, *shape))
    a, fa, va = flux_weighted_homogenize(flux, mats, mmap, region, cell_volume=2.5)
    b, fb, vb = flux_weighted_homogenize(flux, mats, mmap, region,
                                         cell_volume=np.full(shape, 2.5))
    np.testing.assert_allclose(fb, fa, rtol=1e-14)
    np.testing.assert_allclose(vb, va, rtol=1e-14)
    for ma, mb in zip(a, b):
        np.testing.assert_allclose(mb.sigma_a, ma.sigma_a, rtol=1e-14)
    # a non-uniform volume array is honoured cell by cell
    vol = rng.uniform(0.5, 1.5, size=shape)
    c, _, vc = flux_weighted_homogenize(flux, mats, mmap, region, cell_volume=vol)
    np.testing.assert_allclose(vc, [vol[region == i].sum() for i in range(2)],
                               rtol=1e-14)


def _reference():
    absorber = Material(name="poison", diffusion=[1.1, 0.5], sigma_a=[0.01, 0.20],
                        nu_sigma_f=[0, 0], sigma_s=[[0, 0.03], [0, 0]])
    mats = [PWR_TWO_GROUP, absorber]
    n = 24
    grid = Grid(shape=(n, n, 1), size=(60.0, 60.0, 1.0))
    dV = (60.0 / n) ** 2
    mmap = np.zeros((n, n, 1), dtype=np.int64)
    mmap[n // 2:, :, :] = 1                        # right half poison
    mmap[5:8, 5:8, :] = 1                          # a poison patch in the fuel half
    region = np.zeros((n, n, 1), dtype=np.int64)
    region[n // 2:, :, :] = 1                      # two homogenization regions
    res = SP3EigenSolver(grid, mats, material_map=mmap,
                         bc=("vacuum", "vacuum", "reflective"), device="cpu"
                         ).solve(tol_k=1e-9, tol_source=1e-8)
    assert res.converged
    return res.flux_numpy, mats, mmap, region, dV


def test_flux_weighted_homogenization_preserves_reaction_rates():
    flux, mats, mmap, region, dV = _reference()
    hmats, rflux, rvol = flux_weighted_homogenize(flux, mats, mmap, region, cell_volume=dV)
    G = 2
    sa = np.array([m.sigma_a for m in mats])
    nf = np.array([m.nu_sigma_f for m in mats])
    ss = np.array([m.sigma_s for m in mats])       # (M, G, G)
    fmap = mmap.reshape(-1)
    phi = flux.reshape(G, -1)

    for i in range(2):
        cells = region.reshape(-1) == i
        for g in range(G):
            w = phi[g][cells] * dV
            # absorption and fission rates
            for xs, hom in ((sa, hmats[i].sigma_a), (nf, hmats[i].nu_sigma_f)):
                fine = (xs[fmap[cells], g] * w).sum()
                homr = hom[g] * rflux[i, g] * rvol[i]
                assert abs(homr - fine) <= 1e-9 * max(fine, 1.0)
            # out-scatter rate from group g
            for gp in range(G):
                if gp == g:
                    continue
                fine = (ss[fmap[cells], g, gp] * w).sum()
                homr = hmats[i].sigma_s[g, gp] * rflux[i, g] * rvol[i]
                assert abs(homr - fine) <= 1e-9 * max(fine, 1.0)


def test_homogenized_region_is_a_valid_material():
    flux, mats, mmap, region, dV = _reference()
    hmats, _, _ = flux_weighted_homogenize(flux, mats, mmap, region, cell_volume=dV)
    assert len(hmats) == 2
    for m in hmats:
        assert np.all(m.diffusion > 0)
        assert np.all(m.sigma_a >= 0)
        assert np.isclose(m.chi.sum(), 1.0)
    # region 0 is fuel-dominated (fissile), region 1 is the poison half
    assert hmats[0].is_fissile


def _reflective_assembly():
    # A heterogeneous assembly with a central absorber cluster (an intra-assembly
    # gradient), reflective -- the SPH *generation* geometry, where matching the
    # region reaction rates is equivalent to matching k_inf.
    poison = Material(name="poison", diffusion=[1.15, 0.55], sigma_a=[0.009, 0.12],
                      nu_sigma_f=[0, 0], sigma_s=[[0, 0.03], [0, 0]])
    mats = [PWR_TWO_GROUP, poison]
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
    region = np.digitize(rr, [6, 12]).reshape(n, n, 1).astype(np.int64)   # 3 rings
    return grid, mats, mmap, region, dV


def test_sph_correction_preserves_the_sp3_eigenvalue():
    # Full pipeline: SP3 reference -> flux-weighted homogenize -> SPH-correct so
    # the coarse diffusion reproduces the SP3 k_inf. The mu*Phi = phi_ref
    # condition preserves the region reaction rates, hence the eigenvalue.
    grid, mats, mmap, region, dV = _reflective_assembly()
    ref = SP3EigenSolver(grid, mats, material_map=mmap, bc="reflective",
                         device="cpu").solve(tol_k=1e-9, tol_source=1e-8)
    hmats, rflux, rvol = flux_weighted_homogenize(ref.flux_numpy, mats, mmap, region,
                                                  cell_volume=dV)

    def coarse_solve(materials):
        res = DiffusionEigenSolver(grid, materials, material_map=region,
                                   bc="reflective", device="cpu"
                                   ).solve(tol_k=1e-10, tol_source=1e-9)
        return region_average(res.flux_numpy, region), res.k_eff

    k_homog = coarse_solve(hmats)[1]
    out = sph_correct(hmats, region, rflux, coarse_solve, region_volume=rvol,
                      tol=1e-8)

    assert out.converged
    err_homog = abs(k_homog - ref.k_eff) * 1e5
    err_sph = abs(out.k_eff - ref.k_eff) * 1e5
    assert err_homog > 20.0                      # homogenization alone has real error
    assert err_sph < 0.5                          # SPH restores the reference k
    assert np.allclose(out.factors, 1.0, atol=0.15)   # well-homogenized: factors near 1


def _leaky_colorset(cuts=(12, 24)):
    # fuel | poisoned-fuel stripe | fuel, vacuum on the x ends (net leakage),
    # reflective y. Three column homogenization regions cut at the given fine
    # cells (default 30 and 60 cm: the middle region is exactly the stripe
    # column; (12, 18) cuts at 30 and 45 cm, so the region at the x = 90 cm
    # vacuum face mixes stripe and fuel).
    n = 36
    grid = Grid(shape=(n, 12, 1), size=(90.0, 30.0, 1.0))
    dV = (90.0 / n) * (30.0 / 12)
    mmap = np.zeros((n, 12, 1), dtype=np.int64)
    mmap[n // 3:2 * n // 3, 4:8, :] = 1
    region = np.zeros((n, 12, 1), dtype=np.int64)
    region[cuts[0]:cuts[1], :, :] = 1
    region[cuts[1]:, :, :] = 2
    return grid, [PWR_TWO_GROUP, POISON], mmap, region, dV, VACUUM_X


def _generation(grid, mats, mmap, region, dV, bc, coarse=None):
    """SP3 reference on the fine mesh, homogenized per region, and the coarse
    diffusion solve: on the fine mesh itself, or on ``coarse = (grid, region,
    cell_volume)`` with the coarse mesh's own region map and volumes."""
    ref = SP3EigenSolver(grid, mats, material_map=mmap, bc=bc,
                         device="cpu").solve(**TIGHT)
    assert ref.converged
    hmats, rflux, rvol = flux_weighted_homogenize(ref.flux_numpy, mats, mmap,
                                                  region, cell_volume=dV)
    cgrid, cregion, cdV = coarse if coarse is not None else (grid, region, dV)

    def coarse_solve(materials):
        res = DiffusionEigenSolver(cgrid, materials, material_map=cregion, bc=bc,
                                   device="cpu").solve(**TIGHT)
        return region_average(res.flux_numpy, cregion, cell_volume=cdV), res.k_eff
    return ref.k_eff, hmats, rflux, rvol, coarse_solve


def _k_after_scaling(hmats, mu, solve, group, c, d_scaling=None):
    m = mu.copy()
    m[:, group] *= c
    return solve([_scale_material(hmats[i], m[i], d_scaling=d_scaling)
                   for i in range(len(hmats))])[1]


def test_per_group_normalization_is_a_gauge_with_mu_d():
    # Reflective generation geometry: the factors are fixed only up to one
    # constant per group, and under D~ = mu D rescaling a group's factors is an
    # exact symmetry of the corrected operator, so k cannot depend on it. Under
    # the pre-audit D~ = D / mu it does (measured +830 to +980 pcm on this
    # colorset), which is why that convention made k depend on normalization.
    grid, mats, mmap, region, dV, _ = _leaky_colorset(cuts=(12, 18))
    k_ref, hmats, rflux, rvol, solve = _generation(grid, mats, mmap, region, dV,
                                                   "reflective")
    out = sph_correct(hmats, region, rflux, solve, region_volume=rvol, tol=1e-9)
    assert out.converged and out.residual <= 1e-9
    assert out.normalization == "flux-volume"
    assert abs(out.k_eff - k_ref) * 1e5 < 1e-3
    for g in range(2):
        assert abs(_k_after_scaling(hmats, out.factors, solve, g, 1.3)
                   - out.k_eff) * 1e5 < 1e-3
    inverse = abs(_k_after_scaling(hmats, out.factors, solve, 0, 1.3, "inverse")
                  - solve([_scale_material(hmats[i], out.factors[i], "inverse")
                           for i in range(3)])[1]) * 1e5
    assert inverse > 100.0


def test_sph_reproduces_a_reflective_colorset_on_a_coarse_node_mesh():
    # Workflow (b): reference on the fine mesh, factors for a coarse node mesh
    # (7.5 x 15 cm nodes) whose region map and cell volumes are its own. The
    # corrected coarse model reproduces k and the region reaction rates.
    grid, mats, mmap, region, dV, _ = _leaky_colorset(cuts=(12, 18))
    cgrid = Grid(shape=(12, 2, 1), size=(90.0, 30.0, 1.0))
    cx = np.digitize((np.arange(12) + 0.5) * 7.5, [30.0, 45.0])
    cregion = np.broadcast_to(cx[:, None, None], (12, 2, 1)).astype(np.int64).copy()
    k_ref, hmats, rflux, rvol, solve = _generation(
        grid, mats, mmap, region, dV, "reflective", coarse=(cgrid, cregion, 7.5 * 15.0))
    k_homog = solve(hmats)[1]
    out = sph_correct(hmats, None, rflux, solve, region_volume=rvol, tol=1e-9)
    assert out.converged
    assert abs(k_homog - k_ref) * 1e5 > 50.0              # coarse mesh + homogenization
    assert abs(out.k_eff - k_ref) * 1e5 < 1e-3
    assert abs(_k_after_scaling(hmats, out.factors, solve, 1, 0.7) - out.k_eff) < 1e-8
    # reaction rates normalized to total production (Sec. 6 item 1)
    Phi = np.asarray(solve(out.corrected_materials)[0])

    def rates(materials, flux):
        sa = np.array([m.sigma_a for m in materials]) * flux * rvol[:, None]
        nf = np.array([m.nu_sigma_f for m in materials]) * flux * rvol[:, None]
        return sa / nf.sum(), nf / nf.sum()
    for got, want in zip(rates(out.corrected_materials, Phi), rates(hmats, rflux)):
        np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-12)


def test_only_the_fixed_k_formulation_reproduces_k_on_a_vacuum_colorset():
    # With a vacuum boundary a normalization is no longer a gauge, so the
    # normalized SPH leaves a k error (Laboure et al. 2019, Sec. 3.1). Fixing
    # k = k_ref and dropping the normalization (their Sec. 3.2; Griffin)
    # reproduces k and the absolute region fluxes.
    grid, mats, mmap, region, dV, bc = _leaky_colorset()
    k_ref, hmats, rflux, rvol, solve = _generation(grid, mats, mmap, region, dV, bc)
    k_homog = solve(hmats)[1]
    assert abs(k_homog - k_ref) * 1e5 > 100.0

    normalized = sph_correct(hmats, region, rflux, solve, region_volume=rvol, tol=1e-9)
    assert normalized.converged
    assert abs(normalized.k_eff - k_ref) * 1e5 > 50.0    # the leakage inconsistency

    out = sph_correct(hmats, region, rflux, solve, normalization="none",
                      k_ref=k_ref, tol=1e-9)
    assert out.converged and out.normalization == "none"
    assert abs(out.k_eff - k_ref) * 1e5 < 1e-3
    Phi = np.asarray(solve(out.corrected_materials)[0])
    np.testing.assert_allclose(out.factors * out.flux_scale * Phi, rflux, rtol=1e-8)
    # the three methods find the same factors
    jfnk = sph_correct(hmats, region, rflux, solve, normalization="none",
                       k_ref=k_ref, tol=1e-8, method="jfnk")
    assert jfnk.converged
    np.testing.assert_allclose(jfnk.factors, out.factors, rtol=1e-6)


def test_fixed_k_reports_its_floor_when_no_solution_exists():
    # Region 2 sits at the vacuum face and mixes stripe and fuel. A
    # region-uniform mu under D~ = mu D cannot change the flux shape inside
    # it; at the vacuum face it only moves the effective Marshak coefficient
    # between alpha/mu -> 0 and infinity, and the reference leaks more than
    # that range allows. The solver must say so instead of claiming success.
    grid, mats, mmap, region, dV, bc = _leaky_colorset(cuts=(12, 18))
    k_ref, hmats, rflux, _, solve = _generation(grid, mats, mmap, region, dV, bc)
    out = sph_correct(hmats, region, rflux, solve, normalization="none",
                      k_ref=k_ref, tol=1e-9)
    assert not out.converged
    assert out.residual > 1e-3
    assert abs(solve(out.corrected_materials)[1] - out.k_eff) < 1e-10


def _reflected_slab():
    # water | fuel | fuel with a poison stripe | fuel | water, vacuum on the x
    # ends: the assembly-level situation of Griffin's workflow, where the
    # reflector is meshed but not corrected.
    nx, ny, h = 48, 12, 2.5
    grid = Grid(shape=(nx, ny, 1), size=(nx * h, ny * h, 1.0))
    mmap = np.zeros((nx, ny, 1), dtype=np.int64)
    mmap[:6] = 2
    mmap[-6:] = 2
    mmap[18:30, 4:8] = 1
    cuts = [15.0, 45.0, 75.0, 105.0]
    region = np.broadcast_to(np.digitize((np.arange(nx) + 0.5) * h, cuts)[:, None, None],
                             mmap.shape).astype(np.int64).copy()
    # coarse node mesh, 5 x 5 cm, with its own region map and volumes
    cgrid = Grid(shape=(24, 6, 1), size=(nx * h, ny * h, 1.0))
    cregion = np.broadcast_to(np.digitize((np.arange(24) + 0.5) * 5.0, cuts)[:, None, None],
                              (24, 6, 1)).astype(np.int64).copy()
    return (grid, [PWR_TWO_GROUP, POISON, WATER], mmap, region, h * h,
            (cgrid, cregion, 25.0))


def test_fuel_only_fixed_k_sph_on_a_coarse_node_mesh():
    grid, mats, mmap, region, dV, coarse = _reflected_slab()
    k_ref, hmats, rflux, rvol, solve = _generation(grid, mats, mmap, region, dV,
                                                   VACUUM_X, coarse=coarse)
    fuel = np.array([False, True, True, True, False])
    out = sph_correct(hmats, None, rflux, solve, normalization="none", k_ref=k_ref,
                      sph_regions=fuel, tol=1e-9)
    assert out.converged
    assert abs(out.k_eff - k_ref) * 1e5 < 1e-3
    np.testing.assert_array_equal(out.factors[~fuel], 1.0)   # reflector untouched
    Phi = np.asarray(solve(out.corrected_materials)[0])
    np.testing.assert_allclose((out.factors * out.flux_scale * Phi)[fuel], rflux[fuel],
                               rtol=1e-8)
    # the normalized formulation restricted to the fuel does not reproduce k
    fv = sph_correct(hmats, None, rflux, solve, region_volume=rvol,
                     sph_regions=fuel, tol=1e-9)
    assert fv.converged and abs(fv.k_eff - k_ref) * 1e5 > 20.0


def test_negative_reference_flux_in_a_corrected_region_warns():
    grid, mats, mmap, region, dV = _reflective_assembly()
    hmats, rflux, rvol = flux_weighted_homogenize(np.ones((2,) + grid.shape), mats,
                                                  mmap, region, cell_volume=dV)
    rflux = rflux.copy()
    rflux[2, 0] = -1e-3

    def solve(materials):
        res = DiffusionEigenSolver(grid, materials, material_map=region,
                                   bc="reflective", device="cpu").solve(tol_k=1e-8)
        return region_average(res.flux_numpy, region), res.k_eff
    with pytest.warns(RuntimeWarning, match=r"regions \[2\]"):
        out = sph_correct(hmats, region, rflux, solve, region_volume=rvol, max_iter=2)
    assert out.factors[2, 0] == 1.0
    # excluding the region explicitly is silent
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sph_correct(hmats, region, rflux, solve, region_volume=rvol, max_iter=2,
                    sph_regions=[True, True, False])


def test_sph_correct_rejects_incomplete_formulations():
    grid, mats, mmap, region, dV = _reflective_assembly()
    hmats, rflux, rvol = flux_weighted_homogenize(np.ones((2,) + grid.shape), mats,
                                                  mmap, region, cell_volume=dV)
    def never(materials):
        raise AssertionError("no solve expected")
    with pytest.raises(ValueError, match="region_volume"):
        sph_correct(hmats, region, rflux, never)
    with pytest.raises(ValueError, match="k_ref"):
        sph_correct(hmats, region, rflux, never, normalization="none")
    with pytest.raises(ValueError, match="normalization"):
        sph_correct(hmats, region, rflux, never, normalization="sum")
    with pytest.raises(ValueError, match="sph_regions"):
        sph_correct(hmats, region, rflux, never, region_volume=rvol,
                    sph_regions=[True, False])


def test_sph_kinetics_divides_speeds_by_the_factors():
    kin = Kinetics(velocities=[2.0e7, 2.2e5], beta=[0.002, 0.004], decay=[0.08, 1.0])
    mu = np.array([[1.1, 0.8], [0.9, 1.25]])
    out = sph_kinetics(kin, mu)
    np.testing.assert_allclose(out.velocities, kin.velocities / mu)
    np.testing.assert_array_equal(out.beta, kin.beta)
    np.testing.assert_array_equal(out.decay, kin.decay)
    with pytest.raises(ValueError, match="groups"):
        sph_kinetics(kin, np.ones((2, 3)))


def test_sph_kinetics_preserves_the_generation_time():
    # Under phi = mu Phi the time-derivative term scales like every other term
    # multiplying phi: 1/v -> mu/v. Reference: fine heterogeneous diffusion;
    # SPH model: coarse node mesh. Measured +20.7 % with v, +0.2 % with v/mu.
    v = np.array([1.0e7, 3.0e5])
    grid, mats, mmap, region, dV, _ = _leaky_colorset(cuts=(12, 18))
    kw = dict(bc="reflective", device="cpu")

    def generation_time(g, materials, mat_map, cell_volume, inv_v):
        fwd = DiffusionEigenSolver(g, materials, material_map=mat_map, **kw).solve(**TIGHT)
        adj = DiffusionEigenSolver(g, materials, material_map=mat_map, **kw).solve(
            adjoint=True, **TIGHT)
        cells = np.asarray(mat_map).reshape(-1)
        chi = np.array([m.chi for m in materials])[cells].T
        nf = np.array([m.nu_sigma_f for m in materials])[cells].T
        F, A = fwd.flux_numpy.reshape(2, -1), adj.flux_numpy.reshape(2, -1)
        return ((A * inv_v(cells) * F).sum()
                / ((A * chi).sum(0) * (nf * F).sum(0)).sum()), fwd

    lam_ref, fwd = generation_time(grid, mats, mmap, dV, lambda c: (1 / v)[:, None])
    hmats, rflux, rvol = flux_weighted_homogenize(fwd.flux_numpy, mats, mmap, region,
                                                  cell_volume=dV)
    cgrid = Grid(shape=(12, 2, 1), size=(90.0, 30.0, 1.0))
    cx = np.digitize((np.arange(12) + 0.5) * 7.5, [30.0, 45.0])
    cregion = np.broadcast_to(cx[:, None, None], (12, 2, 1)).astype(np.int64).copy()

    def solve(materials):
        res = DiffusionEigenSolver(cgrid, materials, material_map=cregion,
                                   **kw).solve(**TIGHT)
        return region_average(res.flux_numpy, cregion, cell_volume=7.5 * 15.0), res.k_eff
    out = sph_correct(hmats, None, rflux, solve, region_volume=rvol, tol=1e-9)
    assert out.converged
    kin = sph_kinetics(Kinetics(velocities=v, beta=[0.0065], decay=[0.08]), out.factors)
    lam_v, _ = generation_time(cgrid, out.corrected_materials, cregion, 1.0,
                               lambda c: (1 / v)[:, None])
    lam_mu, _ = generation_time(cgrid, out.corrected_materials, cregion, 1.0,
                                lambda c: (1 / kin.velocities[c]).T)
    assert abs(lam_v / lam_ref - 1) > 0.10
    assert abs(lam_mu / lam_ref - 1) < 0.005


def _rz_three_region_problem():
    """r-z cylinder with fuel / poisoned ring / fuel radial regions."""
    from garnet import CylindricalModel

    poison = Material(name="poison", diffusion=[1.15, 0.55], sigma_a=[0.009, 0.12],
                      nu_sigma_f=[0, 0], sigma_s=[[0, 0.03], [0, 0]])
    model = (CylindricalModel(radius=90.0, height=60.0, cells=(30, 6))
             .fill(PWR_TWO_GROUP).add_box(poison, r=(30.0, 60.0)))
    model.set_boundary(r="vacuum", z="reflective")
    region = model.material_map.reshape(model.grid.shape)  # one region per material
    return model, region


def test_region_average_uses_the_supplied_cell_volumes_on_rz():
    # The reference region flux is weighted by the annular volumes; a coarse
    # average with unit volumes overweights the axis cells, which SPH would
    # read as a factor (r-z single region: ratio 1.38).
    model, region = _rz_three_region_problem()
    res = model.run(device="cpu", tol_k=1e-10, tol_source=1e-9)
    volumes = model.grid.cell_volumes
    _, rflux, _ = flux_weighted_homogenize(res.flux_numpy, model.materials,
                                           region, region, cell_volume=volumes)
    np.testing.assert_allclose(
        region_average(res.flux_numpy, region, cell_volume=volumes), rflux,
        rtol=1e-12)
    unit = region_average(res.flux_numpy, region)
    assert np.abs(unit / rflux - 1.0).max() > 0.05     # the bias being fixed


def test_rz_sph_with_consistent_volumes_leaves_exact_regions_unchanged():
    # One material per region and the same discretization: homogenization is
    # exact, so consistent volume weighting must give mu = 1 and the reference k.
    from garnet import DiffusionEigenSolver as Solver

    model, region = _rz_three_region_problem()
    res = model.run(device="cpu", tol_k=1e-11, tol_source=1e-10)
    volumes = model.grid.cell_volumes
    hmats, rflux, rvol = flux_weighted_homogenize(res.flux_numpy, model.materials,
                                                  region, region, cell_volume=volumes)

    def coarse_solve(materials):
        r = Solver(model.grid, materials, material_map=region, bc=model._full_bc(),
                   device="cpu").solve(tol_k=1e-11, tol_source=1e-10)
        return region_average(r.flux_numpy, region, cell_volume=volumes), r.k_eff

    out = sph_correct(hmats, region, rflux, coarse_solve, region_volume=rvol,
                      tol=1e-8)
    assert out.converged
    np.testing.assert_allclose(out.factors, 1.0, atol=1e-6)
    assert abs(out.k_eff - res.k_eff) < 1e-8


def test_unused_region_index_raises_a_clear_error():
    grid, mats, mmap, region, dV, bc = _leaky_colorset()
    region = np.where(region == 2, 3, region)           # index 2 never used
    flux = np.ones((2,) + grid.shape)
    with pytest.raises(ValueError, match=r"empty regions: \[2\]"):
        flux_weighted_homogenize(flux, mats, mmap, region, cell_volume=dV)


def test_unconverged_anderson_returns_materials_matching_its_k():
    # Stopped at max_iter, the result must pair k_eff with the materials it
    # returns, not with an unsolved pending Anderson update, and must say it
    # did not converge.
    grid, mats, mmap, region, dV, bc = _leaky_colorset()
    ref = SP3EigenSolver(grid, mats, material_map=mmap, bc=bc,
                         device="cpu").solve(tol_k=1e-9, tol_source=1e-8)
    hmats, rflux, rvol = flux_weighted_homogenize(ref.flux_numpy, mats, mmap, region,
                                                  cell_volume=dV)

    def coarse_solve(materials):
        res = DiffusionEigenSolver(grid, materials, material_map=region, bc=bc,
                                   device="cpu").solve(tol_k=1e-11, tol_source=1e-10)
        return region_average(res.flux_numpy, region), res.k_eff

    out = sph_correct(hmats, region, rflux, coarse_solve, region_volume=rvol,
                      tol=1e-12, max_iter=3)
    assert not out.converged and out.residual > 1e-12
    assert abs(coarse_solve(out.corrected_materials)[1] - out.k_eff) < 1e-9


def test_apply_sph_factors_matches_the_corrected_materials():
    grid, mats, mmap, region, dV = _reflective_assembly()
    hmats, rflux, rvol = flux_weighted_homogenize(np.ones((2,) + grid.shape), mats,
                                                  mmap, region, cell_volume=dV)
    from garnet import apply_sph_factors
    mu = np.array([[1.1, 0.9], [0.8, 1.2], [1.0, 0.7]])
    for a, b in zip(apply_sph_factors(hmats, mu),
                    [_scale_material(m, mu[i]) for i, m in enumerate(hmats)]):
        np.testing.assert_array_equal(a.diffusion, b.diffusion)
        np.testing.assert_array_equal(a.sigma_s, b.sigma_s)
    with pytest.raises(ValueError, match="factors"):
        apply_sph_factors(hmats, mu[:2])
