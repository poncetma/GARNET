"""SPH on the 2D HP-MR, TriSP3 as the transport reference (slow, full-core).

The HP-MR cross sections are already assembly-homogenized, so on this mesh SPH
is not correcting a spatial homogenization -- it is folding the SP3-vs-diffusion
*angular* difference into the diffusion constants. That difference lives almost
entirely in the near-black B4C drum arc, whose self-shielding plain diffusion
misses (it over-counts the absorption). SPH generates a per-material factor that
makes coarse TriDiffusion reproduce the TriSP3 eigenvalue -- where uncorrected
diffusion is off by ~60 pcm.

The core has a vacuum boundary, so the factors are solved with the eigenvalue
fixed at the reference and no normalization (Laboure et al. 2019, Sec. 3.2), as
Griffin does. The outer Be reflector, which carries the vacuum face, is not
corrected: correcting it lets a common rescaling of every factor of a group
act as a boundary coefficient, and the fit then moves all fast factors to
~0.26 to absorb a 10 pcm mismatch. Leaving it uncorrected keeps the factors
near 1 (SPH audit 2026-11-13, docs/numerical_implementation_notes.md).

Marked slow: a full-core SP3 solve plus a Newton-iterated sequence of
diffusion solves. Run with `pytest -m slow` (or unmarked in the nightly suite).
"""
import numpy as np
import pytest

from garnet import (TriDiffusionEigenSolver, TriSP3EigenSolver,
                   flux_weighted_homogenize, region_average, sph_correct)
from garnet.benchmarks.hpmr import (CENTRAL, DRUM_ABSORBER, DRUM_BE, FUEL,
                                    build_hpmr2d)

# corrected regions: everything but the void and the vacuum-facing Be reflector
INTERIOR = (FUEL, CENTRAL, DRUM_BE, DRUM_ABSORBER)

TIGHT = dict(tol_k=1e-9, tol_source=1e-8)


@pytest.mark.slow
def test_sph_folds_sp3_self_shielding_into_hpmr_diffusion():
    # Drums inserted (angle=0): the arc faces the core, maximal self-shielding.
    p = build_hpmr2d(refine=4, drum_angle_deg=0.0, absorber="raster")
    dV = p.grid.cell_volume
    common = dict(active=p.active, mask_bc=p.mask_bc)

    ref = TriSP3EigenSolver(p.grid, p.materials, p.material_map, **common).solve(**TIGHT)
    dif = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map, **common).solve(**TIGHT)
    assert ref.converged and dif.converged

    region = p.material_map                       # one region per material type
    hmats, rflux, _ = flux_weighted_homogenize(
        ref.flux_numpy, p.materials, p.material_map, region, cell_volume=dV)

    def coarse_solve(materials):
        r = TriDiffusionEigenSolver(p.grid, materials, region, **common).solve(**TIGHT)
        return region_average(r.flux_numpy, region), r.k_eff

    corrected = np.isin(np.arange(len(hmats)), INTERIOR)
    out = sph_correct(hmats, region, rflux, coarse_solve, normalization="none",
                      k_ref=ref.k_eff, sph_regions=corrected, tol=1e-7)

    gap_dif = abs(dif.k_eff - ref.k_eff) * 1e5
    gap_sph = abs(out.k_eff - ref.k_eff) * 1e5
    assert out.converged
    assert gap_dif > 30.0                         # diffusion misses the arc self-shielding
    assert gap_sph < 0.1                          # SPH reproduces the SP3 eigenvalue

    # the correction is carried by the absorber (thermal factor ~0.76: diffusion
    # over-absorbs in the near-black arc), while the other corrected materials
    # stay within 10 % of 1 and the uncorrected ones are untouched.
    mu = out.factors
    assert 1.0 - mu[DRUM_ABSORBER, -1] > 0.15
    assert np.abs(mu[[FUEL, CENTRAL, DRUM_BE]] - 1.0).max() < 0.1
    np.testing.assert_array_equal(mu[~corrected], 1.0)
