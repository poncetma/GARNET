"""SPH on the 2D HP-MR with the transport reference swapped across families.

The SPH pipeline (:mod:`garnet.sph`) is reference-agnostic: it consumes the
*physical scalar flux* phi0 of any garnet eigensolver Result and the region
reaction rates that flux implies. For the block transport solvers phi0 is the
reconstructed 0th angular moment (SP3: phi0 = Phi1 - 2 phi2; SDPN: the even-
moment vector dotted with the phi0 closure weights), so swapping the reference
between SP3, SDP1 and SDP2 feeds SPH three genuinely different angular
treatments of the near-black B4C drum arc -- not a re-extraction of the same
field.

This module checks that each family drives the pipeline to convergence and folds
its own transport self-shielding into coarse TriDiffusion (fixed-k SPH, the
vacuum-facing Be reflector uncorrected, as in tests/validation/test_hpmr_sph.py),
and that the three families yield SPH factors that agree with one another far
more tightly than they depart from 1 (they are all transport, differing only
in angular order). The companion example
``examples/hpmr_sph_reference_families.py`` prints the full comparison table.

Marked slow: three full-core transport solves plus three Newton-iterated
sequences of diffusion solves. Run with ``pytest -m slow``.
"""
import numpy as np
import pytest

from garnet import (TriDiffusionEigenSolver, TriSDP1EigenSolver,
                   TriSDP2EigenSolver, TriSP3EigenSolver,
                   flux_weighted_homogenize, region_average, sph_correct)
from garnet.benchmarks.hpmr import (CENTRAL, DRUM_ABSORBER, DRUM_BE, FUEL,
                                    build_hpmr2d)

TIGHT = dict(tol_k=1e-9, tol_source=1e-8)
# corrected regions: everything but the void and the vacuum-facing Be reflector
INTERIOR = (FUEL, CENTRAL, DRUM_BE, DRUM_ABSORBER)

# The three matched-DoF transport references SPH can fold into diffusion.
REFERENCE_FAMILIES = {
    "SP3": TriSP3EigenSolver,
    "SDP1": TriSDP1EigenSolver,
    "SDP2": TriSDP2EigenSolver,
}


def _run_sph(problem, reference_cls):
    """Full SPH pipeline on the HP-MR from one transport reference family.

    Returns (reference_k, corrected_k, factors, converged): the reference
    eigenvalue, the SPH-corrected coarse-diffusion eigenvalue, the per-material
    SPH factors, and whether the factor solve converged.
    """
    p = problem
    dV = p.grid.cell_volume
    common = dict(active=p.active, mask_bc=p.mask_bc)

    ref = reference_cls(p.grid, p.materials, p.material_map, **common).solve(**TIGHT)
    assert ref.converged, f"{reference_cls.__name__} reference did not converge"

    region = p.material_map                       # one region per material type
    hmats, rflux, _ = flux_weighted_homogenize(
        ref.flux_numpy, p.materials, p.material_map, region, cell_volume=dV)

    def coarse_solve(materials):
        r = TriDiffusionEigenSolver(p.grid, materials, region, **common).solve(**TIGHT)
        return region_average(r.flux_numpy, region), r.k_eff

    out = sph_correct(hmats, region, rflux, coarse_solve, normalization="none",
                      k_ref=ref.k_eff, tol=1e-7,
                      sph_regions=np.isin(np.arange(len(hmats)), INTERIOR))
    return ref.k_eff, out.k_eff, out.factors, out.converged


@pytest.fixture(scope="module")
def hpmr_inserted():
    # Drums inserted (angle=0): the arc faces the core, maximal self-shielding
    # -- the state where the transport-vs-diffusion angular gap is largest.
    return build_hpmr2d(refine=4, drum_angle_deg=0.0, absorber="raster")


@pytest.mark.slow
@pytest.mark.parametrize("family", list(REFERENCE_FAMILIES))
def test_sph_folds_each_transport_family_into_hpmr_diffusion(hpmr_inserted, family):
    p = hpmr_inserted
    common = dict(active=p.active, mask_bc=p.mask_bc)
    dif = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                  **common).solve(**TIGHT)
    assert dif.converged

    ref_k, sph_k, mu, converged = _run_sph(p, REFERENCE_FAMILIES[family])

    gap_dif = abs(dif.k_eff - ref_k) * 1e5        # uncorrected diffusion vs transport
    gap_sph = abs(sph_k - ref_k) * 1e5            # SPH-corrected diffusion vs transport
    assert converged
    assert gap_dif > 30.0                         # diffusion misses the arc self-shielding
    assert gap_sph < 0.1                          # SPH reproduces this family's eigenvalue
    # the correction is carried by the absorber's thermal factor
    assert 1.0 - mu[DRUM_ABSORBER, -1] > 0.15
    assert np.abs(mu[[FUEL, CENTRAL, DRUM_BE]] - 1.0).max() < 0.1


@pytest.mark.slow
def test_sph_reference_families_agree_on_the_factors(hpmr_inserted):
    # SP3, SDP1 and SDP2 are all transport treatments of the drum arc. With the
    # eigenvalue fixed at each family's own reference, the comparison that is
    # about SPH (not about the references) is the factors: the absorber's
    # correction must agree across families far more tightly than it departs
    # from 1 (measured: thermal 0.752-0.760, i.e. a 0.008 spread on a 0.24
    # correction).
    p = hpmr_inserted
    factors = {}
    for name, cls in REFERENCE_FAMILIES.items():
        ref_k, sph_k, mu, converged = _run_sph(p, cls)
        assert converged
        factors[name] = mu[DRUM_ABSORBER]

    mu = np.array(list(factors.values()))         # (family, group)
    spread = (mu.max(axis=0) - mu.min(axis=0)).max()
    correction = np.abs(1.0 - mu).max()
    assert spread < 0.02
    assert spread < 0.2 * correction
