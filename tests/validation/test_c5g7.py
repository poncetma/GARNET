"""OECD/NEA C5G7 MOX benchmark (2D quarter core).

Most tests use the pin-cell homogenized model at one cell per pin. They check
the geometry construction (pin counts, symmetry, reaction-rate conservation of
the homogenization) and that the whole SPN/SDPN angular hierarchy behaves
consistently on C5G7. The hierarchy tests deliberately instantiate every solver
so they always exercise the *current* coefficient tables (_SPN_C/_SDPN_C in
garnet.operator) -- coefficient regressions (like the paper's SDP2/SDP3 table
errors) surface here as broken ordering or a blown eigenvalue.

The homogenized model's closeness to the transport reference is a cancellation,
not accuracy: homogenizing each pin cell raises k by 380-660 pcm against the
pin-resolved model, and one cell per pin lowers it by about 225 pcm. The
pin-resolved comparison with Carreno et al. (2024) and the NEA MCNP pin powers
is the slow test at the end; the V&V report is
benchmark-results/carreno2024-sdpn-vv/.
"""

import numpy as np
import pytest

from garnet import (DiffusionEigenSolver, SP3EigenSolver, SP5EigenSolver,
                   SP7EigenSolver, SDP1EigenSolver, SDP2EigenSolver,
                   SDP3EigenSolver)
from garnet.benchmarks import build_c5g7_2d
from garnet.benchmarks.c5g7 import FUEL_FRACTION, K_REFERENCE_2D, pin_powers
from tests._femffusion import femffusion_path, requires_femffusion

NEA_PIN_POWERS = "examples/2D_C5G7/c5g7_transport.ref"


def test_c5g7_geometry():
    prob = build_c5g7_2d(cells_per_pin=2)
    assert prob.grid.shape == (102, 102, 1)
    assert prob.material_map.shape == (102, 102, 1)
    assert len(prob.materials) == 7
    # 4 assemblies x (264 fuel + 24 GT + 1 FC) pin cells; the rest is water.
    counts = np.bincount(prob.pin_map.ravel(), minlength=7)
    assert counts[4] == 4 * 24            # guide tubes
    assert counts[5] == 4 * 1             # fission chambers
    assert counts[:4].sum() == 4 * 264    # fuel pins
    assert counts[6] == 51 * 51 - 4 * 289
    # quarter-core diagonal symmetry of the layout
    assert np.array_equal(prob.pin_map, prob.pin_map.T)
    # homogenization conserved reaction rates at the mixing level
    uo2_cell = prob.materials[0]
    from garnet.benchmarks._c5g7_data import C5G7_XS
    expected = FUEL_FRACTION * np.array(C5G7_XS["UO2"]["nu_fission"])
    assert np.allclose(uo2_cell.nu_sigma_f, expected)


@pytest.mark.parametrize("s", [1, 3, 4, 8, 13, 16])
def test_c5g7_exact_pin_area(s):
    """Exact cell fractions carry the cylinder's pi r^2 at any resolution, and
    the pin-resolved builder places them; the 16 x 16 sampled default does not
    (-0.66% fuel at s = 4)."""
    from garnet.benchmarks.c5g7 import pin_coverage
    assert pin_coverage(s, None).mean() == pytest.approx(FUEL_FRACTION, rel=1e-12)
    prob = build_c5g7_2d(cells_per_pin=s, pin_resolved=True, exact_area=True,
                         outer_bc="vacuum")
    fuel_cells = prob.pin_map.repeat(s, 0).repeat(s, 1).T < 4
    covered = np.where(prob.material_map[..., 0] < 4, 1.0, 0.0)
    covered = np.where(prob.mix_material[..., 0] >= 0, prob.mix_weight[..., 0], covered)
    assert covered[fuel_cells].sum() / s ** 2 == pytest.approx(1056 * FUEL_FRACTION, rel=1e-12)
    assert prob.bc[0] == ("reflective", "vacuum") and prob.bc[1] == ("reflective", "vacuum")


def test_c5g7_solves_to_reasonable_k():
    prob = build_c5g7_2d(cells_per_pin=1)
    res = DiffusionEigenSolver(prob.grid, prob.materials, prob.material_map,
                               bc=prob.bc, device="cpu").solve(
        tol_k=1e-6, tol_source=1e-5)
    assert res.converged
    # Homogenized diffusion at one cell per pin: measured -165 pcm, a
    # cancellation of homogenization (+380) and mesh (-220) errors.
    assert abs(res.k_eff - K_REFERENCE_2D) / K_REFERENCE_2D < 0.02, res.k_eff


@pytest.fixture(scope="module")
def family_keff():
    """k_eff of the full angular hierarchy at cells_per_pin=1 (~6 s total)."""
    prob = build_c5g7_2d(cells_per_pin=1)
    out = {}
    for name, cls in [("diffusion", DiffusionEigenSolver),
                      ("sp3", SP3EigenSolver), ("sp5", SP5EigenSolver),
                      ("sp7", SP7EigenSolver), ("sdp1", SDP1EigenSolver),
                      ("sdp2", SDP2EigenSolver), ("sdp3", SDP3EigenSolver)]:
        r = cls(prob.grid, prob.materials, prob.material_map, bc=prob.bc,
                device="cpu").solve(tol_k=1e-6, tol_source=1e-5)
        assert r.converged, (name, r)
        out[name] = r.k_eff
    return out


def test_c5g7_family_lands_near_transport(family_keff):
    """Every angular approximation within 500 pcm of the OpenMC/MCNP reference
    (measured -8 to -165 pcm; a blown coefficient table misses by thousands)."""
    for name, k in family_keff.items():
        assert abs(k - K_REFERENCE_2D) / K_REFERENCE_2D < 5e-3, (name, k)


def test_c5g7_transport_cluster(family_keff):
    """The C5G7 signature (paper Sec. 4.3): the six SPN/SDPN eigenvalues
    stagnate -- they cluster within 150 pcm of each other (measured 36 pcm)."""
    ks = [family_keff[n] for n in ("sp3", "sp5", "sp7", "sdp1", "sdp2", "sdp3")]
    assert max(ks) - min(ks) < 1.5e-3, ks


def test_c5g7_hierarchies_are_monotone(family_keff):
    """Both families rise monotonically with order on this mesh, and each SDPN
    sits above its matched-DoF SPN partner -- the orderings that broke under the
    (pre-fix) SDP2/SDP3 coefficient errors. Being below the transport reference
    here is the mesh error; the homogenized mesh limit lies above it."""
    k = family_keff
    assert k["sp3"] < k["sp5"] < k["sp7"] < K_REFERENCE_2D
    assert k["sdp1"] < k["sdp2"] < k["sdp3"] < K_REFERENCE_2D
    for sdp, sp in (("sdp1", "sp3"), ("sdp2", "sp5"), ("sdp3", "sp7")):
        assert abs(k[sdp] - K_REFERENCE_2D) < abs(k[sp] - K_REFERENCE_2D), \
            (sdp, k[sdp], sp, k[sp])


def _nea_pin_powers():
    lines = open(femffusion_path(NEA_PIN_POWERS)).read().splitlines()
    i = lines.index("Neutronic Power") + 1
    return np.array([[float(v) for v in line.split()] for line in lines[i:i + 51]])


@pytest.mark.slow
@requires_femffusion(NEA_PIN_POWERS)
def test_pin_resolved_diffusion_reproduces_the_paper():
    """Pin-resolved C5G7 as in Carreno et al. (2024): r = 0.54 cm cylinders with
    exact cell areas, vacuum outside, 8 cells per pin side (about a minute).
    Diffusion (the paper's SP1) gives k within 10 pcm of Table 7 (measured
    -6.8 pcm; -11 pcm at the mesh limit) and a pin-power AVG against the NEA MCNP
    pin powers within 0.06 points of the paper's 1.43% (measured 1.46%; 1.43% at
    the mesh limit)."""
    prob = build_c5g7_2d(cells_per_pin=8, pin_resolved=True, exact_area=True,
                         outer_bc="vacuum")
    r = DiffusionEigenSolver(prob.grid, prob.materials, prob.material_map, bc=prob.bc,
                             device="cpu", mix_material=prob.mix_material,
                             mix_weight=prob.mix_weight).solve(tol_k=1e-9, tol_source=1e-8)
    assert r.converged
    assert r.k_eff == pytest.approx(1.18336, abs=10e-5)
    ref = _nea_pin_powers()
    P = pin_powers(prob, r.flux_numpy)
    fuel = ref > 0
    assert np.array_equal(fuel, P > 0)
    avg = 100 * np.mean(np.abs(P[fuel] / ref[fuel] - 1))
    assert avg == pytest.approx(1.43, abs=0.06)
