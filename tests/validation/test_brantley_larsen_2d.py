"""SPN / SDPN validation against Carreno et al. (2024) Table 6 / Fig. 3.

The 2D one-group Brantley-Larsen problem: three fuel bars (1 cm wide, 9 cm tall)
at x in [1,2],[4,5],[7,8], y in [0,9], in a 10x10 cm moderator square;
reflective at x=0 and y=0, vacuum at x=10 and y=10. Every material interface is
grid-aligned, so garnet's 2nd-order FV converges cheaply and the eigenvalue is a
*physical* check of the angular coefficients -- unlike the k_infinity / dense
tests, which only pin internal consistency. This problem caught a sign typo in
the paper's SDP2 c^(1) closure coefficient (Appendix A.3).

With the coupled Marshak vacuum the paper uses -- the default of every SPN/SDPN
solver here except the dedicated SDP1, which has no Marshak term -- garnet sits
+87 (SP1) to +127 pcm above the paper on this 80x80 mesh: that is mesh error,
since the mesh limit equals the paper within 0.6 pcm for every method except
SDP3 (test_marshak_converges_to_paper). It reproduces the full ordering,
including the headline SDPN-beats-SPN-at-equal-DoF result. The per-moment Robin
vacuum, kept for comparison (marshak_vacuum=False), is 53-54 pcm lower for
every method and 73 pcm lower for SDP1. The V&V report is
benchmark-results/carreno2024-sdpn-vv/.
"""

import numpy as np
import pytest

from garnet import (Grid, Material, DiffusionEigenSolver, SP3EigenSolver,
                   SP5EigenSolver, SP7EigenSolver, SDP1EigenSolver,
                   SDP2EigenSolver, SDP3EigenSolver)
from garnet.operator import _SDPN_C
from garnet.solver import SDPNEigenSolver

PAPER = {"SP1": 0.77680, "SP3": 0.79904, "SP5": 0.80280, "SP7": 0.80354,
         "SDP1": 0.80161, "SDP2": 0.80373, "SDP3": 0.80402}
FUEL = Material(name="fuel", diffusion=[1.0 / 4.5], sigma_a=[0.15],
                nu_sigma_f=[0.24], total=[1.5], chi=[1.0])
MOD = Material(name="mod", diffusion=[1.0 / 3.0], sigma_a=[0.07],
               nu_sigma_f=[0.0], total=[1.0], chi=[1.0])


def _problem(n=80):
    h = 10.0 / n
    xc = (np.arange(n) + 0.5) * h
    in_bar = ((xc > 1) & (xc < 2)) | ((xc > 4) & (xc < 5)) | ((xc > 7) & (xc < 8))
    mmap = np.where(in_bar[:, None] & (xc < 9.0)[None, :], 0, 1).astype(int)
    grid = Grid(shape=(n, n, 1), size=(10.0, 10.0, h))
    bc = (("reflective", "vacuum"), ("reflective", "vacuum"), "reflective")
    return grid, mmap[:, :, None], bc


class _SDP1Marshak(SDPNEigenSolver):
    """SDP1 through the general block: the dedicated SDP1 has no Marshak term."""
    _order = 1


# The paper's boundary: the coupled Marshak vacuum, every solver's default
# except the dedicated SDP1 (run through the general order-1 block instead).
MARSHAK_DEFAULT = {"SP1": DiffusionEigenSolver, "SP3": SP3EigenSolver,
                   "SP5": SP5EigenSolver, "SP7": SP7EigenSolver, "SDP1": _SDP1Marshak,
                   "SDP2": SDP2EigenSolver, "SDP3": SDP3EigenSolver}
# The per-moment Robin vacuum, kept for comparison.
ROBIN = {"SP3": SP3EigenSolver, "SP5": SP5EigenSolver, "SP7": SP7EigenSolver,
         "SDP1": SDP1EigenSolver, "SDP2": SDP2EigenSolver, "SDP3": SDP3EigenSolver}


def _k80(cls, **kw):
    grid, mmap, bc = _problem(80)
    r = cls(grid, [FUEL, MOD], material_map=mmap, bc=bc, device="cpu",
            **kw).solve(tol_k=1e-8, tol_source=1e-7)
    assert r.converged, (cls.__name__, r)
    return r.k_eff


@pytest.fixture(scope="module")
def keff():
    return {name: _k80(cls) for name, cls in MARSHAK_DEFAULT.items()}


@pytest.fixture(scope="module")
def keff_robin():
    return {name: _k80(cls, **({} if cls is SDP1EigenSolver else {"marshak_vacuum": False}))
            for name, cls in ROBIN.items()}


@pytest.mark.parametrize("name", list(PAPER))
def test_keff_matches_paper(keff, name):
    """Each method within 150 pcm of Table 6 on this mesh (measured +87 to
    +127 pcm, all mesh error). A gross coefficient error -- e.g. the SDP2 c^(1)
    typo -- shows here as a >1000 pcm miss."""
    assert keff[name] == pytest.approx(PAPER[name], abs=1.5e-3), \
        f"{name}: {keff[name]:.5f} vs paper {PAPER[name]:.5f}"


def test_offset_is_uniform(keff):
    """The garnet-minus-paper gap on this mesh is the mesh error, positive and
    similar for every method: measured +87 to +127 pcm, a spread of 41 pcm."""
    d = np.array([(keff[n] - PAPER[n]) * 1e5 for n in PAPER])
    assert np.all(d > 0) and np.all(d < 150), dict(zip(PAPER, d))
    assert d.max() - d.min() < 60          # tight spread => consistent physics


@pytest.mark.parametrize("name", list(ROBIN))
def test_robin_vacuum_is_low_by_a_known_offset(keff, keff_robin, name):
    """The per-moment Robin vacuum against the coupled Marshak one: measured
    -53 to -54 pcm, and -73 pcm for SDP1, on this leaky core."""
    d = (keff_robin[name] - keff[name]) * 1e5
    assert -80 < d < -45, f"{name}: Robin - Marshak = {d:+.1f} pcm"


def test_hierarchy_ordering(keff):
    """SPN and SDPN both converge monotonically upward toward the transport k."""
    assert keff["SP1"] < keff["SP3"] < keff["SP5"] < keff["SP7"]
    assert keff["SDP1"] < keff["SDP2"] < keff["SDP3"]


def test_sdpn_beats_spn_at_matched_dofs(keff):
    """The paper's headline: at equal degrees of freedom the simplified double-PN
    eigenvalue is closer to the transport reference than SPN."""
    assert keff["SDP1"] > keff["SP3"]      # 2-moment blocks
    assert keff["SDP2"] > keff["SP5"]      # 3-moment blocks
    assert keff["SDP3"] > keff["SP7"]      # 4-moment blocks


# --- coupled Marshak vacuum boundary (matches the paper's exact BC) ------------

def test_marshak_diagonal_g_reduces_to_per_moment():
    """The coupled-boundary machinery with a diagonal g = 1/2 I is identical to
    the per-moment Robin vacuum -- a correctness check on the K formula
    (K reduces to robin_face_term when the moments decouple)."""
    grid, mmap, bc = _problem(48)
    k_pm = SP5EigenSolver(grid, [FUEL, MOD], material_map=mmap, bc=bc, device="cpu",
                          marshak_vacuum=False).solve(tol_k=1e-8, tol_source=1e-7).k_eff

    class SP5diag(SP5EigenSolver):
        _g_coeffs = {2: [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]}

    k_diag = SP5diag(grid, [FUEL, MOD], material_map=mmap, bc=bc, device="cpu",
                     marshak_vacuum=True).solve(tol_k=1e-8, tol_source=1e-7).k_eff
    assert k_diag == pytest.approx(k_pm, abs=1e-6)


# SDP3 with the tables of the authors' FEMFFUSION code (src/matrix_operators/
# matrix_operators_sdpn.cc), which the paper's SDP3 results were computed with.
# They differ from garnet's re-derived rows 2-3 (see garnet/spn.py) and, in
# row 2 of c^(3), from the paper's printed Eq. (A.6).
_FFN_SDP3 = [
    [[1, -2 / 3, 8 / 15, -16 / 35], [-2 / 3, 4 / 9, -16 / 45, 32 / 105],
     [8 / 15, -16 / 45, 64 / 225, -128 / 525],
     [-176 / 125, 352 / 375, -1408 / 1875, 2816 / 4375]],
    [[0, 0, 0, 0], [0, 5 / 9, -4 / 9, 8 / 21], [0, -4 / 9, 16 / 45, -32 / 105],
     [0, 88 / 75, -352 / 375, 704 / 875]],
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 9 / 25, -54 / 175], [0, 0, -594 / 625, 3564 / 4375]],
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 143 / 175]],
]


class _SDP3FEMFFUSION(SDP3EigenSolver):
    _coeffs = {**_SDPN_C, 3: _FFN_SDP3}


MARSHAK = {"SP1": DiffusionEigenSolver, "SP3": SP3EigenSolver, "SP5": SP5EigenSolver,
           "SP7": SP7EigenSolver, "SDP1": _SDP1Marshak, "SDP2": SDP2EigenSolver,
           "SDP3": SDP3EigenSolver, "SDP3-FEMFFUSION": _SDP3FEMFFUSION}


def _marshak_limit(name):
    """Mesh limit with the coupled Marshak vacuum, from n = 80 and 160 at second
    order. The observed order is 1.93-2.00, and this two-mesh estimate is within
    2.2 pcm of the three-mesh limits of the V&V report."""
    cls = MARSHAK[name]
    kw = {} if name == "SP1" else {"marshak_vacuum": True}   # diffusion: Marshak is its vacuum
    ks = []
    for n in (80, 160):
        grid, mmap, bc = _problem(n)
        ks.append(cls(grid, [FUEL, MOD], material_map=mmap, bc=bc, device="cpu",
                      **kw).solve(tol_k=1e-10, tol_source=1e-9).k_eff)
    return ks[1] + (ks[1] - ks[0]) / 3.0


@pytest.mark.parametrize("name", ["SP1", "SP3", "SP5", "SP7", "SDP1"])
def test_marshak_converges_to_paper(name):
    """With the paper's boundary condition, the mesh limit is the paper's Table 6
    value: the same equations, discretized independently (FV here, quadratic FEM
    there), give the same eigenvalue. Measured within 1.5 pcm; 4 pcm allowed."""
    k = _marshak_limit(name)
    assert k == pytest.approx(PAPER[name], abs=4e-5), f"{name}: {k:.6f} vs {PAPER[name]:.5f}"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["SDP2", "SDP3-FEMFFUSION"])
def test_marshak_converges_to_paper_gmres_blocks(name):
    """As above for the GMRES-solved blocks (about 30 s each). SDP3 needs the
    authors' FEMFFUSION tables to reproduce Table 6 (measured +2.2 pcm)."""
    paper = PAPER[name.split("-")[0]]
    k = _marshak_limit(name)
    assert k == pytest.approx(paper, abs=4e-5), f"{name}: {k:.6f} vs {paper:.5f}"


@pytest.mark.slow
def test_rederived_sdp3_sits_below_the_paper():
    """garnet's re-derived SDP3 tables give a mesh limit 9.9 pcm below Table 6
    (-7.9 pcm from this two-mesh estimate). A value near zero would mean the
    tables were changed back to the paper's; see garnet/spn.py."""
    d = _marshak_limit("SDP3") - PAPER["SDP3"]
    assert -12e-5 < d < -4e-5, f"SDP3 - paper = {1e5 * d:+.1f} pcm"
