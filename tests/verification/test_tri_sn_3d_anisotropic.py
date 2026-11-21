"""P_L scattering and user ordinate sets on extruded triangular prisms.

In 3-D the flux carries every real spherical harmonic R_lk (l <= L, all k) of
the signed z cosine; the 2-D build keeps only the z-even ones. Checks, each
broken on purpose:

* LA-13511 slabs laid along z (2 x 2 periodic prisms in-plane, vacuum top and
  bottom, LU engine, Gauss-Legendre S64 polar set, step in z extrapolated from
  200 / 400 layers): the isotropic control 2 is at -9.1 pcm (S64 angular
  error), P1 slab 34 at -7.5 pcm and P2 slab 35 at -4.0 pcm. The slab current
  lives in the z-odd harmonics only. Broken: the basis built on the unsigned
  z cosine |xi| (so the z-odd moments are lost) does not converge within a
  budget the correct basis meets (S16, 50 layers); dropping the moments misses
  by thousands of pcm (P0: +5491 / +4394 pcm);
* the level-scheduled engine (the GPU path, per-ordinate source buffers) and
  the LU engine solve the same discrete P2 problem on a vacuum block of prisms,
  step and SCB: k equal to 1e-9. Broken: the levels engine without the angular
  source (rhs_ang zeroed) differs;
* user ordinate sets (``quadrature=``): octant sets are reflected into the
  eight octants and validated; the Gram check of the harmonics refuses or warns
  on a set that does not integrate P_L exactly; the GeN-Foam quadratureSet
  reader reproduces a level-symmetric S4 octant.
"""

import warnings

import numpy as np
import pytest

from garnet.benchmarks.sood import PROBLEMS, sood_material
from garnet.genfoam import read_quadrature_set
from garnet.materials import Material
from garnet.sn import harmonic_gram_error, octant_quadrature, quadrature_3d
from garnet.tri import TriGrid
from garnet.tri_sn import TriSNTransportSolver

TOL = dict(tol_k=1e-11, tol_source=1e-10)

# level-symmetric S4: (mu1, mu1, mu2) and permutations, equal weights
_MU1 = np.sqrt(0.1225148226554413)
_MU2 = np.sqrt(1.0 - 2.0 * 0.1225148226554413)
LS4 = ([_MU1, _MU1, _MU2], [_MU1, _MU2, _MU1], [_MU2, _MU1, _MU1], [1.0, 1.0, 1.0])


def _z_slab(number, nz, order=None, n_polar=64, max_inner=800, max_outer=500):
    p = PROBLEMS[number]
    L = p.order if order is None else order
    grid = TriGrid(shape=(2, 2, 2, nz), side=1.0, height=2.0 * p.rc)
    return TriSNTransportSolver(grid, [sood_material(p.material, L)], n_polar=n_polar,
                                n_azi=8 if L > 1 else 4, bc=("periodic", "vacuum"),
                                scheme="step", engine="lu", device="cpu",
                                scattering_order=L, max_inner=max_inner
                                ).solve(max_outer=max_outer, **TOL)


def _z_slab_k(number, nz, order=None):
    r = _z_slab(number, nz, order)
    assert r.converged
    return r.k_eff


def _extrapolated(number, order=None):
    a, b = _z_slab_k(number, 200, order), _z_slab_k(number, 400, order)
    return 2.0 * b - a                                   # step: first order in dz


@pytest.mark.parametrize("number", [2, 34, 35])
def test_slabs_along_z_are_critical(number):
    assert abs(_extrapolated(number) - 1.0) < 1.5e-4     # -9.1 / -7.5 / -4.0 pcm


def test_broken_unsigned_z_cosine_loses_the_slab_current(monkeypatch):
    setup = TriSNTransportSolver._setup_anisotropy

    def unsigned(self, *a, **k):
        xi = self.xi
        self.xi = np.abs(xi)
        try:
            return setup(self, *a, **k)
        finally:
            self.xi = xi

    budget = dict(n_polar=16, max_inner=50, max_outer=30)
    good = _z_slab(34, 50, **budget)                     # control: same budget
    assert good.converged and abs(good.k_eff - 1.0) < 1e-2
    monkeypatch.setattr(TriSNTransportSolver, "_setup_anisotropy", unsigned)
    bad = _z_slab(34, 50, **budget)                      # does not converge
    assert not bad.converged or abs(bad.k_eff - 1.0) > 1e-2


def test_broken_without_the_moments():
    assert _z_slab_k(35, 200, order=0) - 1.0 > 1e-2      # +4394 pcm


def _block(engine, scheme):
    """A bare 4 x 4 x 2 x 4 block of prisms, two-group P2 with downscatter."""
    G = 2
    s0 = np.array([[0.30, 0.05], [0.0, 0.60]])
    s1 = np.array([[0.12, 0.02], [0.0, 0.15]])
    s2 = np.array([[0.05, 0.01], [0.0, 0.04]])
    m = Material(diffusion=[1.2, 0.5], sigma_a=[0.02, 0.10], nu_sigma_f=[0.01, 0.20],
                 sigma_s=s0, chi=[1.0, 0.0], total=[0.37, 0.70],
                 sigma_s_moments=np.stack([s0, s1, s2]))
    grid = TriGrid(shape=(4, 4, 2, 4), side=1.5, height=4.0)
    return TriSNTransportSolver(grid, [m], n_polar=4, n_azi=8, bc="vacuum",
                                scheme=scheme, engine=engine, device="cpu",
                                scattering_order=2).solve(**TOL).k_eff


@pytest.mark.parametrize("scheme", ["step", "scb"])
def test_levels_engine_matches_lu_with_p2(scheme):
    assert _block("levels", scheme) == pytest.approx(_block("lu", scheme), abs=1e-9)


def test_broken_levels_without_the_angular_source(monkeypatch):
    sweep = TriSNTransportSolver._sweep_aniso_dev

    def no_angular(self, g, src, ang):
        return sweep(self, g, src, None)

    monkeypatch.setattr(TriSNTransportSolver, "_sweep_aniso_dev", no_angular)
    assert abs(_block("levels", "step") - _block("lu", "step")) > 1e-5


def test_octant_sets_are_closed_and_normalized():
    mu, eta, xi, w = octant_quadrature(*LS4)
    assert len(w) == 24 and w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(mu ** 2 + eta ** 2 + xi ** 2, 1.0, atol=1e-14)
    pts = {tuple(np.round(v, 12)) for v in np.stack([mu, eta, xi], 1)}
    assert all(tuple(np.round(-v, 12)) in pts for v in np.stack([mu, eta, xi], 1))
    assert harmonic_gram_error(mu, eta, xi, w, 2) < 1e-12     # LS S4 is P2-exact
    assert harmonic_gram_error(*quadrature_3d(8, 16), 4) < 1e-12
    with pytest.raises(ValueError):
        octant_quadrature([0.5], [0.5], [0.5], [1.0])        # not a unit vector


def test_inexact_set_is_refused_or_warned():
    one = octant_quadrature([3 ** -0.5], [3 ** -0.5], [3 ** -0.5], [1.0])
    assert harmonic_gram_error(*one, 2) > 0.1
    m = sood_material("PUb-aniso", 2)
    grid = TriGrid(shape=(2, 2, 2, 4), side=1.0, height=1.0)
    kw = dict(bc=("periodic", "vacuum"), engine="lu", device="cpu", scattering_order=2)
    with pytest.raises(ValueError, match="does not integrate"):
        TriSNTransportSolver(grid, [m], quadrature=one, **kw)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = TriSNTransportSolver(grid, [m], quadrature=one, quadrature_check="warn", **kw)
    assert s.M == 8 and any("does not integrate" in str(x.message) for x in w)


def test_user_quadrature_is_validated():
    grid = TriGrid(shape=(2, 2, 2, 4), side=1.0, height=1.0)
    m = sood_material("Ua", 0)
    mu, eta, xi, w = octant_quadrature(*LS4)
    kw = dict(bc=("periodic", "vacuum"), engine="lu", device="cpu")
    with pytest.raises(ValueError, match="closed"):
        TriSNTransportSolver(grid, [m], quadrature=(mu[:-1], eta[:-1], xi[:-1], w[:-1]), **kw)
    with pytest.raises(ValueError, match="prisms"):
        TriSNTransportSolver(TriGrid(shape=(2, 2, 2), side=1.0), [m],
                             quadrature=(mu, eta, xi, w), bc="periodic", engine="lu")
    s = TriSNTransportSolver(grid, [m], quadrature=(mu, eta, xi, 3.0 * w), **kw)
    assert s.w.sum() == pytest.approx(1.0) and s.M == 24


def test_genfoam_quadrature_file_is_read(tmp_path):
    a, b, c, w = LS4
    lst = lambda v: "nonuniform List<scalar> 3 (" + " ".join(f"{x:.16e}" for x in v) + ");"
    f = tmp_path / "quadratureSet"
    f.write_text("FoamFile { version 2.0; format ascii; class dictionary; }\n"
                 "discreteDirectionsOct 3;\n"
                 f"directionUxOct {lst(a)}\ndirectionUyOct {lst(b)}\n"
                 f"directionUzOct {lst(c)}\n"
                 f"directionWeightsOct {lst(np.full(3, np.pi / 6))}\n")
    got = read_quadrature_set(f)
    for x, y in zip(got, octant_quadrature(*LS4)):
        np.testing.assert_allclose(x, y, atol=1e-15)
