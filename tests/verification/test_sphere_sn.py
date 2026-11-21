"""One-dimensional spherical S_N (garnet.sphere_sn) against the LA-13511
analytic sphere criticality problems (Sood et al. 1999).

Every problem is exactly critical (k = 1) at its tabulated radius r_c:

| Problem | Case | r_c (cm) |
|---|---|---|
| 14 | one group, isotropic | 7.428998 (plus tabulated scalar fluxes) |
| 24 | one group, isotropic | 22.017156 |
| 41 | one group, P1 | 18.30563081 |
| 43 | one group, P1 with a negative kernel | 18.30563081 |
| 55 | two groups, isotropic | 16.049836 |

Measured k - 1 with 400 shells, Gauss-Legendre S32 / S64 / S128, and the
angular extrapolation (the error falls as N^-2):

| Problem | S32 | S64 | S128 | Extrapolated |
|---|---|---|---|---|
| 14 | +17.0 | +4.3 | +1.1 | +0.02 pcm |
| 24 | +0.5 | +0.1 | 0.0 | 0.00 pcm |
| 41 | +1.3 | +0.4 | +0.1 | +0.04 pcm |
| 43 | -0.4 | -0.7 | -0.8 | -0.8 pcm |
| 55 | +2.2 | +0.7 | +0.4 | +0.24 pcm |

The mesh is converged to 0.3 pcm at 50 shells (second order). The scalar
fluxes of problem 14 match the tabulated ratios to 2e-4.

Broken on purpose:

* the angular redistribution term dropped (alpha = 0): problem 14 misses by
  thousands of pcm;
* the P1 moment of problem 43 dropped: misses by thousands of pcm;
* S8 instead of S64: misses by far more than the tolerance (k too high), so
  the tolerance is not loose.
"""

import numpy as np
import pytest

from garnet.benchmarks.sood import PROBLEMS, sood_material
from garnet.sphere_sn import SphereSNSolver


def _k(number, n_mu=64, shells=200, order=None, alpha_zero=False):
    p = PROBLEMS[number]
    m = sood_material(p.material, p.order if order is None else order)
    s = SphereSNSolver(np.linspace(0.0, p.rc, shells + 1)[1:], m, n_mu=n_mu)
    if alpha_zero:
        s.alpha[:] = 0.0
    r = s.solve()
    assert r.converged
    return r.k_eff, s, r


@pytest.mark.parametrize("number,tol", [(14, 6e-5), (24, 1e-5), (41, 1e-5), (43, 1.5e-5),
                                        (55, 1.5e-5)])
def test_sood_spheres_are_critical(number, tol):
    k, _, _ = _k(number)
    assert abs(k - 1.0) < tol                     # S64: 4.3 / 0.1 / 0.4 / 0.7 / 0.7 pcm


def test_angular_extrapolation_reaches_the_analytic_k():
    ks = [_k(14, n_mu=n, shells=100)[0] for n in (32, 64, 128)]
    ext = ks[2] + (ks[2] - ks[1]) / 3.0           # error ~ N^-2
    assert abs(ext - 1.0) < 3e-6                  # 0.1 pcm at 100 shells (mesh error)
    assert ks[0] > ks[1] > ks[2] > 1.0            # monotone from above


def test_sphere_fluxes_match_the_tabulated_ratios():
    p = PROBLEMS[14]
    _, s, r = _k(14, n_mu=128, shells=800)
    rc = np.concatenate(([0.0], s.radii))
    mid = 0.5 * (rc[:-1] + rc[1:])
    f = r.flux[0]
    # quadratic fits to the shell averages near the centre and the surface
    centre = np.polyval(np.polyfit(mid[:6], f[:6], 2), 0.0)
    surface = np.polyval(np.polyfit(mid[-6:], f[-6:], 2), p.rc)
    got = [np.interp(x * p.rc, mid, f) if x < 1.0 else surface for x in p.flux_positions]
    np.testing.assert_allclose(np.array(got) / centre, p.flux[0], rtol=0, atol=3e-4)


def test_broken_without_angular_redistribution():
    k, _, _ = _k(14, alpha_zero=True)
    assert abs(k - 1.0) > 1e-2


def test_broken_without_the_p1_moment():
    k, _, _ = _k(43, order=0)
    assert abs(k - 1.0) > 1e-2


def test_broken_coarse_quadrature_exceeds_the_tolerance():
    k, _, _ = _k(14, n_mu=8)
    assert k - 1.0 > 6e-5 * 5
