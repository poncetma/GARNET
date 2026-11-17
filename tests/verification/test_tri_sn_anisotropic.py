"""Anisotropic scattering in the triangular S_N solver, and transport correction.

The HP-MR library carries the TRUE total cross section and P0-P5 scattering
moments (the YH moderator's P1 moment reaches 56 % of its total), so a faithful
S_N reference must scatter anisotropically or use transport-corrected data.

Checks:
* zero higher moments reproduce the isotropic solve exactly, on both engines
  and both spatial schemes;
* in an infinite (periodic) medium the fundamental mode is flat and
  isotropic, so anisotropy cannot change k = k_inf;
* the two sweep engines agree;
* the anisotropic P1 solve and the transport-corrected isotropic solve share
  the asymptotic diffusion limit, so their difference shrinks under mesh
  refinement (first order: the anisotropic source is flat per cell);
* the transport correction keeps the neutron balance and commutes with
  blending;
* the net absorption closes the balance on the true total, so every solver
  keeps the Be (n,2n) gain and S_N's within-group scattering is the stored P0.
"""

import numpy as np
import pytest

from garnet import Material, net_absorption, transport_corrected
from garnet.tri import TriGrid
from garnet.tri_sn import TriSNTransportSolver

TIGHT = dict(tol_k=1e-11, tol_source=1e-10)


def _one_group(s1, s2=0.0, st=1.0, sa=0.08, nsf=0.1):
    return Material(name="m", diffusion=[1.0 / (3.0 * st)], sigma_a=[sa],
                    nu_sigma_f=[nsf], sigma_s=[[st - sa]], total=[st],
                    sigma_s_moments=np.array([[[st - sa]], [[s1]], [[s2]]]))


@pytest.mark.parametrize("scheme", ["step", "scb"])
@pytest.mark.parametrize("engine", ["lu", "levels"])
def test_zero_moments_reproduce_the_isotropic_solve(scheme, engine):
    grid = TriGrid(shape=(10, 10, 2), side=1.0)
    kw = dict(n_polar=2, n_azi=8, scheme=scheme, engine=engine)
    iso = TriSNTransportSolver(grid, [_one_group(0.0)], **kw).solve(**TIGHT)
    ani = TriSNTransportSolver(grid, [_one_group(0.0)], scattering_order=2,
                               **kw).solve(**TIGHT)
    assert ani.k_eff == pytest.approx(iso.k_eff, abs=1e-11)


def test_engines_agree_with_anisotropic_scattering():
    grid = TriGrid(shape=(10, 10, 2), side=1.0)
    k = [TriSNTransportSolver(grid, [_one_group(0.3, 0.05)], n_polar=2, n_azi=8,
                              scheme="scb", engine=e, scattering_order=2
                              ).solve(**TIGHT).k_eff for e in ("lu", "levels")]
    assert k[0] == pytest.approx(k[1], abs=1e-10)


def test_anisotropy_is_inert_in_an_infinite_medium():
    grid = TriGrid(shape=(6, 6, 2), side=1.0)
    r = TriSNTransportSolver(grid, [_one_group(0.4, 0.1, sa=0.02, nsf=0.024)],
                             n_polar=2, n_azi=8, scheme="scb", engine="lu",
                             bc="periodic", scattering_order=2).solve(**TIGHT)
    assert r.k_eff == pytest.approx(0.024 / 0.02, abs=1e-12)


def test_forward_scattering_leaks_more_and_transport_correction_follows():
    # P1 forward scattering lowers k in a bare core; the transport-corrected
    # isotropic solve captures most of it, and the remaining difference halves
    # when the cells are halved (same physical size).
    mat = _one_group(0.4, sa=0.02, nsf=0.024)
    tc = transport_corrected(mat, "diagonal")
    diffs = []
    for n in (16, 32):
        grid = TriGrid(shape=(n, n, 2), side=16.0 / n)
        kw = dict(n_polar=3, n_azi=12, scheme="scb", engine="levels")
        k_iso = TriSNTransportSolver(grid, [_one_group(0.0, sa=0.02, nsf=0.024)],
                                     **kw).solve(**TIGHT).k_eff
        k_p1 = TriSNTransportSolver(grid, [mat], scattering_order=1,
                                    **kw).solve(**TIGHT).k_eff
        k_tc = TriSNTransportSolver(grid, [tc], **kw).solve(**TIGHT).k_eff
        assert k_iso - k_p1 > 0.03                     # anisotropy matters
        diffs.append(abs(k_p1 - k_tc) / (k_iso - k_p1))
    assert diffs[0] < 0.12 and diffs[1] < 0.6 * diffs[0]


def test_quadrature_too_coarse_for_the_harmonics_is_rejected():
    grid = TriGrid(shape=(4, 4, 2), side=1.0)
    with pytest.raises(ValueError, match="quadrature"):
        TriSNTransportSolver(grid, [_one_group(0.3, 0.1)], n_polar=1, n_azi=4,
                             scattering_order=4, engine="lu")


def test_transport_correction_keeps_the_balance_and_commutes_with_blending():
    from garnet.benchmarks.hpmr_core_pins import hpmr_pin_core_materials
    from garnet.blend import blend_materials

    _, pins = hpmr_pin_core_materials("g11")
    fuel, moderator = pins[1], pins[2]
    for method in ("outscatter", "diagonal"):
        with pytest.warns(RuntimeWarning) if method == "outscatter" else _no_warning():
            tm = transport_corrected(moderator, method)
        corr = (moderator.sigma_s_moments[1].sum(1) if method == "outscatter"
                else np.diag(moderator.sigma_s_moments[1]))
        np.testing.assert_allclose(tm.sigma_t, moderator.sigma_t - corr, rtol=1e-14)
        np.testing.assert_allclose(tm.diffusion, 1.0 / (3.0 * tm.sigma_t), rtol=1e-14)
        assert tm.sigma_s_moments is None
        # linear data: correcting a blend equals blending the corrections
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a = transport_corrected(blend_materials([fuel, moderator], [0.6, 0.4]), method)
            b = blend_materials([transport_corrected(fuel, method),
                                 transport_corrected(moderator, method)], [0.6, 0.4])
        np.testing.assert_allclose(a.sigma_t, b.sigma_t, rtol=1e-13)
        np.testing.assert_allclose(a.diffusion, b.diffusion, rtol=1e-13)


def test_library_total_is_the_true_total_and_moments_are_carried():
    from garnet.benchmarks.hpmr_core_pins import hpmr_pin_core_materials

    core, pins = hpmr_pin_core_materials("g11", absorption="library")
    for m in pins:
        assert m.scattering_order == 5
        np.testing.assert_allclose(m.sigma_s_moments[0], m.sigma_s, rtol=1e-14)
        # sigma_a + P0 out-scatter reproduces the stored total: a TRUE total
        np.testing.assert_allclose(m.sigma_a + m.sigma_s.sum(1), m.sigma_t, rtol=1e-3)
    # the YH moderator's P1 moment is a large fraction of its total
    moderator = pins[2]
    assert (moderator.sigma_s_moments[1].sum(1) / moderator.sigma_t).max() > 0.5
    # ... except Be, whose P0 matrix counts (n,2n) twice in the fastest group
    be = next(m for m in core if m.name == "be_reflector")
    excess = be.sigma_a + be.sigma_s.sum(1) - be.sigma_t
    assert excess[0] > 2 * be.sigma_a[0] and np.abs(excess[1:]).max() < 1e-4


def test_net_absorption_keeps_the_n2n_gain_in_every_solver():
    from garnet.benchmarks.hpmr_core_pins import hpmr_pin_core_materials

    core, pins = hpmr_pin_core_materials("g11")             # net by default
    for m in core[1:] + pins:                              # core[0] is the void
        # the balance a transport code solves: removal = total - self-scatter
        np.testing.assert_allclose(m.removal, m.sigma_t - np.diag(m.sigma_s),
                                   rtol=1e-12, atol=1e-15)
    be = next(m for m in core if m.name == "be_reflector")
    assert be.sigma_a[0] < 0 < be.removal.min()            # net (n,2n) production
    # net absorption and the transport correction commute
    lib = next(m for m in hpmr_pin_core_materials("g11", absorption="library")[0]
               if m.name == "be_reflector")
    a, b = (net_absorption(transport_corrected(lib, "diagonal")),
            transport_corrected(net_absorption(lib), "diagonal"))
    np.testing.assert_allclose(a.sigma_a, b.sigma_a, rtol=1e-13, atol=1e-16)
    # S_N's within-group scattering (total minus removal) is the stored P0
    grid = TriGrid(shape=(2, 2, 2), side=1.0)
    mmap = np.zeros(grid.shape, int)
    mmap[0] = 1                                            # Be row, fuel row
    s = TriSNTransportSolver(grid, [pins[1], be], mmap, n_polar=1, n_azi=4, engine="lu")
    np.testing.assert_allclose(s.ss_self[:, 0, 0, 0], np.diag(be.sigma_s), rtol=1e-12)


def test_out_scatter_correction_keeps_the_balance_in_s_n():
    # The out-scatter correction (0.4) exceeds the within-group P0 (0.3) of
    # group 1, so total - removal is negative there. S_N keeps it, and an
    # infinite medium returns the analytic k_inf from removal and transfer;
    # clipping it at zero would give 1.75.
    sa, nsf, s12 = np.array([0.02, 0.1]), np.array([0.01, 0.15]), 0.6
    p0 = np.array([[0.3, s12], [0.0, 0.9]])
    p1 = np.array([[0.1, 0.3], [0.0, 0.2]])
    mat = Material(name="m", diffusion=[1.0, 1.0], sigma_a=sa, nu_sigma_f=nsf,
                   sigma_s=p0, total=sa + p0.sum(1),
                   sigma_s_moments=np.array([p0, p1]))
    with pytest.warns(RuntimeWarning, match="negative"):
        tc = transport_corrected(mat, "outscatter")
    assert tc.sigma_t[0] < tc.removal[0]
    k_inf = nsf[0] / (sa[0] + s12) + nsf[1] * s12 / (sa[0] + s12) / sa[1]
    grid = TriGrid(shape=(4, 4, 2), side=1.0)
    k = TriSNTransportSolver(grid, [tc], n_polar=1, n_azi=4, scheme="scb",
                             engine="lu", bc="periodic").solve(**TIGHT).k_eff
    assert k == pytest.approx(k_inf, rel=1e-10)


def test_negative_absorption_needs_a_positive_removal():
    ok = Material(name="n2n", diffusion=[1.3, 0.4], sigma_a=[-0.01, 0.05],
                  nu_sigma_f=[0.0, 0.0], sigma_s=[[0.2, 0.03], [0.0, 0.8]])
    assert ok.removal.min() > 0
    with pytest.raises(ValueError, match="non-negative"):
        Material(name="bad", diffusion=[1.3, 0.4], sigma_a=[-0.04, 0.05],
                 nu_sigma_f=[0.0, 0.0], sigma_s=[[0.2, 0.03], [0.0, 0.8]])


class _no_warning:
    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("error", RuntimeWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)
