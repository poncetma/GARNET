"""Adjoint k-eigenproblem of the triangular S_N solver (``adjoint=True``).

The adjoint angular flux is the forward-type solution with every group-to-group
transfer transposed (all Legendre orders) and the fission spectrum and
production swapped, at reversed direction. Checks:

* one group: the problem is self-adjoint, so the adjoint scalar flux equals the
  forward one to round-off (both schemes, both engines, P0 and P2, non-convex
  vacuum boundary);
* several groups with upscatter, P2 transfer moments, a volume-mixed absorber:
  the adjoint k equals the forward k to the solver tolerance, while the
  adjoint flux differs from the forward flux;
* negative controls: an adjoint without the transposition, with only P0
  transposed, or without the fission swap misses k by 300-14 000 pcm, so the
  k test can fail;
* perturbation reciprocity: for perturbations of down- and upscatter
  transfer, within-group scattering and nu*Sigma_f, the first-order estimate
  -<phi+, (dS + dF/k) phi> / <phi+, F phi> matches the direct change of 1/k
  with an error that falls 10x when the perturbation falls 10x (second
  order), on both schemes. Broken on purpose (the forward flux as the
  weight), the transfer cases miss by 65-92 % and the nu*Sigma_f case by
  2.1e-3 with a first-order error ratio of 5: both criteria are needed.
* point-kinetics parameters (``sn_effective_kinetics``): beta_eff against a
  shift of the fission spectrum toward the delayed spectra (second order,
  ratio 10.000); Lambda from the P2 flux moments against the 1/v insertion
  (Sigma_t + eps/v, central difference: the exact first-order Lambda):
  step 1.2e-4, while the scalar-flux (P0) Lambda misses by 4.1 %. Broken on
  purpose (adjoint odd moments not sign-flipped) Lambda misses by 8.5 %.
* cost: the adjoint sweeps the groups thermal to fast (its transfers are
  transposed), in the outer loop and in the CMFD power iteration, so it
  needs about the forward's CMFD passes per outer (7.3 against 6.3). Broken on
  purpose (the CMFD power in the forward order) it takes 50 passes per outer
  (84 with the outer loop also in the forward order); at HP-MR refine 32 those
  host LU solves ran for hours.
"""

import functools

import numpy as np
import pytest

from garnet import Kinetics, Material
from garnet.tri import TriGrid
from garnet.tri_sn import TriSNTransportSolver, sn_effective_kinetics

TIGHT = dict(tol_k=1e-12, tol_source=1e-11)
N = 10


def _mat3(name, st, s0, s1, s2, nsf, chi=(0.8, 0.2, 0.0), sa_shift=(0.0, 0.0, 0.0)):
    s0, st = np.array(s0, float), np.array(st, float)
    sa = st - s0.sum(axis=1) - np.array(sa_shift)
    return Material(name=name, diffusion=1.0 / (3.0 * st), sigma_a=sa, nu_sigma_f=nsf,
                    sigma_s=s0, chi=list(chi), total=st,
                    sigma_s_moments=np.array([s0, s1, s2]))


FUEL = dict(st=[0.30, 0.55, 0.90],
            s0=[[0.20, 0.06, 0.010], [0.0, 0.40, 0.09], [0.0, 0.02, 0.70]],
            s1=[[0.06, 0.010, 0.001], [0.0, 0.10, 0.015], [0.0, 0.004, 0.12]],
            s2=[[0.01, 0.002, 0.0], [0.0, 0.02, 0.003], [0.0, 0.001, 0.02]],
            nsf=[0.008, 0.02, 0.15])
REFL = _mat3("refl", [0.35, 0.60, 0.95],
             [[0.24, 0.10, 0.004], [0.0, 0.48, 0.11], [0.0, 0.05, 0.89]],
             [[0.12, 0.03, 0.001], [0.0, 0.20, 0.03], [0.0, 0.02, 0.30]],
             [[0.03, 0.005, 0.0], [0.0, 0.05, 0.006], [0.0, 0.004, 0.06]],
             [0.0, 0.0, 0.0], chi=(0.0, 0.0, 0.0))
ABS = _mat3("abs", [0.40, 0.90, 3.0],
            [[0.25, 0.05, 0.002], [0.0, 0.40, 0.05], [0.0, 0.01, 0.5]],
            [[0.05, 0.005, 0.0], [0.0, 0.05, 0.005], [0.0, 0.0, 0.05]],
            np.zeros((3, 3)), [0.0, 0.0, 0.0], chi=(0.0, 0.0, 0.0))


def _fuel(**perturb):
    d = {k: np.array(v, float).copy() for k, v in FUEL.items()}
    shift = np.zeros(3)
    for (gf, gt), eps in perturb.get("transfer", {}).items():
        d["s0"][gf, gt] += eps
    for g, eps in perturb.get("self", {}).items():
        shift[g] += eps                 # lower absorption: self-scatter = total - removal rises
    for g, eps in perturb.get("nsf", {}).items():
        d["nsf"][g] += eps
    return _mat3("fuel", d["st"], d["s0"], d["s1"], d["s2"], d["nsf"], sa_shift=shift)


def _geometry():
    grid = TriGrid(shape=(N, N, 2), side=1.5)
    mmap = np.ones((N, N, 2), int)
    mmap[2:7, 2:7] = 0
    mmap[0:2, 4:7] = 2
    mix_m = np.full((N, N, 2), -1)
    mix_w = np.zeros((N, N, 2))
    mix_m[8, 2:5], mix_w[8, 2:5] = 2, 0.35          # a volume-mixed absorber strip
    active = np.ones((N, N, 2), bool)
    active[N - 1, :4] = False                       # a non-convex boundary
    return grid, mmap, mix_m, mix_w, active


@functools.lru_cache(maxsize=None)
def _base(scheme, adjoint):
    return _solve([_fuel(), REFL, ABS], adjoint, scheme)


def _solve(mats, adjoint, scheme="step", order=2, engine="lu"):
    grid, mmap, mix_m, mix_w, active = _geometry()
    s = TriSNTransportSolver(grid, mats, mmap, active=active, n_polar=2, n_azi=8,
                             scheme=scheme, engine=engine, scattering_order=order,
                             mix_material=mix_m, mix_weight=mix_w, adjoint=adjoint)
    return s, s.solve(**TIGHT)


def _one_group(name, st, s0, s1, s2, nsf):
    return Material(name=name, diffusion=[1 / (3 * st)], sigma_a=[st - s0], nu_sigma_f=[nsf],
                    sigma_s=[[s0]], total=[st],
                    sigma_s_moments=np.array([[[s0]], [[s1]], [[s2]]]))


@pytest.mark.parametrize("scheme,engine,order", [("step", "lu", 0), ("step", "levels", 2),
                                                 ("scb", "lu", 2), ("scb", "levels", 0)])
def test_one_group_is_self_adjoint(scheme, engine, order):
    mats = [_one_group("f", 0.6, 0.45, 0.15, 0.03, 0.18), _one_group("r", 0.8, 0.77, 0.4, 0.1, 0.0)]
    grid, mmap, _, _, active = _geometry()
    kw = dict(n_polar=2, n_azi=8, scheme=scheme, engine=engine, scattering_order=order,
              active=active)
    rf = TriSNTransportSolver(grid, mats, np.minimum(mmap, 1), **kw).solve(**TIGHT)
    ra = TriSNTransportSolver(grid, mats, np.minimum(mmap, 1), adjoint=True, **kw).solve(**TIGHT)
    f, a = np.asarray(rf.flux)[0][active], np.asarray(ra.flux)[0][active]
    assert ra.k_eff == pytest.approx(rf.k_eff, abs=1e-12)
    np.testing.assert_allclose(a / a.sum(), f / f.sum(), rtol=1e-11)


@pytest.mark.parametrize("scheme", ["step", "scb"])
@pytest.mark.parametrize("order", [0, 2])
def test_multigroup_adjoint_has_the_forward_k_and_its_own_flux(scheme, order):
    mats = [_fuel(), REFL, ABS]
    _, rf = _solve(mats, False, scheme, order)
    _, ra = _solve(mats, True, scheme, order)
    assert ra.k_eff == pytest.approx(rf.k_eff, abs=1e-11)
    f, a = np.asarray(rf.flux), np.asarray(ra.flux)
    assert np.abs(a / a.max() - f / f.max()).max() > 0.3      # not the forward flux


def _broken(kind):
    def make(self):
        G = self.G
        if kind != "no transposition":
            self.scatter = [[self.scatter[gt][gf] for gt in range(G)] for gf in range(G)]
        if kind == "fission not swapped":
            self._scat_l = [[[sl[gt][gf] for gt in range(G)] for gf in range(G)]
                            for sl in self._scat_l]
            return
        fissile = np.any(self.nsf > 0.0, axis=0)
        self.nsf, self.chi = np.where(fissile[None], self.chi, 0.0), self.nsf.copy()
    return make


@pytest.mark.parametrize("kind,margin", [("no transposition", 1e-2), ("P0 transposed only", 1e-3),
                                         ("fission not swapped", 1e-2)])
def test_broken_adjoints_miss_k(kind, margin, monkeypatch):
    """Negative controls: each ingredient of the transform matters, so the k
    test above can fail (P0 only: ~300 pcm; the others ~14 000 pcm)."""
    mats = [_fuel(), REFL, ABS]
    _, rf = _solve(mats, False)
    monkeypatch.setattr(TriSNTransportSolver, "_make_adjoint", _broken(kind))
    _, ra = _solve(mats, True)
    assert abs(ra.k_eff - rf.k_eff) > margin


PERTURBATIONS = {"down-transfer": dict(transfer={(0, 2): 1.0}),
                 "upscatter": dict(transfer={(2, 1): 1.0}),
                 "within-group": dict(self={1: 1.0}), "nu-fission": dict(nsf={1: 1.0})}


@pytest.mark.parametrize("scheme,case", [("step", c) for c in PERTURBATIONS]
                         + [pytest.param("scb", c, marks=pytest.mark.slow)
                            for c in ("upscatter", "nu-fission")])
def test_first_order_perturbation_with_the_adjoint_is_second_order_accurate(scheme, case):
    perturbation = PERTURBATIONS[case]
    sf, rf = _base(scheme, False)
    _, ra = _base(scheme, True)
    G = sf.G
    phi = [np.asarray(rf.flux[g]).reshape(-1) for g in range(G)]
    adj = [np.asarray(ra.flux[g]).reshape(-1) for g in range(G)]
    lam = 1.0 / rf.k_eff
    fission = sum(sf.nsf[g].reshape(-1) * phi[g] for g in range(G))
    denom = sum((adj[g] * sf.chi[g].reshape(-1) * fission).sum() for g in range(G))
    errors = []
    for eps in (1e-3, 1e-4):
        scaled = {k: {i: eps * v for i, v in d.items()} for k, d in perturbation.items()}
        sp, rp = _solve([_fuel(**scaled), REFL, ABS], False, scheme)
        assert np.allclose(sp.st, sf.st)                  # only isotropic terms change
        dS = 0.0
        for gt in range(G):
            acc = (sp.ss_self[gt] - sf.ss_self[gt]).reshape(-1) * phi[gt]
            for gf in range(G):
                if gf != gt:
                    new = 0.0 if sp.scatter[gf][gt] is None else sp.scatter[gf][gt]
                    old = 0.0 if sf.scatter[gf][gt] is None else sf.scatter[gf][gt]
                    acc = acc + np.asarray(new - old).reshape(-1) * phi[gf]
            dS += (adj[gt] * acc).sum()
        dfission = sum((sp.nsf[g] - sf.nsf[g]).reshape(-1) * phi[g] for g in range(G))
        dF = sum((adj[g] * sf.chi[g].reshape(-1) * dfission).sum() for g in range(G))
        predicted = -(dS + lam * dF) / denom
        direct = 1.0 / rp.k_eff - lam
        errors.append(abs(predicted - direct) / abs(direct))
    assert errors[1] < 1.2e-3
    assert errors[0] / errors[1] == pytest.approx(10.0, rel=0.05)      # second order


V_KIN = np.array([2.0e9, 4.0e7, 3.0e5])
BETA = np.array([0.0025, 0.004])
CHI_D = np.array([[0.5, 0.5, 0.0], [0.3, 0.7, 0.0]])
KIN = Kinetics(velocities=V_KIN, beta=BETA, decay=[0.08, 1.2], chi_delayed=CHI_D)


def _with(m, chi=None, eps_v=0.0):
    return Material(name=m.name, diffusion=m.diffusion,
                    sigma_a=np.asarray(m.sigma_a) + eps_v / V_KIN, nu_sigma_f=m.nu_sigma_f,
                    sigma_s=m.sigma_s, chi=m.chi if chi is None else chi,
                    total=np.asarray(m.total) + eps_v / V_KIN,
                    sigma_s_moments=m.sigma_s_moments)


def test_beta_eff_matches_a_delayed_spectrum_perturbation():
    sf, rf = _base("step", False)
    sa, ra = _base("step", True)
    p = sn_effective_kinetics(sf, rf, sa, ra, KIN)
    assert p.beta_total > BETA.sum() * 1.1          # softer delayed spectra matter here
    fuel = _fuel()
    shift = (BETA[:, None] * (CHI_D - np.asarray(fuel.chi)[None])).sum(axis=0)
    lam = 1.0 / rf.k_eff
    errors = []
    for eps in (1e-1, 1e-2):
        k = _solve([_with(fuel, chi=np.asarray(fuel.chi) + eps * shift), REFL, ABS],
                   False)[1].k_eff
        predicted = -lam * eps * (p.beta_total - BETA.sum())
        errors.append(abs(predicted - (1.0 / k - lam)) / abs(1.0 / k - lam))
    assert errors[1] < 1e-4
    assert errors[0] / errors[1] == pytest.approx(10.0, rel=0.05)


def test_generation_time_from_moments_matches_the_inverse_velocity_insertion():
    sf, rf = _base("step", False)
    sa, ra = _base("step", True)
    p = sn_effective_kinetics(sf, rf, sa, ra, KIN)
    p0 = sn_effective_kinetics(sf, rf, sa, ra, KIN, angular=False)
    eps = 10.0
    k_up, k_down = (_solve([_with(m, eps_v=s * eps) for m in (_fuel(), REFL, ABS)],
                           False)[1].k_eff for s in (1.0, -1.0))
    exact = rf.k_eff * (1.0 / k_up - 1.0 / k_down) / (2.0 * eps)
    assert p.generation_time == pytest.approx(exact, rel=1e-3)          # 1.2e-4
    assert abs(p0.generation_time / exact - 1.0) > 1e-2                 # 4.1 %: moments matter


def _cmfd_passes(adjoint, monkeypatch=None):
    """(k, CMFD power passes per outer) on the three-group core."""
    grid, mmap, mix_m, mix_w, active = _geometry()
    s = TriSNTransportSolver(grid, [_fuel(), REFL, ABS], mmap, active=active, n_polar=2,
                             n_azi=8, scheme="scb", scattering_order=2, mix_material=mix_m,
                             mix_weight=mix_w, adjoint=adjoint)
    calls = [0]
    make = s._make_cmfd_solver

    def counted(A):
        f = make(A)

        def solve(b):
            calls[0] += 1
            return f(b)
        return solve

    s._make_cmfd_solver = counted
    r = s.solve(**TIGHT)
    return r.k_eff, calls[0] / (r.outer_iterations * s.G)


def test_adjoint_cmfd_power_sweeps_groups_thermal_to_fast(monkeypatch):
    kf, fwd = _cmfd_passes(False)
    ka, adj = _cmfd_passes(True)
    assert ka == pytest.approx(kf, abs=1e-11)
    assert adj < 1.5 * fwd                                  # 7.3 against 6.3
    import garnet.tri_sn as tri_sn
    power = tri_sn._cmfd_power
    monkeypatch.setattr(tri_sn, "_cmfd_power",
                        lambda *a, group_order=None, **kw: power(*a, **kw))
    kb, broken = _cmfd_passes(True)
    assert kb == pytest.approx(kf, abs=1e-10)               # same answer, ...
    assert broken > 5 * fwd                                 # ... 50 passes per outer
