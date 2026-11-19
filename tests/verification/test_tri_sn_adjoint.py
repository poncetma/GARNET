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
"""

import functools

import numpy as np
import pytest

from garnet import Material
from garnet.tri import TriGrid
from garnet.tri_sn import TriSNTransportSolver

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
