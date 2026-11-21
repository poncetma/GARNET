"""One-dimensional spherical S_N k-eigenvalue solver.

Spherically symmetric problems (bare or layered spheres: Godiva, the LA-13511
sphere benchmarks) reduce to one radial dimension and one angular cosine mu,

    mu d psi/dr + (1 - mu^2)/r d psi/dmu + Sigma_t psi = S(r, mu),

with the angular redistribution term carrying neutrons toward mu = 1 as they
stream. Discretization (Lewis and Miller, "Computational Methods of Neutron
Transport", ch. 4):

* Gauss-Legendre ordinates mu_n (ascending, weights summing to 2), and the
  angular differencing coefficients alpha_{n+1/2} = alpha_{n-1/2} - w_n mu_n,
  alpha_{1/2} = alpha_{N+1/2} = 0, which make the discrete redistribution
  conserve neutrons and keep a spatially flat isotropic flux exact;
* the cell balance on shell i between radii r_{i-1/2}, r_{i+1/2}, areas
  A = 4 pi r^2 and volume V, diamond-differenced in r and in mu;
* the starting direction mu = -1, where the redistribution vanishes, swept
  inward first to give the angular cell-edge fluxes psi_{1/2, i};
* vacuum at the outer surface, reflection at the centre (A_{1/2} = 0);
* scattering as Legendre moments up to order L: S = sum_l (2l + 1)/2
  Sigma_s,l phi_l P_l(mu) + chi nu Sigma_f phi / (2 k), with
  phi_l = sum_n w_n P_l(mu_n) psi_n (``Material.sigma_s_moments``, moments
  without the (2l + 1) factor).

Within-group source iteration, Gauss-Seidel over groups and power iteration
on k. Diamond differencing can produce negative angular fluxes in optically
thick cells; ``fixup=True`` sets them to zero and recomputes the cell flux
from the balance (the solver reports how many fix-ups it made).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre

from .materials import Material

try:                                     # the sweep is a tight double loop
    from numba import njit
except ImportError:                      # pragma: no cover - numba is optional
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


@dataclass
class SphereSNResult:
    k_eff: float
    flux: np.ndarray          # (G, I) shell-average scalar flux
    converged: bool
    outer_iterations: int
    inner_iterations: int
    fixups: int
    solve_seconds: float


@njit(cache=True)
def _sweep(mu, w, alpha, Ap, Am, V, st, src, src_start, fixup):
    """One transport sweep of one group. src (N, I) and src_start (I,) are the
    total sources per unit mu (emission density / 2 for isotropic). Returns
    the shell-average angular flux (N, I) and the number of fix-ups."""
    N = mu.shape[0]
    I = V.shape[0]
    psi = np.zeros((N, I))
    edge = np.zeros(I)          # psi_{n-1/2, i}: angular edge flux of each shell
    inner = np.zeros(N)         # psi_{n, 1/2} at the centre, mu < 0 directions
    nfix = 0
    # starting direction mu = -1: -d psi/dr + st psi = s, swept inward
    out = 0.0
    for i in range(I - 1, -1, -1):
        dr = (np.sqrt(Ap[i] / (4.0 * np.pi)) - np.sqrt(Am[i] / (4.0 * np.pi)))
        c = 2.0 / dr
        p = (src_start[i] + c * out) / (st[i] + c)
        o = 2.0 * p - out
        if fixup and o < 0.0:
            # outgoing set to zero: -(in - 0)/dr + st p = s
            o = 0.0
            p = (src_start[i] + out / dr) / st[i]
            nfix += 1
        edge[i] = p
        out = o
    half = N // 2
    # mu < 0: from the surface inward
    for n in range(half):
        m = -mu[n]
        out = 0.0                       # vacuum: no incoming at the surface
        for i in range(I - 1, -1, -1):
            dA = Ap[i] - Am[i]
            a = dA / w[n]
            num = V[i] * src[n, i] + m * (Ap[i] + Am[i]) * out \
                + a * (alpha[n + 1] + alpha[n]) * edge[i]
            den = 2.0 * m * Am[i] + 2.0 * a * alpha[n + 1] + st[i] * V[i]
            p = num / den
            o_space = 2.0 * p - out
            o_angle = 2.0 * p - edge[i]
            if fixup and (o_space < 0.0 or o_angle < 0.0):
                nfix += 1
                # set the negative outgoing flux to zero and redo the balance
                fs = 0.0 if o_space < 0.0 else 1.0
                fa = 0.0 if o_angle < 0.0 else 1.0
                num2 = V[i] * src[n, i] + m * Ap[i] * out + a * alpha[n] * edge[i]
                den2 = st[i] * V[i]
                # outgoing faces: space (A-) and angle (alpha+), each either
                # diamond (2 p - in) or zero
                num2 += fs * m * Am[i] * out + fa * a * alpha[n + 1] * edge[i]
                den2 += fs * 2.0 * m * Am[i] + fa * 2.0 * a * alpha[n + 1]
                p = num2 / den2
                o_space = fs * (2.0 * p - out)
                o_angle = fa * (2.0 * p - edge[i])
            psi[n, i] = p
            out = o_space
            edge[i] = o_angle
        inner[n] = out
    # mu > 0: from the centre outward, reflected incoming at r = 0
    for n in range(half, N):
        m = mu[n]
        out = inner[N - 1 - n]
        for i in range(I):
            dA = Ap[i] - Am[i]
            a = dA / w[n]
            num = V[i] * src[n, i] + m * (Ap[i] + Am[i]) * out \
                + a * (alpha[n + 1] + alpha[n]) * edge[i]
            den = 2.0 * m * Ap[i] + 2.0 * a * alpha[n + 1] + st[i] * V[i]
            p = num / den
            o_space = 2.0 * p - out
            o_angle = 2.0 * p - edge[i]
            if fixup and (o_space < 0.0 or o_angle < 0.0):
                nfix += 1
                fs = 0.0 if o_space < 0.0 else 1.0
                fa = 0.0 if o_angle < 0.0 else 1.0
                num2 = V[i] * src[n, i] + m * Am[i] * out + a * alpha[n] * edge[i]
                den2 = st[i] * V[i]
                num2 += fs * m * Ap[i] * out + fa * a * alpha[n + 1] * edge[i]
                den2 += fs * 2.0 * m * Ap[i] + fa * 2.0 * a * alpha[n + 1]
                p = num2 / den2
                o_space = fs * (2.0 * p - out)
                o_angle = fa * (2.0 * p - edge[i])
            psi[n, i] = p
            out = o_space
            edge[i] = o_angle
    return psi, nfix


class SphereSNSolver:
    """k-eigenvalue S_N solver on concentric spherical shells.

    radii         : outer radii of the shells (I,), increasing; the first shell
                    starts at r = 0. Units of the cross sections (cm).
    materials     : a Material or list of them.
    material_map  : (I,) material index of each shell (default all 0).
    n_mu          : number of Gauss-Legendre ordinates (even).
    scattering_order : Legendre order L of the scattering source; None uses
                    the materials' moments (``sigma_s_moments``), 0 isotropic
                    (then the TRUE total with an isotropic kernel, unless the
                    data are transport-corrected).
    fixup         : negative-flux fix-up (set to zero, rebalance).
    """

    def __init__(self, radii, materials, material_map=None, n_mu: int = 16,
                 scattering_order: int | None = None, fixup: bool = True):
        r = np.asarray(radii, dtype=float)
        if r.ndim != 1 or r.size < 1 or np.any(np.diff(r) <= 0) or r[0] <= 0:
            raise ValueError("radii must be increasing and positive")
        if n_mu < 2 or n_mu % 2:
            raise ValueError("n_mu must be even and >= 2")
        mats = [materials] if isinstance(materials, Material) else list(materials)
        I = r.size
        mmap = (np.zeros(I, int) if material_map is None
                else np.asarray(material_map, dtype=int).reshape(I))
        self.G = mats[0].n_groups
        self.radii = r
        edges = np.concatenate(([0.0], r))
        self.Am = 4.0 * np.pi * edges[:-1] ** 2
        self.Ap = 4.0 * np.pi * edges[1:] ** 2
        self.V = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
        mu, w = legendre.leggauss(n_mu)            # ascending, sum(w) = 2
        self.mu, self.w = mu, w
        alpha = np.zeros(n_mu + 1)
        for n in range(n_mu):
            alpha[n + 1] = alpha[n] - w[n] * mu[n]
        alpha[-1] = 0.0                            # exact up to round-off
        self.alpha = alpha
        G = self.G
        L = (max((m.sigma_s_moments.shape[0] - 1) if m.sigma_s_moments is not None else 0
                 for m in mats) if scattering_order is None else int(scattering_order))
        self.L = L
        for m in mats:
            if m.n_groups != G:
                raise ValueError("all materials need the same group count")
        # per-shell data
        st = np.array([m.sigma_t for m in mats])[mmap]           # (I, G)
        self.st = st.T.copy()                                     # (G, I)
        self.nsf = np.array([m.nu_sigma_f for m in mats])[mmap].T
        self.chi = np.array([m.chi for m in mats])[mmap].T
        mom = np.zeros((len(mats), L + 1, G, G))
        for k, m in enumerate(mats):
            if m.sigma_s_moments is not None:
                n = min(L, m.sigma_s_moments.shape[0] - 1)
                mom[k, :n + 1] = m.sigma_s_moments[:n + 1]
            else:
                mom[k, 0] = m.sigma_s
        self.scat = mom[mmap]                                     # (I, L+1, G, G)
        self.P = np.array([legendre.legval(mu, np.eye(L + 1)[l]) for l in range(L + 1)])
        self.P_start = np.array([(-1.0) ** l for l in range(L + 1)])
        self._Pw = self.P * self.w
        self.fixup = bool(fixup)

    def _moments(self, psi):
        """Legendre flux moments (L+1, I) of an angular flux (N, I)."""
        # einsum, not @: tiny BLAS products cost milliseconds each under
        # multithreaded BLAS (35x slower P2 solves on a shared login node)
        return np.einsum("ln,ni->li", self._Pw, psi)

    def _group_source(self, g, mom, fission, k):
        """(N, I) and starting-direction (I,) source per unit mu for group g,
        from the moments of all groups (G, L+1, I) and the fission density."""
        L = self.L
        S_l = np.einsum("ilf,fli->li", self.scat[:, :, :, g], mom)   # (L+1, I)
        S_l *= ((2.0 * np.arange(L + 1) + 1.0) / 2.0)[:, None]
        iso = self.chi[g] * fission / (2.0 * k)
        src = np.einsum("ln,li->ni", self.P, S_l) + iso[None, :]
        start = self.P_start @ S_l + iso
        return src, start

    def solve(self, tol_k: float = 1e-10, tol_source: float = 1e-9,
              max_outer: int = 2000, max_inner: int = 500) -> SphereSNResult:
        t0 = time.perf_counter()
        G, L, I = self.G, self.L, self.V.size
        mom = np.zeros((G, L + 1, I))
        mom[:, 0] = 1.0
        k = 1.0
        fission = (self.nsf * mom[:, 0]).sum(axis=0)
        total = float((fission * self.V).sum())
        inner_total = fixups = 0
        converged = False
        outer = 0
        for outer in range(1, max_outer + 1):
            for g in range(G):
                for _ in range(max_inner):
                    src, start = self._group_source(g, mom, fission, k)
                    psi, nfix = _sweep(self.mu, self.w, self.alpha, self.Ap, self.Am,
                                       self.V, self.st[g], src, start, self.fixup)
                    fixups += nfix
                    inner_total += 1
                    new = self._moments(psi)
                    d = np.max(np.abs(new[0] - mom[g, 0])) / max(np.max(np.abs(new[0])), 1e-300)
                    mom[g] = new
                    if d < 0.1 * tol_source:
                        break
            fission_new = (self.nsf * mom[:, 0]).sum(axis=0)
            total_new = float((fission_new * self.V).sum())
            k_new = k * total_new / total
            change = np.max(np.abs(fission_new / total_new - fission / total)) \
                / max(np.max(np.abs(fission_new / total_new)), 1e-300)
            dk = abs(k_new - k)
            k, fission, total = k_new, fission_new, total_new
            if dk < tol_k and change < tol_source:
                converged = True
                break
        return SphereSNResult(k_eff=k, flux=mom[:, 0].copy(), converged=converged,
                              outer_iterations=outer, inner_iterations=inner_total,
                              fixups=fixups, solve_seconds=time.perf_counter() - t0)
