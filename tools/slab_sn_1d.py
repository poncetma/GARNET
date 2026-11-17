"""Independent 1-D slab k-eigenvalue S_N (diamond difference, Gauss-Legendre).

    python tools/slab_sn_1d.py [PROBLEMS] [N_MU] [CELLS]

A reference written separately from garnet.tri_sn (no shared code beyond the
cross-section tables of garnet.benchmarks.sood), in the spirit of OpenSN's
transport_anisotropic_scattering_verification test: the plane-geometry
transport equation with Legendre scattering

    mu dpsi/dx + Sigma_t,g psi_g = 1/2 sum_g' sum_l (2l+1) Sigma_s,l(g'->g) P_l(mu) phi_l,g'
                                    + chi_g / (2 k) sum_g' nu Sigma_f,g' phi_0,g'

(phi_l = sum_n w_n P_l(mu_n) psi_n, weights summing to 2), vacuum on both
faces, Gauss-Legendre S_N, diamond differencing on a uniform mesh. Within-group
problems are solved by GMRES on the Legendre moments (one sweep per
matrix-vector product); groups by Gauss-Seidel; k by power iteration.
Diamond differencing is second order, so two meshes give a Richardson
estimate. Used to check the LA-13511 slab problems independently of the
triangular solver.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def slab_keff(nu_sf, chi, st, moments, half_thickness, cells, n_mu,
              tol=1e-11, max_outer=2000):
    """(k, phi0 (G, cells), x) for a bare slab of the given half-thickness.

    nu_sf, chi, st : (G,); moments : (L+1, G, G) [l][from][to], no (2l+1).
    """
    nu_sf, chi, st = (np.asarray(a, float) for a in (nu_sf, chi, st))
    S = np.asarray(moments, float)
    L1, G = S.shape[0], st.size
    mu, w = np.polynomial.legendre.leggauss(n_mu)                  # weights sum to 2
    P = np.array([np.polynomial.legendre.Legendre.basis(l)(mu) for l in range(L1)])
    h = 2.0 * half_thickness / cells
    x = (np.arange(cells) + 0.5) * h - half_thickness
    pos = mu > 0
    order_pos, order_neg = np.flatnonzero(pos), np.flatnonzero(~pos)

    def sweep(g, q):
        """Cell-average psi (n_mu, cells) for the angular source q (n_mu, cells)."""
        psi = np.empty((n_mu, cells))
        for idx, rng in ((order_pos, range(cells)), (order_neg, range(cells - 1, -1, -1))):
            a = 2.0 * np.abs(mu[idx]) / h
            inc = np.zeros(idx.size)
            for i in rng:
                avg = (q[idx, i] + a * inc) / (st[g] + a)
                psi[idx, i] = avg
                inc = 2.0 * avg - inc
        return psi

    def to_angular(mom):                                           # (L1, cells) -> (n_mu, cells)
        return 0.5 * ((2 * np.arange(L1) + 1)[:, None] * P).T @ mom

    def moments_of(psi):
        return (P * w) @ psi

    def within_group(g, ext):
        """Solve for the moments of group g given the external moments source ext (L1, cells)."""
        sgg = S[:, g, g][:, None]
        rhs = moments_of(sweep(g, to_angular(ext))).ravel()

        def matvec(v):
            m = v.reshape(L1, cells)
            return (m - moments_of(sweep(g, to_angular(sgg * m)))).ravel()

        A = LinearOperator((L1 * cells, L1 * cells), matvec=matvec, dtype=float)
        sol, info = gmres(A, rhs, rtol=1e-13, atol=0.0, restart=60, maxiter=400)
        if info != 0:
            raise RuntimeError(f"GMRES did not converge in group {g} (info {info})")
        return sol.reshape(L1, cells)

    phi = np.zeros((G, L1, cells))
    phi[:, 0] = 1.0
    k = 1.0
    for outer in range(max_outer):
        fission = (nu_sf[:, None] * phi[:, 0]).sum(axis=0)        # (cells,)
        new = np.zeros_like(phi)
        for g in range(G):
            ext = np.zeros((L1, cells))
            ext[0] = chi[g] * fission / k
            for gp in range(G):
                if gp != g:
                    src = new[gp] if gp < g else phi[gp]
                    ext += S[:, gp, g][:, None] * src
            new[g] = within_group(g, ext)
        new_fission = (nu_sf[:, None] * new[:, 0]).sum(axis=0)
        k_new = k * new_fission.sum() / fission.sum()
        change = abs(k_new - k)
        phi, k = new, k_new
        if change < tol and outer > 3:
            break
    return k, phi[:, 0], x


def problem_data(number):
    from garnet.benchmarks.sood import PROBLEMS, sood_material
    p = PROBLEMS[number]
    m = sood_material(p.material, p.order)
    S = (m.sigma_s_moments if m.sigma_s_moments is not None else m.sigma_s[None])
    return p, (m.nu_sigma_f, m.chi, m.sigma_t, S)


def main():
    numbers = ([int(n) for n in sys.argv[1].split(",")] if len(sys.argv) > 1
               else [2, 12, 34, 35, 54, 71])
    n_mu = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    cells = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    print(f"independent 1-D slab S_N: Gauss-Legendre S{n_mu}, diamond difference, "
          f"{cells // 2} and {cells} cells, Richardson (second order)\n", flush=True)
    for num in numbers:
        p, data = problem_data(num)
        ks = []
        for c in (cells // 2, cells):
            t0 = time.perf_counter()
            k, phi, x = slab_keff(*data, p.rc, c, n_mu)
            ks.append(k)
            line = (f"  {num:3d} {p.identifier:12s} cells {c:5d}: k - 1 = {1e5 * (k - 1):+9.3f} pcm "
                    f"[{time.perf_counter() - t0:.0f} s]")
            if p.flux:
                centre = np.interp(0.0, x, phi[0])
                vals = np.array([[np.interp(r * p.rc, x, phi[g]) for r in p.flux_positions[:-1]]
                                 for g in range(phi.shape[0])]) / centre
                err = 100 * (vals / np.array(p.flux)[:, :-1] - 1)
                line += f"  flux err interior max {np.abs(err).max():.4f} %"
            print(line, flush=True)
        print(f"  {num:3d} {p.identifier:12s} Richardson: k - 1 = "
              f"{1e5 * (ks[1] + (ks[1] - ks[0]) / 3.0 - 1):+9.3f} pcm\n", flush=True)


if __name__ == "__main__":
    main()
