---
title: "Verification of anisotropic scattering in GARNET's triangular S$_N$ solver"
date: "17 November 2026 — branch SPH-dev"
geometry: margin=2.2cm
fontsize: 10pt
colorlinks: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

# Summary

`TriSNTransportSolver(scattering_order=L)` now scatters anisotropically up to
Legendre order $L$, using the library's moments $\Sigma_{s,l}$
(`Material.sigma_s_moments`). This note documents its verification.

- **Code-level identities.** Zero higher moments reproduce the isotropic solver
  to $10^{-11}$ in $k$. The two sweep engines agree to $10^{-10}$. An infinite
  medium keeps $k = k_\infty$ to $10^{-12}$. GPU and CPU agree to $10^{-7}$ on the
  HP-MR core.
- **Analytic criticality benchmarks.** The test set is LA-13511 [1]: bare
  cylinders and slabs that are exactly critical at a tabulated size, with
  linearly anisotropic or P2 scattering. In every problem the anisotropic case
  is as accurate as its isotropic control on the same meshes.
  - Cylinders converge to $-5.5$ and $-3.9$ pcm, against $-5.8$ pcm for the
    isotropic control. One of them has a strongly forward-peaked kernel that is
    negative for backward scattering.
  - Removing the anisotropy changes $k$ by 2900 to 21 600 pcm, so the
    benchmarks discriminate.
- **Independent reference.** A separately written 1-D S$_N$ solver converges to
  $k = 1$ within 3 pcm on every slab except one. On that one (two-group
  problem 71) it lands 31.5 pcm low, where GARNET is at about $-35$ pcm. The
  31.5 pcm offset is therefore in the problem's tabulated data, not in GARNET.
- 18 automated tests cover these checks: `tests/verification/test_tri_sn_anisotropic.py`
  and `tests/verification/test_tri_sn_sood.py`.

# 1 Scope and method

**Implementation.** The scattering source is expanded in real spherical
harmonics (Schmidt semi-normalized associated Legendre functions, polar axis
$z$). Only the $(l,k)$ with $l + k$ even are kept, since the others vanish in
2-D. The flux moments are $S_j = \sum_m w_m R_j(\Omega_m)\,\psi_m$, and the
angular source is $q_m = \sum_j (2l_j+1)\,R_j(\Omega_m)\,\Sigma_{s,l_j} S_j$,
flat per cell (and per corner for SCB). The quadrature's Gram matrix is checked
on construction, and a quadrature too coarse for $L$ is rejected. DSA uses
$D = 1/(3(\Sigma_t - \Sigma_{s1,gg}))$, and CMFD rescales the higher moments
with the scalar flux. The method works with both the level-scheduled and LU
sweep engines and with step and SCB differencing, in 2-D only.

**Verification levels.**

1. Code-level identities (unit tests).
2. The LA-13511 analytic set.
3. An independent 1-D discrete-ordinates reference, following OpenSN's
   anisotropic-scattering test [2].
4. Consistency checks on the pin-resolved HP-MR core.

**Why not C5G7.** The C5G7 benchmark specifies transport-corrected 7-group data
with isotropic scattering only. It cannot exercise the anisotropic path, and
OpenSN runs it as an isotropic regression.

# 2 Code-level checks

| Check | Result | Acceptance |
|---|---|---|
| Zero higher moments against the isotropic solve (step/SCB × LU/levels) | equal | $10^{-11}$ in $k$ |
| LU against level-scheduled engine, P2, SCB | equal | $10^{-10}$ |
| Periodic infinite medium, P2 | $k = k_\infty$ | $10^{-12}$ |
| Bare core, $\Sigma_{s1}/\Sigma_t = 0.4$: gap between P1 and the diagonal transport correction, as a share of the anisotropy effect, at 1 / 0.5 / 0.25 cm cells | 7.0 / 3.3 / 1.8 % | halves with $h$ |
| Quadrature unable to integrate the harmonics | rejected | `ValueError` |
| Transport-corrected data with a negative within-group term, infinite medium | analytic $k_\infty$ (1.4677) | $10^{-10}$ relative |
| HP-MR pin-resolved core, `sn-p2:2:8`, A100 against CPU | 0.9940510 both | $10^{-7}$ |

The P1/transport-correction gap is the first-order error of the cell-flat
anisotropic source. The two descriptions share the diffusion limit, so the gap
must vanish under refinement, and it does at first order. The
negative-within-group check guards a defect found and fixed during this work.
The solver used to clip $\Sigma_t - \text{removal}$ at zero, which created
neutrons wherever out-scatter-corrected data drive it negative. On HP-MR that
meant +1470 pcm at refine 16.

# 3 Analytic benchmarks (LA-13511)

LA-13511 [1] tabulates the exact critical size of homogeneous one- and
two-group media. $k - 1$ at that size is therefore the solver's error.
Cross sections are taken from the report's tables, with $\Sigma_{s,l}$ without
the $(2l+1)$ factor, as there. For the U-235 cylinders, LA-13511 gives the
anisotropic part as $\Sigma_{s1} = 0.1$ and $0.5 \times c\,\Sigma_t$. The source
solution [4] makes all secondary neutrons linearly anisotropic, which at
$k = 1$ is the same emission source as anisotropic scattering alone.

| No. | Identifier | Geometry | Scattering | Critical dimension |
|---|---|---|---|---|
| 13 | Ua-1-0-CY | cylinder | isotropic (control) | $r = 5.284935$ cm |
| 36 | Ua-1-1-CY | cylinder | P1, $\Sigma_{s1}/\Sigma_{s0} = 0.17$ | $r = 5.514296811$ cm |
| 37 | Ub-1-1-CY | cylinder | P1, $0.86$, kernel negative near $\mu=-1$ | $r = 6.940205668$ cm |
| 2, 12 | PUa-1-0-SL, Ua-1-0-SL | slab | isotropic (controls) | $a = 1.853722$, $2.872934$ cm |
| 34 | PUb-1-1-SL | slab | P1, kernel negative near $\mu=-1$ | $a = 0.79606$ cm |
| 35 | PUb-1-2-SL | slab | P2 | $a = 0.78396$ cm |
| 54 | URRa-2-0-SL | slab, two-group | isotropic (control) | $a = 7.566853$ cm |
| 71 | URRa-2-1-SL | slab, two-group | P1 within-group and fast$\to$slow | $a = 9.491600$ cm |

**Geometry on the triangular lattice.**

- **Cylinders.** Cells cut by the circle are blended with a void at their exact
  covered area (closed-form circle–triangle intersection), so the disk keeps
  its exact area. The vacuum boundary sits on the disk.
- **Slabs.** The solver has no reflective boundary. A slab is built from fuel
  strips followed by a purely absorbing, non-scattering layer 40 optical
  thicknesses deep, on a grid periodic in both directions. The layer returns no
  neutrons, so each face sees an exact vacuum and the slab is infinite along
  the strips. $k$ does not change (to 0.1 pcm) when the layer's optical depth
  per cell varies from 0.1 to 4.

Every problem was run on five meshes with quadrature (4, 16) and SCB, then at
(8, 32) and (12, 48) on the finest mesh. The table gives $k - 1$ in pcm.

| No. | Scattering | Finest mesh, (4, 16) | Mesh limit (Richardson) | (12, 48) $-$ (4, 16) | Estimate |
|---|---|---|---|---|---|
| 13 | isotropic | $-47.3$ | $-24.6$ | $+18.8$ | $\mathbf{-5.8}$ |
| 36 | P1 | $-45.5$ | $-25.0$ | $+19.5$ | $\mathbf{-5.5}$ |
| 37 | P1, negative kernel | $-41.2$ | $-28.8$ | $+24.9$ | $\mathbf{-3.9}$ |
| 2 | isotropic | $+97.9$ | $+98.4$ | $-133.8$ | $-35.4$ |
| 12 | isotropic | $+44.8$ | $+46.3$ | $-65.7$ | $-19.3$ |
| 34 | P1, negative kernel | $+67.3$ | $+67.4$ | $-96.6$ | $-29.2$ |
| 35 | P2 | $+74.8$ | $+75.1$ | $-101.2$ | $-26.0$ |
| 54 | isotropic, two-group | $-17.8$ ($-0.2$ at 5120 cells) | $+0.9$ | $-7.4$ | $\mathbf{-6.5}$ |
| 71 | P1, two-group | $-42.7$ ($-29.5$ at 5120 cells) | $-28.9$ | $-8.9$ | $\mathbf{-37.8}$ |

![Mesh convergence of $k$ at quadrature (4, 16): anisotropic problems next to their isotropic controls. The dashed line is the independent 1-D S$_N$ result for problem 71 (S256, mesh-extrapolated).](figures/convergence.pdf){width=100%}

**Reading the results.**

- **Cylinders.** The two P1 cylinders track the isotropic control at every mesh
  (figure, panel a) and extrapolate to the same few pcm.
- **One-group slabs.** These are optically thin (0.6–0.9 mean free paths across
  the half-thickness). Their residuals, isotropic and anisotropic alike, are
  the 2-D product quadrature: it still moves $k$ by 65–135 pcm between (4, 16)
  and (12, 48). The independent 1-D reference converges to $k = 1$ on all four
  (section 4), so these residuals are the GARNET quadrature, not the data.
- **Problem 71.** The isotropic two-group slab converges to $k = 1$, and its
  interior fluxes match the tabulated values within 0.013 %. The anisotropic
  slab 71 converges to about $-35$ pcm, with interior fluxes 0.1 % off.
  - The independent 1-D solver gives $-31.5$ pcm.
  - The fast$\to$slow P1 transfer alone is worth +215 pcm, and negating it
    would cost +430 pcm, so its sign and direction are right.
  - The tabulated critical half-thickness of problem 71 (from Bosler's 1972
    thesis [3], not accessible) is therefore about 30 pcm from the specified
    data.

**Discrimination.** The table below changes the scattering on the same mesh and
gives the effect on $k$, relative to the anisotropic solve. Every check above
would fail by thousands of pcm if the moments were ignored.

| No. | Mesh | Moments dropped (true total) | Diagonal transport correction | P2 dropped |
|---|---|---|---|---|
| 34 | 160 cells | $+5479$ pcm | $+2385$ pcm | – |
| 35 | 160 cells | $+4376$ pcm | $+1385$ pcm | $-1011$ pcm |
| 36 | 32 cells | $+3265$ pcm | $+326$ pcm | – |
| 37 | 32 cells | $+21614$ pcm | $+2625$ pcm | – |
| 71 | 320 cells | $+15390$ pcm | $+619$ pcm | – |

# 4 Independent 1-D reference

`tools/slab_sn_1d.py` shares no code with `garnet.tri_sn`; only the cross-section
tables are common. It solves plane-geometry transport with Legendre scattering:

- Gauss–Legendre S$_N$;
- diamond differencing on a uniform mesh;
- GMRES on the moments for each within-group problem;
- Gauss–Seidel over groups and power iteration for $k$.

This is the pattern of OpenSN's anisotropic-scattering verification [2].
Results are Richardson-extrapolated from 1000 and 2000 cells, $k - 1$ in pcm:

| No. | S64 | S128 | S256 |
|---|---|---|---|
| 2 (isotropic) | $-8.9$ | $-2.3$ | $-0.7$ |
| 12 (isotropic) | $-4.8$ | $-1.2$ | $-0.3$ |
| 34 (P1) | $-7.3$ | $-1.8$ | $-0.4$ |
| 35 (P2) | $-3.8$ | $+1.8$ | $+3.1$ |
| 54 (two-group, isotropic) | $-0.5$ | $-0.1$ | $0.0$ |
| 71 (two-group, P1) | $-32.2$ | $-31.7$ | $\mathbf{-31.5}$ |

# 5 Consistency on the HP-MR pin-resolved core

On the faithful 11-group HP-MR core (P0–P5 library data, refine 16), the
quantities below are given against P2 with quadrature (4, 16) on the same mesh.

| Variant | $\Delta k$, drums in / out | Pin-power RMS, in / out |
|---|---|---|
| P1 / P3, quadrature (4, 16) | $-17$ / $+1$, and $-2$ / $-1$ pcm | 0.02 / 0.03 %, 0.00 / 0.00 % |
| P2, quadrature (6, 24) | $-3.5$ / $-1.4$ pcm | 0.09 / 0.10 % |
| isotropic, out-scatter transport correction | $+27$ / $+19$ pcm | 0.14 / 0.06 % |
| isotropic, diagonal transport correction | $+103$ / $-109$ pcm | 0.86 / 0.71 % |
| isotropic, no correction (former references) | $+2200$ / $+1171$ pcm | 4.1 / 1.8 % |

P2 is converged in scattering order and quadrature. The out-scatter
transport-corrected isotropic solve is a close substitute, but only after the
negative-within-group fix.

# 6 Limitations

- **2-D only.** Prismatic (3-D) S$_N$ scatters isotropically.
- **Cell-flat anisotropic source.** The anisotropic contribution converges at
  first order in the cell size (section 2).
- **Optically thin slabs.** Accuracy there is limited by the product quadrature
  (about $\pm 35$ pcm at (12, 48)), for isotropic and anisotropic scattering
  alike.
- **LU engine.** The anisotropic path is about 15 times slower than the
  isotropic one (two-group slab: 32 s against 2 s on CPU). The level-scheduled
  engine is the production path.
- **No fixed-source mode.** OpenSN's fixed-source P3 slab test is therefore not
  replicated. P3 is covered by the HP-MR order study and the infinite-medium
  identity.
- **LA-13511 problems not run:**
  - the spheres;
  - the Pu-239 (a) anisotropic slabs 32 and 33, whose moments are illegible in
    the available copy;
  - the 1000 cm two-group D$_2$O slab, 73.

# 7 Reproduction

- **Tests.** `pytest tests/verification/test_tri_sn_anisotropic.py tests/verification/test_tri_sn_sood.py`
  (18 tests, about 2.5 min on CPU).
- **GARNET series.** `python tools/verify_sn_sood.py PROBLEMS gpu scb`, as
  Merlin7 jobs 252765–252767; the slabs use the LU engine on CPU. The 2560- and
  5120-cell points for 54 and 71 were separate runs of the same builder
  (`garnet.benchmarks.sood.build_sood_problem`).
- **Independent 1-D reference.** `python tools/slab_sn_1d.py 2,12,34,35,54,71 N 2000`
  for N = 64, 128, 256 (jobs 252779–252781).
- **Evidence.** All outputs are in `docs/qa/sph_audit_evidence/verify_sn_sood.out`.
  The figure is produced by `make_figures.py`. The broader context is
  `docs/qa/sph_audit_2026-11-13.md`, section 10.6.

# References

1. A. Sood, R. A. Forster, D. K. Parsons, *Analytical Benchmark Test Set for
   Criticality Code Verification*, LA-13511, Los Alamos National Laboratory (1999);
   also Prog. Nucl. Energy 42 (2003) 55–106.
2. OpenSN regression test `transport_anisotropic_scattering_verification.py`
   (directory `test/python/modules/linear_boltzmann_solvers/transport_steady`),
   <https://github.com/open-sn/opensn>.
3. G. E. Bosler, *Critical Slab Solution to the Two-Group Neutron Transport
   Equation for Linearly Anisotropic Scattering*, PhD thesis, University of
   Virginia (1972).
4. R. Sanchez, B. D. Ganapol, Nucl. Sci. Eng. 84 (1983) 61.
