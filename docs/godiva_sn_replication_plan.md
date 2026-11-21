# Plan: the GeN-Foam Godiva S_N case in GARNET (neutronics only)

Status: items 0 and 1 done 2026-11-21 (reader `garnet/genfoam.py`, 1D spherical S_N `garnet/sphere_sn.py`, study `tools/godiva_1d.py`); items 2-5 (3D) not started.

## 1D results (2026-11-21)

GeN-Foam's own data (`Godiva_SN/constant/neutroRegion/nuclearData`), R = 8.7407 cm, GARNET 1D spherical S_N (diamond difference, Gauss–Legendre, P_L):

| Case | k | vs experiment k = 1 | vs GeN-Foam 0.980106 |
|---|---|---|---|
| P2 (GeN-Foam's order), mesh and angle extrapolated | **0.994256** | −574 pcm | +1415 pcm |
| P2, S64, 200 shells | 0.994301 | −570 pcm | +1420 pcm |

- **Discretization error:**
  - mesh: +0.03 pcm at 200 shells (25 → 400 shells changes k by 1.7 pcm);
  - angle: +4.5 pcm at S64, falling as N^-2. S8 is +241 pcm, S16 +63 pcm.
- **Scattering order (S64, 200 shells), relative to P2:**
  - P1: −481 pcm;
  - P3: −3 pcm;
  - P5: −2 pcm, so P2 is converged in Legendre order;
  - P0 with the true total: +10,737 pcm;
  - P0 transport-corrected: +332 pcm (out-scatter) and +402 pcm (diagonal). The isotropic shortcut costs 3–4 hundred pcm here, as the risk below expected.
- **Data conventions confirmed independently:**
  - The file's D equals 1/(3(Σt − Σ_g' Σs1[g→g'])) to 0.4 % in groups 2–18. So the P_l matrices are the plain moments (no 2l+1), with rows = from, as GeN-Foam's S_N assumes.
  - Σχ = 1 + 9e-8; β = 637 pcm.
- **Conclusion:**
  - These 23-group data give k = 0.9943 for Godiva when converged: 574 pcm below the experiment. That is a property of the data (group structure and homogenization), not of the solver.
  - GeN-Foam's tutorial value, 0.980106, lies 1415 pcm below the converged value for the same data.
  - Candidate causes, for the 3D steps to separate: 20 power iterations from a flat start; the 32-direction quadrature; first-order upwind on the tetrahedra.
  - The 1D cannot reproduce a 3D ordinate set, so the quadrature share needs item 2.

## The GeN-Foam case

`Tutorials/reactorCases/Godiva_SN` (GeN-Foam develop, 2025-09-30):

- **Geometry:** a bare HEU sphere, R = 8.74 cm.
- **Mesh:** a full-sphere, mostly tetrahedral mesh, 177,655 cells (about 4.5 faces per cell), with a single vacuum patch.
- **Data:** `nuclearData`, in SI units (per metre):
  - 23 groups;
  - P0–P2 scattering matrices (`legendreMoments 3`);
  - prompt and delayed spectra, 1/v;
  - 8 delayed families (β, λ);
  - one zone, `uranium`.
- **Quadrature:** read from files.
  - Default `quadratureSet`: 4 directions per octant, three permutations of (0.210, 0.210, 0.955) plus (1/√3, 1/√3, 1/√3); 32 directions.
  - `quadratureSet16`: 16 per octant.
  - `quadratureSet1`: 1 per octant.
  - Chebyshev–Legendre S4 and S8 sets.
- **Solver:**
  - finite volume, first-order upwind;
  - each (group, direction) solved by BiCGStab/DILU;
  - at most 3 inner iterations per group;
  - k by power iteration through the time loop; no acceleration.
- **Its "reference"** is a regression value: `Alltest` runs 20 power iterations on 4 cores and checks k = 0.980106 to 0.1 %. That is neither converged in iterations nor in mesh. Godiva is critical (k = 1.000 ± 0.001), so the 23-group data, if homogenized consistently, should give k close to 1.

A replication should therefore produce a converged k for these data and its mesh and angular convergence, and explain the 0.980, rather than match 0.980106.

## Can GARNET approximate it with a structured mesh? Yes

GARNET's 3D S_N is the extruded triangular-prism lattice (`TriSNTransportSolver` on `TriGrid(shape=(nr, nc, 2, nz))`):

- step and SCB (3 unknowns per prism);
- CPU LU or GPU level-scheduled sweeps;
- DSA and CMFD.

A sphere on prisms has two discretization choices:

- **Staircase:** a prism is fuel if its centroid lies inside R, otherwise it is excised. Excised cells already act as vacuum (`active`). The fuel volume error is controlled by adjusting R so the staircase mass is preserved.
- **Volume-fraction blend:** boundary prisms take fuel at their exact volume fraction, mixed with void (the `mix_weight` mechanism used for the drum arcs). This conserves mass and converges smoothly in h, at the cost of a smeared surface.

Both converge as h → 0. Comparing them gives the geometric error bar.

**Size.** At h = 0.25 cm (side) and dz = 0.25 cm, the sphere holds about 300k prisms. With SCB and 32 directions the sweep tables are about 0.7 GB per group, about 16 GB for 23 groups, which fits a GH200. The 128-direction set (16 per octant) would need about 64 GB and per-group table rebuilds; the step scheme needs far less.

**Rejected alternatives:**
- **3D Cartesian voxels:** the Cartesian S_N engine is 2D only (`nz == 1`), so it would need a 3D wavefront sweep. More work than the prisms, no gain.
- **The GeN-Foam tetrahedral mesh itself:** needs an unstructured 3D S_N engine. That is a superset of the 2D unstructured engine in `docs/arc_conforming_reference_plan.md`, several weeks.

**Cheap exact companion: 1D spherical S_N.** Godiva is spherically symmetric. A 1D spherical S_N gives a mesh- and angle-converged k for these data at negligible cost:
- weighted diamond difference with angular redistribution;
- Gauss–Legendre S_N in μ;
- P_L scattering in Legendre polynomials of μ.

It is the reference both 3D models should converge to. It also checks the data independently of any 3D geometry error.

## Can we add the required 3D quadrature? Yes, easily

GARNET already has a 3D product quadrature (Gauss–Legendre polar × uniform azimuthal, full sphere). The sweeps take any ordinate set (μ, η, ξ, w). Needed:

- **A `quadrature=(mu, eta, xi, w)` option** on the tri S_N engine, plus a reader for GeN-Foam's `quadratureSet` files: the octant lists expanded by sign reflection, weights normalized to 1.
- **Level-symmetric and Chebyshev–Legendre sets,** for convergence studies against the product sets.
- **The existing moment-integration check,** extended to the full 3D basis. With P2 scattering the set must integrate the l ≤ 2 harmonics exactly; the 4-per-octant set must be checked, and is refused or flagged if it fails.

## What has to be built

| # | Item | Work | Estimate |
|---|---|---|---|
| 0 | **GeN-Foam `nuclearData` reader** to GARNET `Material` and `Kinetics` | per m → per cm; P0–P2 matrices with the `scatteringMatrixP<l>` orientation checked (from → to); prompt and delayed χ; IV → speeds; β, λ. Test: values and one-group balance. | 0.5 day |
| 1 | **1D spherical S_N** (reference) | as above; k-eigen with Anderson. Verification: Sood LA-13511 bare spheres, exactly critical, P0 and anisotropic. Broken on purpose: redistribution term dropped. | 1.5–2 days |
| 2 | **3D quadrature sets** | user sets in the tri engine, the GeN-Foam reader, level-symmetric / Chebyshev–Legendre generators. Moment check in 3D. Test: P_L integration and k invariance under octant permutation. | 0.5–1 day |
| 3 | **P_L scattering in 3D** | full spherical-harmonic basis, all (l, k): only the 2D build keeps the z-even half. Angular source in the 3D level sweep and the LU path. DSA with D = 1/(3(Σt − Σs1)). Tests: P2 infinite medium; extruded 2D problem reproduced in 3D; Sood spheres with P1/P2 scattering. | 2–3 days |
| 4 | **Godiva prism geometry** | staircase (mass-preserving) and volume-fraction blends; builder taking R, h, dz. Test: fuel volume exact (blend) or within tolerance (staircase). | 0.5–1 day |
| 5 | **Runs and analysis** | 1D: mesh and S_N convergence. 3D prisms: h = 0.5 / 0.35 / 0.25 cm, both surface treatments, 32 and 128 directions. Comparison with GeN-Foam's 0.980106 and with k = 1. | about 1 day of work, a few GPU-hours |

**Total:**
- **About 6–9 working days** for the full comparison: 3D P2 on prisms plus the 1D spherical reference.
- **About 3 days** for a reduced version: the 1D spherical reference plus 3D prisms with transport-corrected P0. That version is not like-for-like with GeN-Foam's P2: in a fast system, transport-correcting the P2 anisotropy can cost a few hundred pcm.
- **Optional:** running GeN-Foam ourselves (OpenFOAM is not installed here; a container would do) to converge its own k with more iterations and finer meshes.

## Risks

- **The data may not reproduce k ≈ 1 even when converged.** That depends on how they were homogenized and on the transport correction already in them. The 1D reference settles this before any 3D work.
- **Memory for 128 directions in 3D SCB:** per-group sweep tables (as `lu_cache="group"` does for the LU engine), or the step scheme.
- **Sphere staircase on prisms:** the error falls slowly, so it should be bracketed by the two surface treatments and extrapolated in h.
