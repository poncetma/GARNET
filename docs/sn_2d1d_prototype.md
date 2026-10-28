# CPU planar S_N / axial diffusion prototype

The first four [implementation steps](sn_2d1d_implementation_plan.md), M0–M3,
are implemented, together with the eigenvalue iteration part of M4. The
prototype solves steady **prescribed multigroup sources** and **k-eigenvalue
problems** on triangular-prism extrusions, including self-scattering,
downscatter, and upscatter. It returns scalar flux, both sets of interface
currents, and conservation/convergence diagnostics.

## Run it

From a checkout:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  python3 examples/sn_2d1d_fixed_source.py --compare-3d

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  python3 examples/sn_2d1d_fixed_source.py --case c5g7 --compare-3d

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  python3 examples/sn_2d1d_fixed_source.py --eigenvalue --case c5g7 --compare-3d
```

Both commands run step and SCB. `--scheme step`, `--layers 16`, and
`--height 12` select individual configurations. `--output result.json` saves
cross-section/source/geometry provenance, implementation checksums, quadrature,
iteration counts, timings, and residuals. All material data are explicit or
already bundled; no cross-section generation or download occurs.

The C5G7 option uses its existing seven-group tables in a small derived
triangular problem. It is not an official C5G7 geometry or reference result.

## Low-level use

```python
import numpy as np
from garnet import Material, TriGrid
from garnet.sn_2d1d import SN2D1DSolver

grid = TriGrid((4, 4, 2, 8), side=2.0, height=12.0)
material = Material(diffusion=[0.8], sigma_a=[0.12], nu_sigma_f=[0.0])
solver = SN2D1DSolver(grid, material, axial_bc=("reflective", "vacuum"),
                     scheme="scb", n_polar=2, n_azi=8)
source = np.ones((1,) + grid.shape)
result = solver.solve(source, tol=1e-9)
assert result.converged, result.reason
flux = result.flux
print(result.balance_residual, result.flux_disagreement)
```

`material_map`, `active`, `mix_material`, and `mix_weight` follow the existing
triangular solvers. The active footprint must remain constant in z; material
and blend data may vary by layer. Identical material planes share an existing
S_N instance. The prototype runs on CPU with `engine="lu"` or `"levels"`.

The radial boundary is vacuum, or periodic with the CPU LU engine. Axial
boundaries independently use the existing diffusion Robin vocabulary,
including reflective and Marshak vacuum. Axial periodicity is unsupported.
For reflective ends each column needs positive removal somewhere, so its
standalone diffusion solve is nonsingular. Coefficients are fixed at
construction; rebuild the solver after material changes.

`solve(source)` retains its prescribed-source meaning: production coefficients
do not add a source automatically. To converge fission multiplication, use
`solve_eigenvalue()`:

```python
fissile = material.replace(nu_sigma_f=[0.15])
eigensolver = SN2D1DSolver(grid, fissile, axial_bc=("reflective", "vacuum"),
                          scheme="scb", n_polar=2, n_azi=8)
eigen = eigensolver.solve_eigenvalue(tol_k=1e-8, tol_source=1e-8)
assert eigen.converged, eigen.reason
print(eigen.k_eff, eigen.outer_iterations, eigen.balance_residual)
```

### Eigenvalue iteration

The fission source uses the existing `Fields.fission_source` and blended
emission spectra. Each power update solves a prescribed source
`chi * sum_g(nu_sigma_f[g] * phi[g]) / k_eff` through the verified fixed-source
solver. Scattering and upscatter remain implicit in that inner solve; an
additional lagged energy iteration is unnecessary.

The eigenvalue update uses the ratio of volume-integrated fission production.
The flux is normalized to unit integrated production after each accepted
update. At the end, both current fields, leakages, radial/axial scalar fluxes,
and balance defects are reconstructed from that normalized flux and its own
fission source. This prevents a stationary eigenvalue from concealing an
inaccurate inner solve or a lagged final RHS.

Convergence requires relative eigenvalue change below `tol_k`, normalized
fission-source and scalar-flux changes below `tol_source`, and all three
physical residuals below `tol_source`. `tol_inner` defaults to the smaller of
1e-9 and 5% of the outer tolerances; set it explicitly when comparing inner
accuracy. `max_outer` limits power updates, and `max_inner` bounds each
fixed-source solve. `coupling="picard"` and its relaxation are also supported.
Inner failure, exhausted updates, and failed final balance checks return
`converged=False` with a reason.

`initial_k` must be positive. `initial_flux` may broadcast to the grid and
must be non-negative on active cells with positive fission production. The
result records `k_history`, relative k/source/flux-change histories, and inner
iteration counts. `iterations` totals inner work; `outer_iterations` counts
accepted power updates. No CMFD acceleration is applied in this prototype.
`fission_source` has shape `grid.shape` and contains
`sum_g(nu_sigma_f[g] * flux[g])`; its volume integral is one.

## Reused implementation and coupling

- `_sn_plane.py` adapts `TriSNTransportSolver` without changing its step/SCB
  discretization, sparse factors, level sweeps, or face-flux definitions.
  Its within-group path reuses `_solve_group`; returned arrays own their data.
- `axial_diffusion.py` uses the existing `harmonic_mean`, `normalize_bc`,
  `face_alpha`, and `robin_face_term` helpers. On uniform columns its action
  is verified against the existing `GroupOperator`. A batched Thomas solve
  adds independent-column inversion and caches factors.
- `sn_2d1d.py` reuses `Fields` for material lookup/blending. One column per
  active triangle avoids additional homogenization.

The default coupling solves

```text
(I - T S + T Lz) phi = T q
```

with SciPy GMRES. `T` is the existing planar sweep, `S` the complete
multigroup scattering operator, and `Lz` the axial current divergence. The
preconditioner solves axial removal/diffusion columns, reducing stiffness as
the axial mesh is refined. Final angular/face-current reconstruction uses
`q + S phi - Lz phi`. Signed effective sources are preserved.

For the explicit plane/column iteration:

```python
comparison = solver.solve(source, coupling="picard", relaxation=0.5,
                          maxiter=1200, tol=1e-9)
```

This alternates existing within-group planar solves with axial diffusion
solves and relaxes the shared scalar state. Its flux and leakage update
histories are recorded. Thin layers can converge slowly; Krylov is the default.
`maxiter` bounds actual Krylov iterations or Picard updates, and exhaustion or
a failed planar scattering solve returns `converged=False` with a reason.

## Fields and diagnostics

`flux`, `radial_flux`, `axial_flux`, `radial_leakage`, `axial_leakage`, and
`balance_defect` use `(groups, nr, nc, 2, nz)`. Radial currents use
`(groups, 3, nr, nc, 2, nz)`, outward from each triangle on its three ordered
edges. Axial currents use `(groups, nr, nc, 2, nz+1)`, positive in increasing z.
The lower exterior outward current is therefore the negative of the stored
lower z-current. All inactive entries are zero.

Success requires the requested tolerance for the planar flux fixed-point
residual, the cell balance scaled by absolute reaction/source/leakage terms,
and the disagreement between radial and axial scalar flux. Solver histories
alone do not establish convergence. The result also records sweeps, axial
solve applications, negative effective-source counts, and elapsed solve time.

## Verification and recorded comparisons

The new verification tests cover independent dense angular block equations
with two groups, upscatter, material jumps, masked cells, and mixed axial
boundaries; planar LU/levels and step/SCB conservation; signed-source
linearity and buffer ownership; analytic axial convergence; exact 2D and 1D
limits; continuous manufactured-solution convergence; 1–64-layer stability;
seven-group C5G7 tables; blends; the existing HP-MR builder with its active mask
and polar mixtures; and explicit nonconvergence reporting.

Reproduce them with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m pytest -q \
  tests/verification/test_axial_diffusion.py tests/verification/test_sn_2d1d.py
```

The [synthetic comparison](qa/sn-2d1d-synthetic-2026-10-28.json) and
[C5G7-data comparison](qa/sn-2d1d-c5g7-derived-2026-10-28.json) use the same
3 x 3 x 2 x 4 mesh and prescribed source for hybrid and full 3D methods.
Planar polar order 2 is matched to full-3D polar order 4; both use 8 azimuths.
The full 3D calculation reuses the original triangular S_N group solves and
conservative current folds.

| Derived problem | Radial scheme | Hybrid Krylov iterations | Flux L2 difference from same-mesh full 3D S_N |
|---|---|---:|---:|
| Synthetic two-group | Step | 35 | 3.34% |
| Synthetic two-group | SCB | 36 | 3.55% |
| C5G7 seven-group data | Step | 76 | 7.86% |
| C5G7 seven-group data | SCB | 80 | 8.45% |

All four hybrid solves have final balance/flux-consistency residuals below
2e-12, and all four full-3D comparisons converge. The flux differences include
the axial closure and discretization differences; spatial/angular uncertainty
has not been estimated. They are not published benchmark qualification or
performance claims. The deliberately short, heterogeneous domains provide a
visible check that this approximation is distinct from full 3D transport.

The HP-MR integration check uses the existing builder at `refine=1, nz=10`
with polar absorber mixtures and its placeholder two-group constants. With
a prescribed fast-group source in fuel cells, the CPU levels prototype
converges in 129 Krylov iterations at `tol=1e-8`; final balance and flux
disagreement are below 7e-10. This exercises 330 active radial cells and the
actual builder's mask/blend layout, without a new data-preparation workflow.

An eigenvalue trial on the same HP-MR geometry and placeholder constants,
with CPU levels, `n_polar=1`, and `n_azi=4`, gives `k_eff=0.9408521754` after
126 power updates and 10,023 inner iterations. It uses `tol_k=1e-7`,
`tol_source=1e-6`, `tol_inner=1e-8`, and `max_outer=150`. Integrated fission
production is one; the final balance residual is 4.04e-7 and radial/axial flux
disagreement is 9.10e-7. This is a geometry/integration check with placeholder
data, not a reactor-physics reference. Power iteration remains unaccelerated.

The eigenvalue verification adds 19 cases: independent dense angular
eigenproblems with heterogeneous spectra/upscatter, infinite-medium and axial
diffusion eigenvalues, agreement with the original 2D S_N eigen solver, both
coupling methods, normalization of all returned fields, blended spectra,
initial-amplitude invariance, nonfissile/invalid inputs, inner/outer exhaustion,
and rejection of a stagnant eigenvalue with a poor neutron balance.

The [two-group eigenvalue comparison](qa/sn-2d1d-eigenvalue-synthetic-2026-10-28.json)
and [seven-group eigenvalue comparison](qa/sn-2d1d-eigenvalue-c5g7-derived-2026-10-28.json)
reuse the short derived extrusion and matching polar nodes above. The synthetic
fuel has explicit `nu_sigma_f=[0.025, 0.08]`, `chi=[0.8, 0.2]`; the seven-group
case uses the bundled production/spectrum tables. Both methods' fluxes have
unit integrated production for comparison.

| Derived eigenvalue problem | Scheme | Hybrid k_eff | Full 3D S_N k_eff | Power updates |
|---|---|---:|---:|---:|
| Synthetic two-group | Step | 0.064691563 | 0.065911040 | 40 |
| Synthetic two-group | SCB | 0.060765909 | 0.062794576 | 37 |
| C5G7 seven-group data | Step | 0.022388462 | 0.023628381 | 37 |
| C5G7 seven-group data | SCB | 0.020719841 | 0.022242494 | 29 |

All four hybrid eigenvalue calculations converge with `tol_k=1e-9`,
`tol_source=1e-8`, and `tol_inner=1e-10`, and all four full-3D references
converge. These small, leaky domains are deliberately far subcritical and
retain closure/discretization differences; they establish a runnable
comparison rather than official reactor benchmark accuracy. Spatial/angular
uncertainty remains unestimated.

CMFD, official benchmark builders, GPU batching, and model-level dispatch
remain subsequent milestones.
