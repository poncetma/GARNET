# Planar S_N with axial diffusion: implementation milestones

Date: 28 October 2026. Status: **M0–M3 implemented and verified on CPU;
M4 eigenvalue iteration is implemented; CMFD and M5–M7 remain open**.
The [prototype guide](sn_2d1d_prototype.md) gives usage,
test evidence, and reproducible full-3D comparison results.
This develops the S_N-first recommendation in the
[feasibility study](moc_2d1d_feasibility_2026-10-26.md). The
[benchmark shortlist](sn_2d1d_benchmarks.md) records the available cross sections,
reference solutions, and geometry work needed for each candidate.

The first deliverable is a verified steady multigroup solver on extruded
triangular meshes: planar S_N supplies radial transport and axial diffusion
supplies axial currents. Build the CPU reference before batching the same
operations on GPU. Use one axial column per radial triangle initially, so no
assembly homogenization or new cross-section preparation is required.

## Numerical contract

Use isotropic scattering and isotropic axial leakage, with the existing
angular weights normalized to one. For radial cell `r`, layer `k`, and group
`g`, the intended equations are

```text
Omega_xy . grad_xy(psi[g,k]) + Sigma_t[g,k] psi[g,k]
    = scattering[g,k] + fission[g,k] / k_eff + external_source[g,k]
      - axial_leakage[g,k]

phi_radial[g,k] = sum_m w[m] psi[g,k,m]
axial_leakage[g,k] = (Jz[g,k+1/2] - Jz[g,k-1/2]) / dz[k]

axial_leakage[g,k] + Sigma_removal[g,k] phi_axial[g,k]
    = off_group_scattering[g,k] + fission[g,k] / k_eff
      + external_source[g,k] - radial_leakage[g,k]

radial_leakage[g,k] = sum_radial_faces(outward_current * face_length) / cell_area
```

The scattering source includes within-group scattering in the planar
equation; the axial equation moves it into `Sigma_removal`. Assemble transfer
sources with GARNET's `[source_group, target_group]` convention. Fission means
`chi[g] * sum_h(nu_sigma_f[h] * phi[h])`. Omit it for a nonmultiplying fixed
source; hold it fixed during a fission-source iteration.

For the initial cell-centred axial discretization, use a single shared face
current, positive in increasing z:

```text
Jz[k+1/2] = -(phi[k+1] - phi[k])
             / (dz[k] / (2 D[k]) + dz[k+1] / (2 D[k+1]))
```

Use `D = 1 / (3 Sigma_transport)` from the selected transport-corrected
benchmark constants. Reflective axial faces have zero current. At an outer
vacuum face, Marshak diffusion gives outward current
`phi_cell / (2 + dz / (2 D))`; the lower face needs the corresponding signed
z-current. Transport vacuum means zero incoming angular flux in a plane.
Zero scalar flux at the cell centre is a different boundary condition.

The two scalar flux estimates must agree at convergence. Their cell balances
use the same radial face currents and axial face currents; a scalar-flux
rescaling alone does not establish that agreement. This design follows the
shared-balance construction described in the
[MPACT theory manual, Chapter 3](https://info.ornl.gov/sites/publications/files/Pub69725.pdf).
Iteration stability and axial-diffusion model accuracy are separate questions;
the [Kelley–Larsen study](https://doi.org/10.1016/j.nucengdes.2015.07.026)
motivates testing both.

## Existing code to reuse, and gaps to address

| Area | Existing implementation | Required seam or change |
|---|---|---|
| Planar transport | `TriSNTransportSolver`, step and SCB, LU and levels engines | A fixed-source plane interface with scalar flux, conservative radial face currents, and explicit convergence diagnostics |
| Geometry | `TriGrid(shape=(nr,nc,2,nz))`, uniform `dz` | Shared radial connectivity across layers; validate the extrusion and active mask |
| Axial diffusion | Axial terms in `TriGroupOperator` | A separate batched column operator and solve, without radial diffusion terms |
| Within-group acceleration | Planar DSA, GMRES variants | Preserve plane solver behaviour; do not mistake DSA convergence for coupled convergence |
| Global acceleration | Triangular/prismatic CMFD assembly and power iteration | Radial currents from S_N, axial currents from diffusion, one common balance |
| Device execution | Persistent levels buffers and CUDA graph replay | Batch independent layers; reduce currents on device; remove per-group flux round trips |
| Public results | `SNResult`, model solver dispatch | An opt-in method with flux shape consistent with the prism grid and additional coupling diagnostics |

The triangular transport solver currently accepts vacuum or periodic radial
boundaries, with **vacuum only for the levels engine**. It has no radial
reflective boundary law. The new axial diffusion operator can support
reflective ends independently. Begin published triangular benchmarks on the
full radial domain rather than silently substituting a boundary law.

For equal absolute polar nodes, compare planar `n_polar=p` with full 3D
`n_polar=2*p`, using the same azimuthal rule. Equal parameter values currently
select different polar nodes in `quadrature_2d` and `quadrature_3d`.

## Milestones

Each milestone should be a separately reviewable change. Names below are
proposed module boundaries, not existing APIs.

| ID | Deliverable | Depends on | Completion evidence |
|---|---|---|---|
| M0 | Freeze equations, data, fixtures, and baseline protocol | — | Reviewed conventions; reproducible small references; no external XS workflow |
| M1 | Fixed-source planar engine interface | M0 | Existing 2D solutions and currents reproduced with signed sources |
| M2 | Conservative axial diffusion columns | M0 | Independent matrix/slab comparisons and exact face cancellation |
| M3 | Coupled fixed-source CPU prototype | M1, M2 | Dense coupled-equation agreement, dimensional limits, and thin-layer stability |
| M4 | Multigroup eigenvalue solve (implemented) and shared CMFD (open) | M3 | Accelerated/unaccelerated agreement, balance, and convergence diagnostics |
| M5 | Published benchmark builders and accuracy study | M0; solves need M4 | Takeda-4 qualification; C5G7 follow-on with explicit geometry/reference dependencies |
| M6 | Batched device implementation and cost study | M4; comparisons use M5 | CPU/GPU agreement, device-resident current folding, matched-error timings |
| M7 | Public integration and supported-use decision | M5, M6 | Documented API, reproducible results, stated accuracy envelope |

M1 and M2 can be developed independently. Benchmark transcription and
geometry checks in M5 can also start before the coupled solver exists.
CPU verification does not wait for GPU access.

### M0 — Equations, fixtures, and measurement protocol

- [x] Record source units, angular normalization, array axes, face orientation,
  removal/scattering definitions, and boundary conditions in the method docs.
- [x] Require a common radial mesh and an active footprint constant in z.
  Materials may change by layer. Reject unsupported changing footprints.
  Start with uniform axial layers, matching `TriGrid`; retain explicit layer
  widths in the column algebra without changing the mesh API yet.
- [x] Make small one- and two-group fixtures: uniform medium, two axial
  materials, and heterogeneous radial regions. Use explicit numerical constants
  and manufactured sources, plus selected existing C5G7 material tables.
- [x] Establish full 3D step/SCB reference runs on the same triangular mesh.
  Record the quadrature nodes, mesh, cross sections, tolerances, iteration
  counts, and neutron-balance residual, not just `k_eff`.
- [x] Define a benchmark manifest with source URL, report/table/page, source
  checksum, group order, scattering orientation, transport correction,
  geometry variant, normalization, and reference uncertainty.

No cross-section generation, condensation, resonance processing, SPH fitting,
or ADF library generation belongs in these fixtures. Formatting published
tables into checked-in arrays and deriving diffusion coefficients algebraically
are sufficient.

### M1 — Planar fixed-source contract

- [x] Add a private adapter around the existing triangular transport kernels.
  It should take a group, prescribed non-self-scattering source, initial flux,
  and solve tolerance, and return flux, radial face currents, convergence,
  residual, and sweep count. Keep fission eigenvalue control outside it.
- [x] Make it explicit whether a source includes within-group scattering.
  Fold final currents from a sweep of the complete converged source. Flux
  and currents must refer to the same iterate.
- [x] Preserve step and SCB face-flux definitions. Share one interior face
  current between its two cells, with opposite outward signs.
- [x] Accept signed effective sources: subtracting axial leakage can produce
  them. Do not apply positivity clipping inside the adapter. Report negative
  effective sources separately from negative physical flux.
- [x] Specify buffer ownership: persistent scratch may be reused, but returned
  state needed by another plane must remain valid after the next sweep.

Gate: reproduce the original plane solver for positive-source problems to
its requested tolerance, match LU/levels currents, verify signed-source
linearity, and satisfy each cell's streaming/collision balance. Existing
transport tests remain the regression baseline.

### M2 — Axial diffusion operator

- [x] Implement `axial_diffusion.py` with face conductances, lower/upper
  boundary terms, current reconstruction, leakage divergence, and batched
  tridiagonal column solves. Keep geometry/coefficients separate from RHS work.
- [x] Support vacuum and independently reflective lower/upper faces. Add axial
  periodicity for algebraic verification if needed; a cyclic solve is distinct
  from an ordinary tridiagonal solve.
- [x] Treat material jumps through the two half-cell resistances above.
  Use one current per face, including when reconstructing diagnostics.
- [x] Solve one group at a time with frozen off-group/fission sources at first.
  Cache coefficients while material data stay unchanged; expose invalidation.

Gate: compare with an independently assembled dense matrix on short columns,
including unequal test layer widths and material jumps. Verify reflective
flat-flux limits, a nontrivial analytic diffusion slab, and cancellation of
all internal face contributions in the volume-integrated balance. Unequal
width algebra tests do not imply public nonuniform-mesh support.

### M3 — Coupled fixed-source CPU reference

- [x] Implement `sn_2d1d.py` as the coupling driver. Start with one axial
  column per triangle and no radial restriction/prolongation.
- [x] For frozen external/off-group/fission sources: solve the planes with
  current axial leakage, reconstruct radial leakage, solve the columns with
  that radial leakage, and update shared currents/fluxes. Document update
  ordering and which quantities are lagged.
- [x] Apply controlled under-relaxation to a consistent coupling state. Track
  flux change, current/leakage change, radial/axial flux disagreement, and the
  unmodified coupled-equation residual. Return failure when limits are hit.
- [x] Test shrinking `dz` from the outset: hold the physical height fixed and
  use 1, 2, 4, 8, 16, 32, and 64 layers on small smooth fixtures; use
  interface-aligned refinements for layered cases. A failed Picard iteration should
  lead to a better iteration/preconditioner, not altered physics. Use a
  matrix-free Krylov reference for the linear fixed-source coupled system if
  under-relaxation is inadequate; retain it as a verification option.
- [x] If negative sources require a source-splitting scheme, derive an
  algebraically equivalent formulation and test its fixed point independently.
  Never silently clip leakage or overwrite cross sections to obtain convergence.

Gates:

1. A tiny independently assembled planar-S_N/axial-diffusion block system
   agrees with the iterative solution; this reference must not call the
   production operator to construct its matrix.
2. Reflective axial ends and z-invariant data recover the original 2D solution.
3. A radially uniform periodic CPU fixture reduces to the axial diffusion
   solution. This tests diffusion exactly, not agreement with axial transport.
4. A manufactured solution of the hybrid equations shows axial discretization
   convergence; sharp interfaces retain local and global balance.
5. Thin-layer and strong-absorber scans either converge to the same reference
   or return an explicit convergence failure with usable diagnostics.

Implementation evidence: `garnet/_sn_plane.py`, `garnet/axial_diffusion.py`,
`garnet/sn_2d1d.py`, and their verification tests. Signed sources work directly;
no source splitting or clipping was needed. Axial periodicity was unnecessary
for these checks and remains unsupported. Coefficients are frozen; reconstruct
the operator/solver after material changes. Existing directed S_N face-current
storage is retained, with opposite interior currents checked to round-off.

### M4 — Multigroup eigenvalue iteration and CMFD

- [x] Add fission power iteration with converged multigroup scattering on top
  of M3. Normalize flux and all current/source state by the same factor.
  Keep fixed-source and eigenvalue convergence controls distinct. The prototype
  reuses M3's implicit intergroup solve, including upscatter, rather than adding
  a separate energy Gauss–Seidel iteration.
- [ ] Construct a low-order 3D balance using lateral S_N currents and axial
  diffusion currents. Reuse existing CMFD face algebra where appropriate,
  but make current ownership an explicit input.
- [ ] Fit radial drift coefficients to the planar currents. Use the axial
  diffusion closure for axial faces; do not import currents from a full 3D
  transport sweep. In the CMFD equation axial leakage is already represented
  by its faces, so it must not also appear as an extra RHS leakage correction.
- [ ] Start CMFD on the same prism cells. Coarser radial regions and
  flux-weighted coefficient restriction are later optimizations, after a
  conservative mapping has its own tests.
- [ ] Expose coupled outer iterations, plane sweeps, axial solves, CMFD work,
  residual histories, and negative-source diagnostics in results.

Gate: plain power iteration and CMFD converge to the same hybrid solution.
An independent small eigenproblem checks multigroup transfer and fission
normalization. Include a case with actual upscatter. Check the final fine-cell
balance and radial/axial flux consistency rather than stopping on `k_eff` alone.

The eigenvalue portion is verified against independent angular eigenproblems,
the original 2D S_N solver, and infinite-medium/axial-diffusion limits. Results
include k/source/flux histories and total inner work. A loose-inner-tolerance
test rejects an apparently stationary k whose hybrid balance is inaccurate.
The shared-CMFD portion and its fixed-point/acceleration comparison remain open.

### M5 — Benchmark data, geometry, and accuracy

- [ ] Transcribe and check the four-group **Takeda Model 4** tables and full
  hexagonal layout. Build rods-out, half-inserted, and fully inserted cases;
  place axial material interfaces on mesh faces. Verify material volumes and
  rod patterns before solving. Full radial geometry avoids new reflection code.
- [ ] Compare 2D/1D against converged full 3D triangular S_N with identical
  material data, then against the published references. Report reactivity
  worth, region/group flux, production rates, axial leakage, and `k_eff`.
  If a benchmark uses an axial symmetry face, expand that symmetry in the
  full-3D reference or add a verified reflective law there; the existing
  triangular 3D transport solver cannot supply a reflective reference directly.
- [ ] Use small C5G7-derived extrusions for early seven-group checks, clearly
  labelled as derived test problems rather than official C5G7 configurations.
- [ ] Add the official **C5G7 3D Extension** as the next benchmark package.
  Reuse the eight material tables already in `_c5g7_data.py`. Its square
  quarter-core and radial reflection require a Cartesian planar adapter, or
  an explicitly converged full-domain triangular geometry approximation.
  Prefer the Cartesian adapter for faithful geometry and existing reflective
  boundary handling; reuse the common coupling and column code.
- [ ] Implement its axial reflector details and all three rod configurations
  from the specification. Obtain and check the complete external pin-rate
  reference arrays, or document a new independent multigroup reference, before
  claiming complete pin-power qualification.

Refine radial mesh, axial mesh, and angle resolution separately. An axial
diffusion closure can retain a full-transport bias under axial refinement;
report that bias rather than interpreting it as unfinished iteration.

### M6 — Batched CPU/GPU execution and measured cost

- [ ] Share radial schedules/connectivity across layers. Give material/source
  arrays a plane axis and persistent buffers; avoid one Python solver instance
  per plane in the production path.
- [ ] Implement batched column solves and conservative radial-current folding
  on the array backend/device. Keep material data, scalar flux, sources, face
  currents, and CMFD vectors resident through the coupled iteration. Small
  convergence reductions are distinct from copying full fields per group.
- [ ] Replace host-only CMFD face assembly in the production loop with a
  device operator/update path. Any retained host coarse solve must be small,
  explicit, and included in transfer and end-to-end timing.
- [ ] Preserve existing CUDA graph constraints and buffer lifetimes. Store
  angular work for a plane batch, not an unconditional second full 3D angular
  history. Report memory required by the selected batch size.
- [ ] Measure full 3D step and SCB before and after any shared current-reduction
  improvements. Credit common optimizations to both baselines.
- [ ] Record setup, planar sweeps, axial solves, DSA, current folds/transfers,
  CMFD setup/solve, other orchestration, total time, and peak allocated memory.
  Use device events for asynchronous GPU components and synchronized wall
  time for end-to-end comparisons.

Gate: CPU and GPU reproduce the same hybrid solution. Benchmark identical
meshes and matched physical errors, retaining setup costs and iteration counts.
The proposed performance gate is at least 2x transport-solve speedup at the
required error, or a demonstrated memory reduction that enables an otherwise
infeasible case. It is a decision criterion, not an expected result.

### M7 — Public integration and decision

- [ ] Add an opt-in low-level solver and model dispatch after its supported
  geometry/boundary combinations are qualified. Reject unsupported requests
  rather than falling back to another transport model.
- [ ] Document equations, approximation limits, default tolerances, diagnostics,
  and one self-contained example using checked-in cross sections.
- [ ] Keep small algebraic checks in the regular suite and put expensive
  published refinements/GPU studies in explicitly selected runs. Save manifests,
  reference data, result JSON, and reproduction commands.
- [ ] State which cases meet the application accuracy/cost goals and which need
  a better axial model. Published benchmark agreement and execution speed are
  separate qualification decisions.

## Proposed acceptance thresholds

These are initial engineering targets, to be frozen with the M0 fixtures;
they are not achieved results or published benchmark tolerances.

| Check | Proposed criterion |
|---|---|
| Small fixed-source algebraic reference | Relative flux error <= 1e-8 and scaled equation residual <= 1e-10 |
| Independent small eigenproblem | Absolute `k_eff` difference <= 1e-8; normalized flux error <= 1e-7 |
| Internal face conservation | Cancellation to float64 round-off, scaled by absolute face contributions |
| Converged CPU coupled solution | Scaled balance and radial/axial flux disagreement <= 1e-8 |
| CMFD fixed-point preservation | Within requested flux/current tolerances; absolute `k_eff` difference <= 1e-8 |
| CPU/GPU at the same discretization | Absolute `k_eff` difference <= 2e-8 and normalized flux error <= 1e-6 |
| Initial smooth reactor-case accuracy study | Aim for <= 100 pcm in reactivity and <= 1% RMS production/fission-rate error against a converged full 3D reference; report maximum local error separately |

Define reactivity error as `1e5 * (1/k_reference - 1/k_test)` pcm and rod worth
as `1e5 * (1/k_inserted - 1/k_withdrawn)` pcm. Use volume-integrated rates and
a common normalization. For residuals, scale by sums of absolute production,
reaction, source, and leakage terms so signed-source cancellation does not
hide defects. Use a documented absolute floor for near-zero flux/rate regions.

The 100 pcm/1% targets are a screening goal for selected smooth cases. They
are not hard pass/fail limits for void streaming or rod-tip stress tests, nor
a reason to tune cross sections. Published reference uncertainty and remaining
spatial/angular error must be smaller than the conclusion being claimed.

## Boundary of this implementation

MOC, axial SP3/S_N, coarse-column homogenization, curved-surface tracking,
anisotropic transverse leakage, changing axial footprints, multi-GPU execution,
and time-dependent angular history are follow-on work. A verified steady shape
solver can later connect to quasi-static kinetics. The benchmark suite should
establish when axial diffusion is sufficient before selecting those extensions.
