# 2D/1D transport in GARNET: MOC and planar S_N feasibility

Date: 2026-10-26. Scope: a separate architecture and literature investigation;
the production solvers are unchanged. The accompanying
[inventory](qa/moc-feasibility-2026-10-26.json) records source hashes and CPU
allocation probes. No GPU timing or MOC solution was generated in this study.

Follow-on planning (28 October 2026): the
[planar S_N / axial diffusion milestones](sn_2d1d_implementation_plan.md) turn
the S_N-first recommendation into implementation and verification steps. The
[benchmark shortlist](sn_2d1d_benchmarks.md) identifies supplied multigroup data
and the remaining geometry/reference work. The first four steps now have a
[verified CPU prototype](sn_2d1d_prototype.md), now including fixed-source and
eigenvalue iteration. Shared CMFD, GPU batching, and official benchmark
qualification remain subsequent work.

## Recommendation

**A 2D/1D transport option makes sense for GARNET's extruded reactor models.
Prototype it with the existing planar S_N solver first, then evaluate MOC as
an alternative planar engine.** This separates the value of dimensional
decomposition from the cost and benefit of introducing ray tracing.

MOC has a strong case for heterogeneous assembly reference calculations,
homogenization, and detailed absorber geometry. A replacement of the present
homogenized diffusion/SPN transient solver is a different proposition: transport
adds angular work and state, so this study supplies no basis for expecting it
to accelerate that workflow.

For favourable 3D transport workloads, **roughly 2–4x end-to-end acceleration
is a useful conditional prototype target**, based on the cost scenarios below.
It is not a measured result or a predicted range with known confidence. A
same-mesh planar MOC implementation could be slower than S_N. Gains above 10x
would need substantial reductions in error-matched spatial/angular work,
iteration count, or current implementation overhead. They do not follow from
the name “2D/1D.”

## What the code actually provides

| Component | Observed implementation | Consequence for the study |
|---|---|---|
| [Cartesian S_N](../garnet/sn.py) | 2D diamond-difference wavefronts; NumPy/CuPy; device DSA/GMRES; CMFD | A reusable planar transport engine exists. The steady sweep keeps edge state rather than a full cell angular field. |
| [Triangular/prismatic S_N](../garnet/tri_sn.py) | 2D and full 3D; first-order step or second-order simple corner balance (SCB); CPU LU or a GPU levels engine | The fair baseline is the accelerated levels engine, including SCB where its accuracy permits coarser meshes. |
| Levels engine | Topological schedules, persistent buffers, CUDA graphs, fused moment reductions and SCB block matvecs | MOC would replace an already optimized sweep structure, not an entirely serial reference. |
| S_N global iteration | DSA within groups; CMFD by default; energy Gauss–Seidel | Convergence acceleration is already present. Adding CMFD to MOC is not an independent speedup over this baseline. |
| Prism mesh | `TriGrid(shape=(nr,nc,2,nz))`, uniform axial spacing | A natural starting geometry for plane/column decomposition. |
| Transport data | Scalar total, absorption, fission, and group transfer; `total` falls back to `1/(3D)` | Enough for an initial isotropic comparison, not a complete lattice-physics data and resonance treatment system. |
| Transport transients | Cartesian angular source supported; triangular/prismatic `t_*` adapter requires CPU LU | A GPU 2D/1D transient solver would require additional development in either planar method. |

The introductory docstring in `tri_sn.py` still says the CuPy path was
untested. Later code and the recorded
[T4 transport measurements](gpu_kernel_optimization_plan.md#third-measurement--t4-the-transport-path-2026-07-30)
establish a more advanced baseline: graph replay and SCB contraction fusion
were exercised on GPU. That report records 1.42–2.32x gains from capture and
1.23–1.39x from the SCB contraction changes in its tested configurations.
Those are historical GARNET optimization measurements, not MOC forecasts.

Several costs remain visible in the current source:

- `_run_levels` performs multiple gather, multiply, reduce and scatter kernels
  per dependency level. Graphs reduce dispatch overhead but retain dependency
  barriers, irregular accesses and intermediate memory traffic.
- `_lv_tables` caches nine float64 coefficients per SCB cell/ordinate/group
  for the local inverse, besides connectivity and scratch buffers.
- `_solve_group_dev` keeps inner sweep/DSA arrays on device but returns scalar
  flux to the host per group. The fission/energy orchestration uses host arrays.
- `_sweep_currents_3d` and `_sweep_currents_scb_3d` copy the angular field to the
  host to fold currents. `_cmfd_matrix_3d` assembles on the host. The MG option
  moves most solves onto the device but still builds its hierarchy on the host
  and uses a host solve on the smallest level. The LU option bridges the bus
  on every application.

These are opportunities for S_N as well as for a new MOC implementation. A
comparison must credit device current reductions or CMFD refactoring to the
implementation change that actually supplies them.

## Literature: what is established, including planar S_N

The familiar MPACT configuration uses fine radial MOC, a lower-order axial
solution, transverse leakage coupling, and a global CMFD solve. It exploits
relatively smooth axial variation. This decomposition is separate from the
choice of the radial spatial discretization.
[MPACT 4.4 theory manual](https://info.ornl.gov/sites/publications/Files/Pub202495.pdf).

There is a direct precedent for **planar S_N in a 2D/1D formulation**:

- **Kong et al. (2023), “Convergence study of traditional 2D/1D coupling method
  for k-eigenvalue neutron transport problems with Fourier analysis,” Nuclear
  Engineering and Technology 55, 1350–1364.** Section 2.1 uses planar S_N in
  place of MOC for the Fourier analysis; the radial and axial equations are
  coupled through transverse leakage and CMFD. Section 6 returns to radial
  MOC for reactor benchmark tests. This is evidence of an analysed S_N/S_N
  formulation, with a narrower claim than a production all-S_N reactor code.
  Its stability results also show why choosing an axial S_N solver alone
  does not settle convergence: discretization and the scalar-moment update
  matter. [Published paper](https://doi.org/10.1016/j.net.2022.12.034),
  [author-uploaded full text](https://www.researchgate.net/publication/366661334_Convergence_study_of_traditional_2D1D_coupling_method_for_k-eigenvalue_neutron_transport_problems_with_Fourier_analysis).
- **Jarrett et al. (2019), “Polar Parity for Efficient Evaluation of Anisotropic
  Transverse Leakage in the 2D/1D Transport Method.”** This MPACT implementation
  uses fine-mesh 2D MOC for the even polar-parity component and coarse-mesh
  **2D S_N** for the odd component. It is a practical planar-S_N component
  within 2D/1D, although MOC still supplies part of the planar solution.
  [ANS abstract](https://www.ans.org/pubs/journals/nse/article-47978/),
  [DOI](https://doi.org/10.1080/00295639.2019.1627176).
- **Kelley and Larsen (2015), “A consistent 2D/1D approximation to the 3D neutron
  transport equation.”** This provides a mathematical foundation for radial
  transport with axial diffusion, including iteration stability and relaxation.
  It supports treating 2D/1D as an approximation to an equation, independent
  of a particular ray-tracing implementation.
  [Paper](https://doi.org/10.1016/j.nucengdes.2015.07.026).

In contrast, **Hursin et al. (2014)** implement an **axial** NEM-S_N solver in
DeCART, retaining radial MOC. This is relevant to the axial-engine choice,
but should not be cited as a full planar-S_N precedent.
[ANS paper summary](https://wx1.ans.org/pubs/journals/nse/a_35563).

This search did not establish a widely deployed fine-mesh planar-S_N-only
2D/1D reactor implementation. That is a limit of the evidence found, not a
claim that none exists. GARNET's possible contribution would be its GPU
implementation, triangular/hexagonal geometry, coupling and demonstrated
accuracy/cost; the general idea should not be presented as new.

For GPU feasibility specifically, nTRACER is a strong precedent. Its 2021
paper describes planar MOC, augmented axial MOC and CMFD on GPU, with changes
to data layout and mixed precision.
[Choi et al. (2021)](https://inl.elsevierpure.com/en/publications/practical-acceleration-of-direct-whole-core-calculation-employing/).
An earlier primary report gives useful, explicitly bounded performance evidence:

| nTRACER APR1400 case, 2018 | MOC component CPU → GPU | Whole calculation CPU → GPU |
|---|---:|---:|
| 2D quarter core | 1218.4 → 95.1 s, 12.8x | 1506.8 → 135.4 s, 11.1x |
| Half-3D quarter core, 16 planes | 1561.7 → 118.0 s, 13.2x | 2092.9 → 252.6 s, 8.3x |

The reference CPU nodes used dual E5-2640 v3 processors; the GPU cluster used
GTX 1080 GPUs with E5-2630 v4 hosts. The 2D case used one CPU node versus one
GPU, and the half-3D case used 16 CPU nodes versus 16 GPUs. These compare CPU
and GPU implementations of the same MOC-based code, not GPU MOC against GPU
S_N. CMFD and axial work already limited total acceleration.
[Choi, Kang and Joo, KNS Autumn 2018, Tables 1–3](https://www.kns.org/files/pre_paper/40/18A-125%EC%B5%9C%EB%82%A8%EC%9E%AC.pdf).

## What 2D/1D changes mathematically

With GARNET's angular weights normalized to sum to one, an isotropic-source
plane equation can be written schematically as

```text
Omega_xy · grad_xy(psi_k) + Sigma_t psi_k = q_k - Lz_k
Lz_k = (Jz_top - Jz_bottom) / dz_k
```

Here scalar `Lz` stands for an isotropic approximation to the angular axial
leakage source. An anisotropic version carries its angular moments instead.
The axial solver supplies this leakage; radial currents supply the transverse
leakage for the axial columns. A global coarse solve accelerates the coupled
iteration and enforces a shared neutron balance.

For GARNET, start with the existing triangle as the radial coupling region,
avoiding additional pin/assembly homogenization initially. Later introduce a
separate coarse-region map, flux-weighted axial constants and conservative
restriction/prolongation. Preserve B4C and adjacent reflector distinctions;
one assembly-average axial column can conceal the streaming contrast of
interest. Finer coupling regions increase memory and axial work, so this
choice belongs in the accuracy/cost study.

**Independent plane solves do not remove the number of planes from the work.**
All planes still need radial transport updates. They remove interplane
streaming dependencies from a fine transport sweep, replacing them with axial
and leakage iterations. Coarser axial planes are an additional approximation,
not an automatic consequence of 2D/1D.

Likewise, 2D S_N plus exact, fully angle-resolved axial streaming on the same
fine cells can be an iterative factorization of a 3D discrete transport system.
It becomes the usual approximate 2D/1D method when leakage angular/spatial
structure or axial physics is reduced. Keep those two objectives explicit:
accelerating the same discrete problem and solving a cheaper approximation
need different accuracy claims.

Angular comparison requires care: `quadrature_2d(p,a)` uses the positive nodes
of a **2p-point** polar rule with mirrored weights; `quadrature_3d(p,a)` uses
the full **p-point** rule. Equal API values are not matching polar resolution.
For an isotropic leakage source, folding the two axial hemispheres can halve
radial angle work at matching absolute polar nodes. Anisotropic leakage can
remove that symmetry; the parity literature is relevant here.

## Why MOC could help, and its GPU costs

For a constant source and cross section on one characteristic segment:

```text
ell = ell_xy / sin(theta); tau = Sigma_t * ell
psi_out = psi_in * exp(-tau) + (q / Sigma_t) * (1 - exp(-tau))
psi_average = q / Sigma_t
              + (psi_in - q / Sigma_t) * (1 - exp(-tau)) / tau
```

Evaluate the small-optical-thickness expressions with `expm1`/series and
explicit void limits. Geometry and angular weights turn segment integrals
into region fluxes and surface currents. Track-based region areas must be
consistent with source and rate normalization. Exact attenuation through a
constant segment does not eliminate source-discretization, geometry or angle
error. MOC also uses discrete angles; it retains angular ray effects.
[OpenMOC method and sweep documentation](https://mit-crpg.github.io/OpenMOC/methods/eigenvalue_calculations.html).

A suitable GARNET GPU design would use:

- A Python/NumPy reference engine and a compiled CuPy CUDA kernel for the
  segment recurrence. A Python loop with one CuPy operation per segment would
  recreate the launch problem on a much larger iteration space.
- Flattened track offsets, region IDs, segment lengths and endpoint links.
  Share geometry across planes with identical partitioning; material/source
  data remain plane-dependent. Bucket tracks by length to limit divergence.
- Parallel tracks, planes, polar angles and frozen-source group batches.
  A warp can cooperate across groups/polar angles for coalesced loads. GARNET's
  2/7/11-group workloads need enough independent tracks/planes too; a layout
  designed only for dozens of energy groups can waste lanes. Concurrent
  groups require a deliberate lagged or block iteration instead of assuming
  the existing energy Gauss–Seidel ordering is preserved.
- In-register angular recurrence, device flux/current tallies and device
  coarse coupling. Multiple tracks contribute to one region, requiring
  atomics or staged reductions. These have contention and reproducibility
  costs absent from a cell's uniquely owned S_N update.
- On-demand attenuation evaluation initially. A cached attenuation array
  over segments × polar angles × groups × planes can erase memory savings,
  particularly with feedback-dependent cross sections.
- Float64 validation first, followed by measured mixed precision. Reductions,
  leakage differences and eigenvalue convergence deserve particular care.

Independent tracks are available within a frozen-source vacuum sweep. Reflective,
periodic or domain-connected endpoints introduce boundary dependencies and
iteration. This matters for GARNET's periodic assembly reference problems:
the current levels S_N engine rejects periodic boundaries, whereas CPU LU
supports them. MOC could offer useful GPU capability there, but the boundary
algorithm must be implemented and included in the comparison.

For steady calculations, MOC can accumulate scalar flux and currents while
retaining angular flux only at track boundaries and in the active recurrence.
That avoids the volume-wide angular field used by GARNET's triangular levels
engine. It does not make memory negligible: track metadata, endpoint state and
reduction buffers must fit too. The Cartesian steady S_N engine already avoids
a full cell angular field, so that particular saving is baseline-dependent.

Triangle-based tracking is sufficient for a first MOC test. It preserves the
present raster/material-mixing approximation. To obtain MOC's geometric
advantage for circular pins, shells and rotating annular sectors, introduce
surface-conforming regions and tracking. That is a second work package.
Changing drum material fractions on a fixed triangle partition can reuse tracks;
moving actual curved region boundaries generally requires retracking or a
validated modular update.

## Measured layout evidence and conditional speedup

The CPU inventory constructs the current levels tables and performs one finite,
positive fixed-source sweep for each scheme and mesh below. All cases use
HP-MR `refine=2`, `n_polar=2`, `n_azi=8`, 16 ordinates, polar absorber mixing,
and two-group placeholder data. The 11-group column extrapolates the cached
group-table layout; it is not an 11-group physics calculation.

| Axial layers | Active cells | Sequential levels | Step selected payload at 11 groups | SCB selected payload at 11 groups |
|---|---:|---:|---:|---:|
| 2D | 1,320 | 64 | 3.89 MiB | 21.57 MiB |
| 10 | 13,200 | 73 | 46.91 MiB | 239.85 MiB |
| 20 | 26,400 | 83 | 93.81 MiB | 479.69 MiB |

These are logical payload estimates for the fused CUDA sweep's buffers, level
tables and cached group tables, counted through the matching CPU layouts.
They exclude DSA/CMFD, cross sections, sources, graph objects, allocator
overhead, host setup arrays and transient angular history. They are neither
GPU peak-memory measurements nor whole-solver totals.

At 20 layers, one SCB group's local inverses occupy 29.00 MiB; eleven groups
would use 319.04 MiB. One angular field is 9.67 MiB, copied to the host by a
current-accumulating SCB sweep. Eliminating those inverse tables and reducing
currents on device are concrete opportunities. S_N can also address them by
computing smaller local representations, changing cache policy, or fusing
the sweep and current reduction.

The level count grows from 64 to 83, **not from 64 to 1,280**, on this 20-layer
extrusion: the 3D engine already batches planes in its wavefront. This rules
out predicting a 20x gain merely from processing planes independently. A
batched 2D/1D S_N engine can share radial schedules across planes and retain
their larger batches; twenty Python-driven solver instances would not provide
that implementation.

For a steady solve, a useful work model is

```text
T_SN    = I_SN * (W_SN + DSA + current_fold + CMFD + other)
T_2D1D  = setup + I_2D1D * (W_planar + axial + coupling + CMFD + other)

W_SN roughly scales as G * M_3D * cells_xy * Nz * cell_update_cost
W_MOC roughly scales as G * P * directed_segments_xy * Nz * segment_update_cost
```

Track spacing can make the number of MOC segment updates exceed the S_N
cell/angle updates. Segments also need attenuation and competing tallies.
The advantages must come from better update efficiency, smaller angular
state/tables, coarser error-matched geometry, or fewer global iterations.

For illustration, let `f` be the baseline fraction replaced by the faster
sweep, `r` its speedup, `c` the added axial/coupling cost as a fraction of
baseline time, and `kappa` the ratio of new to old outer iterations. Ignoring
one-time setup:

```text
overall_speedup = 1 / [kappa * (1 - f + f/r + c)]
```

| Assumed f | Assumed sweep gain r | Added c | Iteration ratio | Overall gain |
|---:|---:|---:|---:|---:|
| 0.80 | 3x | 0.05 | 1 | 1.94x |
| 0.90 | 5x | 0.05 | 1 | 3.03x |
| 0.90 | 5x | 0.05 | 2 | 1.52x |
| 0.50 | 5x | 0.05 | 1 | 1.54x |
| 0.90 | 10x | 0.05 | 1 | 4.17x |

These are sensitivity calculations with assumed inputs, not fitted performance
estimates. They explain the conditional 2–4x target and why faster sweeps alone
can yield a modest overall gain. Small meshes, coarse few-group homogenized
regions, many track crossings, weak axial convergence or dominant CMFD setup
can make a new MOC path break even or lose. More detailed radial geometry and
many repeated solves make setup amortization and error-matched coarsening more
favourable. No existing evidence here establishes MOC's advantage over an
equally optimized 2D/1D S_N engine.

## Accuracy and transient limits specific to GARNET

HP-MR's repeated radial geometry and full-height drums favour plane reuse.
Its axial fuel/reflector interfaces, vacuum end faces and strong absorber
contrast make isotropic leakage and coarse axial homogenization hypotheses
to test. Agreement on k alone is insufficient: compare drum reactivity worth,
absorber reaction rates, axial power, local power and directional leakage.

Negative effective sources can occur when leakage is subtracted from a small
physical source. Do not repair them by clipping without checking conservation.
Use a documented leakage treatment, monitor its activation, and measure its
effect on the transport solution. nTRACER's augmented axial MOC uses subgrids,
leakage splitting and scattering/transport-correction treatments to improve
stability. Such measures are part of the solver, rather than optional GPU
optimizations.
[Choi and Joo (2020)](https://doi.org/10.1016/j.anucene.2020.107440).

Axial refinement alone does not force a conventional 2D/1D approximation to
converge to full 3D transport when angular leakage or homogenization closures
remain fixed. These are distinct errors from spatial/angular discretization.
[Jarrett et al., M&C 2017](https://www.kns.org/files/int_paper/paper/MC2017_2017_1/P005S01-01JarrettM.pdf).

GARNET's materials and SPH factors also need interpretation. MOC on the same
homogenized constants solves a different transport discretization of that
effective problem; it cannot reconstruct removed pin physics. Diffusion-calibrated
SPH factors are not automatically transferable to MOC. Use raw transport data
for method comparisons or regenerate equivalence corrections consistently.
The existing pin-resolved assembly builder supplies geometry information but
does not supply a generic surface-based MOC tracker or a resonance library.

For a true time-dependent transport method, backward Euler includes
`psi_old/(v*dt)` as an angular source. Scalar flux and track endpoint values
alone do not generally preserve the old angular distribution inside each
segment. A transient MOC design needs angular history, reconstruction, or an
explicit reduced model, with corresponding memory and error accounting.
The steady MOC memory advantage cannot simply be assumed for transients.
The current triangular adapter explicitly rejects the GPU levels engine for
this per-ordinate source; extending that engine is an alternative development
path. A quasi-static 2D/1D shape solver may be a more economical first connection
to GARNET's existing transient workflows.

## Concrete development and comparison sequence

1. **Establish the transport baseline.** Profile levels S_N step and SCB on the
   intended GPU, with graphs and DSA/CMFD enabled. Record setup, sweeps, DSA,
   current folding/transfers, CMFD setup/solve and peak memory. Use the existing
   component timers but add separate GPU-event timing where needed. Measure
   a version with device current reduction before attributing that saving to MOC.
2. **Build common 2D/1D coupling with planar S_N.** Keep fine radial regions,
   start with isotropic leakage and an axial diffusion reference, then compare
   SP3 and a transport axial option. Batch planes within a shared schedule,
   update leakage and restrict/prolong on device, and give the axial solver
   ownership of axial currents in the shared CMFD balance. Avoid double
   counting them with a separate axial leakage correction. Include relaxation
   and fine-axial-mesh stability probes from the start.
3. **Add a standalone 2D MOC engine.** Track existing triangles first, vacuum
   boundaries first, with region and surface tally conservation. Compare
   fixed-source attenuation, uniform-medium limits, mesh/angle/track refinement
   and k/power with existing S_N. Add periodic endpoint coupling for assembly
   reference work. Keep a shared plane-solver interface returning region flux,
   interface currents, convergence/work diagnostics and optional leakage moments.
4. **Compare planar engines inside identical coupling.** Full 3D S_N versus
   2D/1D S_N isolates decomposition; 2D/1D S_N versus 2D/1D MOC isolates the
   radial engine. Use equal physical data and matched errors, then report both
   identical-mesh and each-method-refined results. Coarse CMFD agglomeration
   and curved-region geometry should be separate ablations.
5. **Extend only where the evidence supports it.** Add true pin/absorber
   surfaces, anisotropic leakage and optional axial subgrids as accuracy
   requires. Connect a verified shape solver to quasi-static kinetics before
   attempting full transient angular-history support.

Suggested decision cases are a homogeneous extruded problem, a heterogeneous
assembly, 3D C5G7 unrodded/rodded configurations, and HP-MR with drums in/out
and axial mesh refinement. HP-MR placeholder data are useful for implementation
checks; use consistent physical cross sections and an independently converged
reference for physics conclusions. Proposed prototype gates are at least 2x
transport-solve acceleration at the required error, or a material reduction
in memory that enables a previously infeasible case. Set k, worth and local
power tolerances from the application before timing the candidates.

Reproduce the six allocation/sweep probes and cost scenarios with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_moc_feasibility
```

The result is a justified path to a GPU 2D/1D option, with planar S_N as the
lower-cost first experiment and MOC as a candidate for improved heterogeneous
transport capability. Choosing between them requires the controlled comparison
above; published CPU-to-GPU MOC factors do not decide that comparison.
