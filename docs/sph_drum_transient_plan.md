# Plan: drum-rotation transient, SPH-DF diffusion against quasi-static S_N

Status: in development since 2026-11-19 (steps 1–3 done, step 4 in part; see
Steps). Follows the SPH audit
([report](qa/sph_audit_2026-11-13.md) §10.6) and is step 6 of its §9.3
development plan.

## Purpose

The steady equivalence is done: on the 2-D HP-MR, SPH with boundary
discontinuity factors (SPH-DF), on drum-sector zones with 5° tables, reproduces
the anisotropic S_N reference.
- **At the table angles:** k and every zone rate exact.
- **Between table points:** k within 38 pcm and pin RMS within 0.4 %.

The model is meant for drum transients, and a transient adds four things that
the steady tests do not see:

1. **Tables interpolated in time.** The zone constants, SPH factors, boundary
   factors and form functions are interpolated in drum angle along θ(t).
2. **Kinetics of the corrected model.** SPH needs per-zone neutron speeds v/μ
   (`sph_kinetics`); the uncorrected 1/v biased the generation time by +20.7 %
   on a colorset.
3. **Boundary factors under a moving flux shape.** γ is a steady-state
   construct, so applying the tabulated γ(θ(t)) in a transient is an
   approximation.
4. **The spatial shape during motion.** Delayed neutrons and the finite drum
   speed make the shape lag the static shape.

The question is whether the tabulated SPH-DF diffusion model reproduces power,
reactivity and the assembly and pin power distribution through a drum
manoeuvre. The reference is S_N, without running a full S_N transient.

## Reference: quasi-static S_N

The reference treats the flux as an S_N shape times an amplitude:
φ(r, E, t) ≈ T(t) ψ_SN(r, E; θ(t)).

- **Shape.** The S_N eigenfunction at the drum angle of the moment
  (`sn-p2:4:16`, net absorption). **Refine 32** (decision 2026-11-19): refine
  16 is 5–178 pcm and up to 2.7 % per assembly from refine 32, and its `k(θ)`
  carries a mesh ripple of ±15–25 pcm (report §10.6). The SPH-DF tables are
  regenerated against refine 32 for the same reason.
- **Amplitude.** T(t) follows point kinetics with the reactivity
  ρ(t) = 1 − 1/k_SN(θ(t)), relative to the initial critical state.
- **Kinetics data.** β and λ are global. Λ comes from the S_N shape.

This is the adiabatic (quasi-static) S_N solution. It is exact in the
slow-motion limit. Its own error, the shape lag, is measured separately (step 5
below) and must be small against the differences being judged.

**Reference data.** k_SN(θ) and the shapes at the angles the manoeuvre passes:
- available: the 49 angles 0°–180° (5° and 7.5° grids) plus 8 off-midpoint angles;
- to add: references every 1° across each manoeuvre window, so that ρ(θ) and
  the shapes are interpolated far more finely than the 5° SPH tables. About
  13 min of GPU per angle at refine 32. The spacing actually needed follows
  from the refine-32 smoothness study (`tools/reference_angle_ripple.py` on
  the 5° grid, running): if refine-32 `k(θ)` is smooth at 5°, 2.5° windows
  suffice.

**Kinetics parameters.** `TriSNTransportSolver` has no adjoint, so there is no
S_N importance for β_eff and Λ. Options, in order of preference:
- **(a) Add the adjoint S_N solve.** Reverse the ordinates, transpose the
  scattering moments, and swap χ and νΣf. It is useful beyond this study,
  e.g. for exact perturbation reactivities and IQS with S_N. Estimated a few
  days, with verification against the forward k and adjoint reciprocity.
- **(b) Unweighted Λ.** Use a forward-weighted (unit-importance) Λ and the same
  global β in both models. The comparison is then about ρ(t) and shapes, not
  about the importance weighting. Acceptable for a first pass.
- **(c) Diffusion adjoint as the weight.** Take the importance from the SPH
  diffusion model; this is not independent of the model under test.

**Decision (2026-11-18): (a).** The adjoint is implemented and verified: `TriSNTransportSolver(..., adjoint=True)`, `tests/verification/test_tri_sn_adjoint.py`. For an isotropic operator (fission, delayed emission) the bilinear weights need only the scalar fluxes. For Λ, ⟨ψ†, (1/v) ψ⟩ needs the angular fluxes, or their moment expansion up to the scattering order.

**Done (2026-11-19).** `sn_effective_kinetics` (`garnet.tri_sn`):
- **β_eff** from the scalar fluxes, which is exact.
- **Λ** from the P_L flux moments. On a P2 test core it is within 1.2e-4 (step) and 2.7e-3 (SCB) of the 1/v insertion (Σ_t + ε/v, two forward solves), which is the exact first-order Λ. The scalar-flux Λ is 4 % off.
- **For the HP-MR reference,** use the moment Λ, and the insertion where a check is wanted. Λ only sets the prompt-jump time scale (tens of ms here), so 0.3 % is ample.

## Model under test: tabulated SPH-DF diffusion

- **Node model.** The drum-band mesh local:4:1 (default). Its 5° tables match
  raster:8 to 5.1 pcm and 0.02 % pin RMS at a third of the cells (report §10.6,
  2026-11-19); raster:8 only as a cross-check. **Done (2026-11-19):** the mesh
  runs in `TransientSolver` (`MeshGrid`, `MeshGroupOperator`,
  `MeshDiffusionEigenSolver`; GMRES, because the implicit non-orthogonal
  correction makes the operator non-symmetric).
- **Tables.** Zones as in the control-drums skill §4. The 5° tables hold zone
  constants, μ, γ and form functions.
- **Time loop.** `TransientSolver` (diffusion) with `problem_at(t)`. It
  interpolates every table in θ(t), applies μ to the zone constants, and
  supplies the per-cell volume mix of the drums' outer layers at θ(t).
- **Kinetics.** `sph_kinetics(kinetics, μ(θ(t)))` for v/μ. The transient's
  kinetics are fixed in time, so `SphDfTable.kinetics(base, θ0)` freezes μ for
  the speeds at the initial angle; the attribution measures v/μ(θ0) against
  v. The 11-group library kinetics exist: `hpmr_kinetics_11g()` (speeds, six
  families, β = 680 pcm, delayed spectrum), so the placeholder risk below is
  closed.
- **Boundary factors in the transient (done 2026-11-19).** `problem_at(t)`
  may return a fifth element `bcf`; it enters the boundary Robin term of the
  initial eigen solve and of every step (raster and mesh). γ = 1 reproduces
  the plain transient exactly and a static γ holds the eigenstate to 1e-9
  (`tests/verification/test_mesh_transient.py`).
- **Step settings on the HP-MR (measured 2026-11-19).** The default
  fixed-point step (Anderson depth 5, no rebalance) stalls on the first step
  (source change 9e-7, amplitude error 1e-4 after 200 sweeps): near
  criticality the source change understates the amplitude error by about
  1/(1 − k_prompt). Use the HP-MR benchmark settings
  (`hpmr_transient_bench`): `rebalance=True, anderson_depth=1,
  max_sweeps=4000, scatter_subsweeps=6, tol_step=1e-6`. A 90° → 92.5° ramp on
  `local:4:1` then converges in 21 sweeps per step, about 21 s per step on one
  CPU core.
- **Tabulated model (done 2026-11-19).** `SphDfTable` (`garnet.equivalence`):
  `state(θ)`, `problem_at(θ(t))`, `kinetics(base, θ0)`;
  `hpmr_sphdf_geometry` gives the HP-MR node geometry at any angle (the mesh
  and zone map are fixed; only the absorber blend moves).
- **Pins.** Power per assembly from the zone fission rates; pin powers from
  the form functions interpolated at θ(t).
- **Initial state.** Both models start critical at θ0: divide νΣf by k(θ0) in
  each model (the solvers' critical adjustment by k0).

## Manoeuvres

Kinetics data: the same delayed-neutron data in both models (global β, decay
constants), and 11-group speeds from the library.

| Case | θ(t) | Purpose |
|---|---|---|
| A, slow ramp out | 90° → 92.5° in 2 s, then hold to 20 s | About 200 pcm ($0.3) inserted. The quasi-static limit checks static accuracy along a path. |
| B, fast ramp out | 90° → 92.5° in 0.2 s | Shape lag and the speeds v/μ |
| C, oscillation | 90° ± 2.5°, period 2 s, 3 periods | Interpolation across table points (87.5°, 92.5° lie between 85°, 90°, 95°) |
| D, insertion (scram-like) | 180° → 0° at 90°/s | Large negative reactivity; passes every table interval and the 165°–180° range where boundary factors are smallest |
| E, ramp across a sector boundary | 40° → 50° in 1 s | The arc edge crosses an aligned sector boundary at 45°: the worst case for interpolation |

ρ(θ) is steepest between 45° and 105° (about 70–90 pcm per degree near 90°).
Amplitudes are chosen to stay below prompt critical.

## Comparisons

- **Power and reactivity.** P(t) normalized to 1 at t = 0. Report the maximum
  relative difference and the energy (∫P dt). Also the reactivity inferred by
  inverse kinetics from each P(t): the SPH model's ρ(t) against
  1 − 1/k_SN(θ(t)).
- **Spatial distribution.** Assembly powers at selected times (start, mid-ramp,
  end, oscillation peaks): worst assembly error and RMS. Pin powers at the
  same times via form functions: RMS and worst pin.
- **Attribution, as in the steady study.**
  - the same transient with exact S_N-derived tables at every step (the
    "exact table" limit) against the 5° tables;
  - v/μ against v;
  - γ(θ(t)) against γ frozen at θ0.

**Indicative targets**, to be confirmed:
- in the quasi-static limit (case A), the steady figures: ρ within 40 pcm,
  assembly powers within 0.3 %, pin RMS within 0.4 %;
- P(t) within 1 % for the $0.3 ramps.

## Steps

| # | Step | Est. effort |
|---|---|---|
| 1 | Add `bcf` (time-dependent) to `TransientSolver`; tests: γ = 1 reproduces the plain transient, a static γ gives the eigen state. **Done**, plus transients on the drum-band mesh (`MeshGrid`) | 1 day |
| 2 | `problem_at` for a tabulated SPH-DF model: interpolation in θ(t), v/μ speeds, form functions. **Done:** `SphDfTable`; refine-32 tables for `local:4:1` being generated | 1 day |
| 3 | Kinetics parameters for the reference with the S_N adjoint. **Done:** `sn_effective_kinetics` (β_eff exact, Λ from P_L moments); to run: forward + adjoint refine-32 solves at θ0 of each case | 1 day |
| 4 | S_N references in the windows of cases A–E (spacing from the refine-32 smoothness study; ~13 GPU-min per angle at refine 32); quasi-static S_N driver. **Driver done:** `adiabatic_point_kinetics` (exact on a homogeneous core) | 1 day |
| 5 | Quasi-static error itself: for the SPH model, compare the full diffusion transient with its own adiabatic/IQS solution (`quasistatic_transient`). This bounds what the quasi-static S_N reference misses. If it is not small for case B, add a short full `TransientSNSolver` window (isotropic or P2) | 1 day |
| 6 | Run cases A–E, the comparisons and the attribution; write up in the audit report | 1–2 days |

## Risks and open points

- **Tiny boundary factors.** γ near 1e-6 for arc-shadowed boundary zones at
  165°–180° enters the operator only through the Robin term, but check the
  transient's conditioning (case D).
- **Kinetics data (closed 2026-11-19).** `hpmr_kinetics_11g()` reads the
  library's own 11-group speeds and six delayed families (β = 680 pcm).
- **The SPH model's own adjoint on the mesh.** Step 5 projects the SPH model
  onto its own adjoint; the mesh operator is non-symmetric on hanging-node
  meshes and has no adjoint yet. Use raster:8 for step 5, or the 1/v
  insertion and prompt-k for that model's Λ and β_eff.
- **Shape lag and S_N anisotropy.** If the quasi-static error (step 5) is
  comparable to the SPH error in the fast cases, the reference for those cases
  must be a full S_N transient. `TransientSNSolver` supports triangles; check
  P2 anisotropy and the cost first.
- **3-D and independent drum angles** are out of scope here: all 12 drums move
  together, in 2-D.
