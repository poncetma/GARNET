# Plan: drum-rotation transient, SPH-DF diffusion against quasi-static S_N

Status: steps 1–6 done 2026-11-19; 2.5° tables and a smoothed S_N k(θ) 2026-11-20
(A, B within 1 % in power); results in the audit report §10.8. Open: the reference's
mesh ripple (`docs/arc_conforming_reference_plan.md`) and the full S_N transient
for deep insertions (in development). Follows the SPH audit
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
- **Decision (2026-11-19): no 1° windows.** Refine-32 `k(θ)` between the 5°
  points deviates from linear interpolation by 7.3 pcm RMS (worst −20 pcm at
  63.75°). A cubic spline does no better (8.0 pcm RMS), so the residual is
  the reference's own mesh ripple, not curvature, and finer sampling would only
  resolve that ripple. Shapes interpolate to 0.17 % pin RMS and 1.7 % worst
  pin. The reference needs exact solves only where the drums hold or powers
  are compared: 87.5° and 92.5° (job 255276) added to the 5° grid. Along the
  ramps ρ(θ) is interpolated linearly between exact points, an uncertainty
  of about 7 pcm, the reference's own.

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
- **Step settings on the HP-MR (measured 2026-11-19).** Use
  `step_solver="monolithic"`: 3.3 s per step on one CPU core for the ramp below,
  with the inner CG of its preconditioner on the operator's symmetric part.
  The default fixed-point step (Anderson depth 5, no rebalance) stalls on the first step
  (source change 9e-7, amplitude error 1e-4 after 200 sweeps): near
  criticality the source change understates the amplitude error by about
  1/(1 − k_prompt). Use the HP-MR benchmark settings
  (`hpmr_transient_bench`): `rebalance=True, anderson_depth=1,
  max_sweeps=4000, scatter_subsweeps=6, tol_step=1e-6`. A 90° → 92.5° ramp on
  `local:4:1` then converges in 21 sweeps per step, about 21 s per step on one
  CPU core, six times the monolithic cost.
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
| 4 | S_N references at the hold and comparison angles (87.5°, 92.5° added to the refine-32 5° grid; no 1° windows, see above); quasi-static S_N driver. **Driver done:** `adiabatic_point_kinetics` (exact on a homogeneous core) | 1 day |
| 5 | Quasi-static error itself: for the SPH model, compare the full diffusion transient with its own adiabatic solution. This bounds what the quasi-static S_N reference misses. If it is not small for case B, add a short full `TransientSNSolver` window (isotropic or P2). **Running (2026-11-19):** `tools/hpmr_drum_transient.py run CASE`: the full transient, the static SPH-DF solves at every angle of the path, and the adiabatic point kinetics with β_eff, Λ projected from the model's own forward and adjoint at θ0 (`project_effective_kinetics` on the mesh). The adiabatic power carries the shape norm Σ pins / ⟨φ0†, V⁻¹ψ(θ)⟩, as the S_N reference does; the amplitude alone is off by 0.025 % after two steps of case B (0.55 % over the full case B). **First results (2026-11-19):** case B (90° → 92.5° in 0.2 s, $0.25): adiabatic vs full power 0.089 % (energy −0.080 %), the same at dt/2; shape lag 0.006 % pin RMS, 0.021 % worst pin. Case A (2 s ramp, hold to 20 s): 0.150 % at 20 s (energy −0.120 %), shape lag 0.027 % worst pin. The power gap is a period effect, not shape lag: with β_eff and Λ projected at 92.5° with the fixed weight φ0† (ρ then equals 1 − k0/k exactly; β_eff −0.008 %, Λ +0.062 %) and moved with the angle, the gap falls to 0.063 % (A) and 0.015 % (B). The rest grows slowly through the hold (ω-mode against static shape). Projecting with the state's own k as k0 is wrong: Λ ∝ k0, an extra +0.18 %. Diagnostic broken on purpose: library β instead of the projected β_eff gives 1.7 %, 2Λ gives 2.5 % (case B). Case C (±2.5°): frozen 0.56 % at the 87.5° trough, moving 0.009 %; under the fixed weight Λ falls 0.56 % at 87.5° (β_eff < 0.01 %). **Decision (2026-11-19): both adiabatic solutions move β_eff and Λ with the angle under the fixed weight φ0†.** SPH-DF: `project_effective_kinetics(solver(θ), ψ(θ), φ0†)` at every angle of the path. S_N: β_eff(θ) exact from the scalar fluxes (checked against the kinetics job at θ0 to 1e-6), Λ(θ) = Λ_moments(θ0) × Λ_P0(θ)/Λ_P0(θ0). The frozen variants are kept for comparison. **Results with moving kinetics (2026-11-19), adiabatic vs full SPH-DF, max power gap / worst-pin shape lag:** A 0.063 % / 0.027 %; B 0.015 % / 0.021 % (the same at dt/2); C 0.009 % / 0.023 %; E2 (43.75° → 46.25°, $0.22, across the 45° sector edge) 0.020 % / 0.029 %. **E (40° → 50°) inserts $0.88:** power 10⁵ at 5 s, adiabatic gap 8.0 % (10.6 % frozen) at a shape lag of only 0.1 %: near prompt critical the stable mode departs from the static shape, so no quasi-static reference can judge E at 1 %; E is a stress case and E2 the sector-edge test. **D (180° → 0°, −$13.4):** the quasi-static construction holds to t = 0.5 s (−$1.6): power 0.06 %, worst pin 0.27 %; 0.45 % / 1.0 % at 0.75 s (−$3.5); then the flux becomes precursor-driven (6.6 % / 10 % at 2 s). D is compared with S_N to 0.5 s; later times need a full S_N transient. **S_N adjoint cost (fixed 2026-11-19).** The refine-32 kinetics jobs ran for hours with the GPU idle: the adjoint swept the groups fast to thermal, against its transposed transfers, so the CMFD power iteration (host sparse LU per group) took 50–84 passes per outer instead of 6–7. The adjoint now sweeps thermal to fast in the outer loop and in `_cmfd_power` (`group_order`); forward unchanged; test `test_adjoint_cmfd_power_sweeps_groups_thermal_to_fast` (broken on purpose: 50 passes per outer) | 1 day |
| 6 | Run cases A–E, the comparisons and the attribution; write up in the audit report. **Done (2026-11-19), audit report §10.8:** shapes hold (pin RMS ≤ 0.17 %, worst pin ≤ 1.6 %, worst assembly ≤ 0.18 %); P(t) is off by 2–12 % (A −12.1 % at 20 s, B −3.3 %, C −3.9 %, E2 +1.8 %, D −2.0 % to 0.5 s), entirely from the static reactivity between 5° table points (4–18 pcm; the SPH-DF k sags 15–20 pcm below the chord near 90°, S_N only 4 pcm): with the S_N k(θ) the SPH-DF kinetics reproduce P(t) within 0.3 % (D 1 %). v/μ against v ≤ 0.025 %; γ frozen 0.06–0.23 % (A, B), up to 3 % in D. Set-up as run: the same runs give the SPH-DF transients (jobs 255329–255339; D waits for the 170°/175° tables). Attribution variants: `v` (library speeds) for B and D, `gamma0` (γ frozen at θ0) for A, B and D; case B also at dt/2. `compare` builds the S_N reference from the refine-32 solves at all exact angles and the S_N kinetics at θ0 = 90°, 180°, 40° (jobs 255256–255258). The "exact table at every step" variant is dropped: the path is continuous, and the steady study already gives the interpolation error at the off-table angles | 1–2 days |

## Risks and open points

- **Tiny boundary factors.** γ near 1e-6 for arc-shadowed boundary zones at
  165°–180° enters the operator only through the Robin term, but check the
  transient's conditioning (case D).
- **Kinetics data (closed 2026-11-19).** `hpmr_kinetics_11g()` reads the
  library's own 11-group speeds and six delayed families (β = 680 pcm).
- **The SPH model's own adjoint on the mesh (closed 2026-11-19).**
  `MeshDiffusionEigenSolver` solves the adjoint with the exact transpose
  of the operator, so step 5 can project the SPH model on the mesh itself.
- **Shape lag and S_N anisotropy.** If the quasi-static error (step 5) is
  comparable to the SPH error in the fast cases, the reference for those cases
  must be a full S_N transient. `TransientSNSolver` supports triangles; check
  P2 anisotropy and the cost first.
- **3-D and independent drum angles** are out of scope here: all 12 drums move
  together, in 2-D.
