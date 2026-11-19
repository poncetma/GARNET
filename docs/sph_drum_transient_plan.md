# Plan: drum-rotation transient, SPH-DF diffusion against quasi-static S_N

Status: plan, 2026-11-18. Not started. Follows the SPH audit
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
  (`sn-p2:4:16`, refine 16, net absorption; the existing reference recipe).
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
  3 min of GPU per angle.

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

**Decision (2026-11-18): (a).** The adjoint is implemented and verified: `TriSNTransportSolver(..., adjoint=True)`, `tests/verification/test_tri_sn_adjoint.py`. For an isotropic operator (fission, delayed emission) the bilinear weights need only the scalar fluxes. For Λ, ⟨ψ†, (1/v) ψ⟩ needs the angular fluxes, or their moment expansion up to the scattering order. That is still to be added in step 3.

## Model under test: tabulated SPH-DF diffusion

- **Node model.** The drum-band mesh local:4:1 (default). Its 5° tables match
  raster:8 to 5.1 pcm and 0.02 % pin RMS at a third of the cells (report §10.6,
  2026-11-19); raster:8 only as a cross-check.
- **Tables.** Zones as in the control-drums skill §4. The 5° tables hold zone
  constants, μ, γ and form functions.
- **Time loop.** `TransientSolver` (diffusion) with `problem_at(t)`. It
  interpolates every table in θ(t), applies μ to the zone constants, and
  supplies the per-cell volume mix of the drums' outer layers at θ(t).
- **Kinetics.** `sph_kinetics(kinetics, μ(θ(t)))` for v/μ. The speeds change
  with time through μ; check that the solver accepts time-dependent
  per-material speeds, or freeze μ for the speeds at the interval start.
- **Missing feature: boundary factors in the transient.** The diffusion
  transient has no `bcf`, while the eigen solvers do. Add a time-dependent `bcf`
  to `TransientSolver`, returned by `problem_at(t)` or as `bcf_at(t)`. It
  enters only the boundary Robin term of the operator, as in the eigen solver.
  Test that a constant γ = 1 reproduces the plain transient and that a static
  γ reproduces the eigen solution as the steady state.
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
| 1 | Add `bcf` (time-dependent) to `TransientSolver`; tests: γ = 1 reproduces the plain transient, a static γ gives the eigen state | 1 day |
| 2 | `problem_at` for a tabulated SPH-DF model: interpolation in θ(t), v/μ speeds, form functions; as a function in `garnet.equivalence` or a tool | 1 day |
| 3 | Kinetics parameters for the reference with the S_N adjoint (done: the adjoint solve; to do: the angular or moment inner product for Λ, and β_eff) | 1 day |
| 4 | S_N references every 1° in the windows of cases A–E (~60 angles, ~3 GPU-hours); quasi-static S_N driver: ρ(θ), shapes, point kinetics | 1 day |
| 5 | Quasi-static error itself: for the SPH model, compare the full diffusion transient with its own adiabatic/IQS solution (`quasistatic_transient`). This bounds what the quasi-static S_N reference misses. If it is not small for case B, add a short full `TransientSNSolver` window (isotropic or P2) | 1 day |
| 6 | Run cases A–E, the comparisons and the attribution; write up in the audit report | 1–2 days |

## Risks and open points

- **Tiny boundary factors.** γ near 1e-6 for arc-shadowed boundary zones at
  165°–180° enters the operator only through the Robin term, but check the
  transient's conditioning (case D).
- **Kinetics data.** The library's delayed-neutron data are not wired into
  GARNET's HP-MR model. The placeholder `HPMR_KINETICS` is 2-group; 11-group
  speeds and β are needed.
- **Shape lag and S_N anisotropy.** If the quasi-static error (step 5) is
  comparable to the SPH error in the fast cases, the reference for those cases
  must be a full S_N transient. `TransientSNSolver` supports triangles; check
  P2 anisotropy and the cost first.
- **3-D and independent drum angles** are out of scope here: all 12 drums move
  together, in 2-D.
