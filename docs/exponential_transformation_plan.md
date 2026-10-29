# Exponential transformation for diffusion transients: assessment and plan

Assessment date: 2026-10-25. Source reviewed: `b496309290d15033234b9b737ad351d4e415d306`.
Development/literature update: 2026-10-26, `572ef80` plus the current
adaptive/output-aware FTM working tree. The historical assessment is retained below;
the phase tables and next-work priorities reflect the new evidence.

**Prototype follow-up:** the shared-frequency diffusion implementation is now
available as `time_scheme="ftm"`. Per the requested priority, the first
comparators are fixed/adaptive BE and adaptive BDF, including thermal coupling;
IQS comparisons are deferred. See the
[implementation, CPU measurements and remaining work](../benchmark-results/exponential-prototype/README.md).
Neutron/precursor adaptive FTM is now available in the fourth checkpoint;
diagonal-only operator updates and coupled thermal adaptation remain pending.
The [proposal follow-up](exponential_proposal_review.md) identifies true
second-order exponential theta, source-group precursor weights and optional
frequency correction as separate experiments, and adds a CN/work comparison.
The [second-order development checkpoint](../benchmark-results/exponential-second-order/README.md)
now implements shared-frequency `ftm2` with linear transformed precursor
production. Further work proceeds in small chunks. The heat-transfer solve
is outside the exponential-transformation scope.
The [second checkpoint](../benchmark-results/exponential-second-order/CHUNK2.md)
adds BE start-up and optional frequency refinement. A resolved rapid-growth
case now demonstrates a temporal benefit from the transform at equal work;
adaptive BDF still uses fewer tested solves. That checkpoint motivated the
subsequent damping experiment before expanding the spatial frequency model.
The [third checkpoint](../benchmark-results/exponential-second-order/CHUNK3.md)
now tests a two-stage damped candidate independently and adds the one-solve
`ftm-bdf2` recurrence to the FV engine. It transforms neutron and precursor
histories together, with BE safeguards and explicit event restart. The
196-configuration comparison retains adaptive BDF as the strongest tested
comparator. It left physical-state error control and a localized shape case
as the next steps, completed below; group/local frequencies remain pending.
The [fourth checkpoint](../benchmark-results/exponential-second-order/CHUNK4.md)
adds opt-in `adaptive_ftm` with physical flux/precursor step doubling, complete
history rollback, exact event alignment and a localized two-group matrix
reference. Adaptive BDF still uses less tested work at matched output error.
The [fifth checkpoint](../benchmark-results/exponential-second-order/CHUNK5.md)
adds optional `output_rtol` midpoint/chord width control in each physical
field and in power. It resolves the rapid-growth linear-output discrepancy
at a declared physical budget, without extra solves per attempt.
The [sixth checkpoint](../benchmark-results/exponential-second-order/CHUNK6.md)
adds opt-in two-solve physical-history estimation and a same-trial audit
against the retained step-doubling oracle. It reduces tested FTM solves at
the 0.1% output/integral/precursor budget, while adaptive BDF remains cheaper.
The [seventh checkpoint](../benchmark-results/exponential-second-order/CHUNK7.md)
assesses TWIGL step/ramp on 20×20 and 40×40 meshes with independent temporal
references, an 80×80 spatial check and stronger output-width BE/BDF controls.
Adaptive BDF remains cheaper at the declared 0.1% temporal budget. Thermal
flux shape differs by 1.10% between 40×40 and 80×80, so the 0.1% spatial-shape
gate remains open. Retain the initial-state resolution diagnosis before
larger transient claims. Setup reuse was subsequently closed (chunk 8). Tight
exponential tails can make physical-history checking more restrictive;
keep its audit mode available and hold local/group frequency expansion.
The [eighth checkpoint](../benchmark-results/exponential-second-order/CHUNK8.md)
closes setup reuse by measurement: rebuilds are 0.15-0.93% of CPU march time,
so caching cannot approach the promotion gate. It adds the E5a control,
`precursor_production="quadratic-physical"` (PARCS physical production), and
compares it with the default and linear physical production on nine cases.
All models accept the same steps; only final precursor errors change, in a
regime-dependent direction and far below budget. No error/work benefit is
found. Allowing FTM `output_rtol=1e-3` also corrects chunk 7: on the TWIGL
ramp FTM2 now ties tuned BDF (87 versus 92 solves on 20×20, 93 versus 92 on
40×40), while BDF remains 1.46-1.85x cheaper on the step. After merging the
audit's BDF startup-estimator fix, FTM2 is unchanged and the cheapest
qualifying tested BDF ramp run needs 101 solves, but a 76-solve run misses the
budget by only 0.7%; the ramp stays a tie and BDF is 1.47-1.87x cheaper on
the step.

**Recommendation: continue the opt-in frequency transformation method (FTM),
retaining shared-frequency controls while testing each extension.** It can reuse
GARNET's matrix-free diffusion engines. Its main possible benefit is fewer
spatial solves at a given temporal error. A speedup over the best existing
adaptive BE/BDF or guarded IQS configuration remains unproven. Published
SIMULATE results do not establish performance in this implementation.

**Literature decision:** the [review of the five supplied PDFs](exponential_literature_review_2026-10-26.md)
directly supports the transformed-theta neutron equation: PARCS v2.6 uses it
with quadratic **physical-source** precursor integration. SIMULATE5-K's 2022
paper instead describes transformed BE, source-group exponential precursor
weights and a separate pseudo-static nodal leakage construction. These are
different contracts, not one method to copy wholesale. Add a controlled
PARCS precursor experiment and a separate, bounded KIOPS matrix-action study.
Neither changes the current production default or the heat equation.

The [scaling assessment](numerical_scaling_plan.md) is a separate workstream.
Establish its error-norm diagnostics before adaptive comparisons; a change in
the error metric must not be mistaken for an FTM improvement.

## Capabilities at the original assessment and relevant history

| Implementation | Consequence |
|---|---|
| [`timescheme.py`](../garnet/timescheme.py): BE, theta methods, BDF1–6, nonuniform coefficients, prediction and order/width control | Much infrastructure is reusable. No exponential/frequency-transformed transient scheme was found. |
| [`TransientSolver`](../garnet/transient.py): precursor elimination, groupwise PCG, optional monolithic FGMRES, per-material kinetics and per-family delayed spectra | Preserve these capabilities. Current precursor formulas solve the **discretized BDF equations**, rather than exactly integrating the continuous precursor ODE for an exponential source. |
| [`MultigroupStepOperator`](../garnet/multigroup.py) and its energy-sweep preconditioner | A shared-frequency prototype can retain the coupled linear-system structure. |
| Adaptive integration requires a `BDF` object and `step_solver="monolithic"` | Adding an FTM class alone will not enable adaptivity: type checks, prediction, history and acceptance need an explicit extension. |
| [`quasistatic.py`](../garnet/quasistatic.py): guarded amplitude/shape factorization and full-diffusion fallback | FTM overlaps with some IQS benefits. Compare independently before composing them. |

The [raw-flux chronological predictor](../benchmark-results/transient-acceleration/step-01-chronological-initial-guess.md)
increased work on the stiff 11-group case. A transformed time discretization
changes the approximation, so that result does not reject FTM; it does warn
against assuming that amplitude prediction improves nested multigroup solves.

The [matched HP-MR study](../benchmark-results/transient-acceleration/step-07j-hpmr-speedup-and-coupling.md)
found adaptive BE and automatic BDF essentially tied on mild motion. The
[LRA stored summary](../benchmark-results/lra2d-adaptive-bdf-report/benchmark-summary.json)
instead records a 5.10x CPU reduction for BDF over BE at comparable first-peak
temporal error (201 versus 3180 accepted steps). FTM must compete with these
existing methods, rather than only a deliberately fine fixed-BE reference.

## Formulation and implementation fit

This is an integrating-factor change of unknown within each step, not a
Fourier transform or a large matrix-exponential solve. Published
[SIMULATE5-K-VVER work](https://doi.org/10.1016/j.anucene.2023.110146)
supports FTM alongside BE. The authors' [M&C 2023 paper, sections 2.1–2.2](https://www.researchgate.net/publication/374111958_Development_and_Benchmarking_of_Transient_Nodal_Code_SIMULATE5-K_Neutron_Kinetics_Solver_for_VVERs_and_Hexagonal_Geometries)
describes local group frequencies, transformed flux integration and analytic
precursor elimination. GARNET can adopt the temporal idea without adopting
SIMULATE's nodal spatial model. SIMULATE-3K-style linear transformed-shape
precursor interpolation is also described by
[Kreher, Smith and Forget](https://doi.org/10.1080/00295639.2021.1980363).

Let `h = t[n+1] - t[n]`, `M_g = 1/v_g`, and write

```text
phi_g(t) = exp(omega * (t - t[n])) psi_g(t).
```

For a frozen frequency, backward differencing of `psi` gives, in physical
endpoint flux variables,

```text
[L_g + M_g (1/h + omega)] phi_g[n+1]
  = M_g exp(omega h) phi_g[n]/h + scattering + prompt + delayed.
```

`L_g` is GARNET's loss operator before the time shift. This preserves face
couplings. Even a later cell-dependent frequency can be used in this endpoint
equation without explicitly differentiating `exp(omega(x) t)` in the leakage
term. Geometry-dependent RHS weights must still match the existing solver.

For the simplest exponential end-of-step source approximation, define

```text
H_i = h phi_1(-(lambda_i + omega) h),
phi_1(z) = (exp(z) - 1)/z,  phi_1(0) = 1,
C_i[n+1] = exp(-lambda_i h) C_i[n] + beta_i H_i S[n+1].
```

Here `S = sum_g nuSigma_f,g phi_g / k0`, retaining the critical adjustment.
In GARNET's total-spectrum convention the emission weight and fixed delayed
source become

```text
w_g = chi_total,g - sum_i chi_delayed,ig beta_i (1 - lambda_i H_i),
delayed_history_g = sum_i chi_delayed,ig lambda_i exp(-lambda_i h) C_i[n].
```

Treating `fields.chi` as a prompt spectrum would silently change the physics.
These expressions also allow mapped material beta/velocity fields and
distinct delayed spectra. Evaluate `H_i` using `expm1` and a series near zero:
`omega = -lambda_i` is a finite limit.

With a shared frequency, one scalar fission source remains sufficient. With
a separate frequency for each **source group**, precursor production instead
contains `sum_h beta_i H_i,h nuSigma_f,h phi_h/k0`. It generally cannot be
represented by changing only destination-group `w_fis[g]`. Group-dependent
FTM with these exponential precursor weights needs extra source contractions
or generalized low-rank fission coupling. PARCS-style physical-source
interpolation is a different alternative that retains a scalar endpoint
source; compare its accuracy separately before expanding the frequency model.

The initial approximation above is first order. A later variant can
interpolate `psi` linearly and integrate precursors with stable `phi_1`/`phi_2`
coefficients; that alone does not make flux integration second order.
**Zero frequency restores BE's flux derivative but not BE's rational
precursor update.** Benchmark a zero-frequency/exact-precursor control
separately, and make fallback invoke the actual existing BE method.

## Potential gain and practical limits

For `y' = alpha y`, transformed BE has amplification
`exp(omega h)/(1 - (alpha - omega)h)`, which is exact at `omega = alpha`.
For a close, fixed frequency its leading growth-rate error depends on
`(alpha - omega)^2 h`, rather than `alpha^2 h`. This explains the opportunity
on near-exponential segments; it gives no general speedup factor for a
superposition of spatial and precursor modes.

Independent [analytic probes](../benchmark-results/transient-numerics-assessment/analytic-probes.json)
use one-family point kinetics with `beta=0.0065`, `lambda=0.08/s`,
`Lambda=1e-4 s`, and reactivity step `rho=0.002`. Frequency is lagged from two
populations. Errors are maximum relative population errors at each run's
step endpoints:

| Window | h | BE error | Lagged FTM error |
|---|---:|---:|---:|
| Entire step response, 0–10 s | 0.01 s | 2.422% | 2.544% |
| Entire step response, 0–10 s | 0.001 s | 0.2832% | 0.3514% |
| Exponential tail, 0.5–10 s | 0.1 s | 0.05990% | 2.88e-9% |

The tail starts from the **exact** state and one exact previous population
sample, removing inherited startup error. These are independent ODE probes,
not GARNET runs, BDF comparisons or measured speedups. They demonstrate both
the opportunity and the weakness of naive frequency prediction at a prompt jump.

If spatial neutronics is fraction `f` of baseline march time, `r` is the
reduction in total attempted spatial steps, and `c` is their average cost
multiplier, an illustrative total speedup is

```text
speedup = 1 / ((1 - f) + f c/r).
```

With `f=0.97` (consistent with existing coupled profiles) and **assumed**,
unmeasured `c=1.2`, halving the attempt count gives 1.63x; quartering it gives
3.12x. With unchanged count it gives 0.84x. These are scenarios, not forecasts.
A step-doubling controller can require three spatial solves, so its overhead
cannot be assumed to be only 20%.

Reasons a benefit may disappear:

- Abrupt insertion, changing shape, opposing regional responses and strong
  feedback need resolution even when global power looks exponential.
- Larger steps weaken the positive BE time diagonal; negative frequency can
  weaken it further. Inner/outer iterations or failures may increase.
  Implicit treatment alone does not establish unconditional stability or
  positivity for a nonlinear frequency update.
- Changing frequency currently requires operator/preconditioner rebuilds
  unless time diagonals are separated from geometric stencil construction.
- Control frames, thermal exchanges and required output endpoints limit step
  widths. Dense output would need its own error check.
- Guarded IQS already removes most fine-step spatial work on slow maneuvers.
  Its [GH200 acceptance](../benchmark-results/hpmr-iqs-gh200/README.md)
  establishes the production comparator, not a measured FTM headroom.

## Separate development plan

The E4 spatial comparison has now been assessed on TWIGL: temporal errors
are independently resolved on each study mesh, while the declared spatial
shape budget and performance-promotion gate remain unmet. Resolve the mesh
error before claiming application-wide 0.1% accuracy. Setup reuse is closed
(measured CPU share below 1%), and E5a found no gain from precursor production
interpolation. Do not broaden FTM frequency models on the present matched-work
result: tune both comparators along the same output-width dimensions first. The two-solve history estimator remains
opt-in, with the three-solve controller retained as its reference.
K0 is a small independent feasibility
gate; a full Krylov integrator is not a prerequisite for continuing FTM.
Keep each experimental change separately switchable and compare resulting
physical errors.

| Phase / status | Concrete work | Gate and deliverable |
|---|---|---|
| E0: TWIGL temporal baseline added; spatial resolution pending | Homogeneous/ramp/event comparisons cover BE, CN and adaptive BDF. TWIGL step/ramp now has independent physical-generator temporal references and output-width BE/BDF controls on two meshes; its thermal shape still needs refinement for a 0.1% spatial claim. LRA and HP-MR remain later applications. Keep original BDF decisions reproducible and report per-field defects separately. IQS remains deferred. | Same spatial/control path, converged temporal reference, power/shape/energy/precursor errors and all accepted/rejected work. |
| E1: implemented | Retain `ftm`, transformed-theta `ftm2` and damped `ftm-bdf2`, with their distinct precursor formulas and zero-frequency controls. | Existing equilibrium, mixed-mode, second-order, material/spectrum/geometry and damping verification remains green. |
| E2: frequency implemented; caching closed | Accepted active-volume neutron populations, actual previous widths, bounded log rates, event resets, BE safeguards and optional correction are implemented. Setup reuse was measured instead of built: all rebuild work is 0.15-0.93% of CPU march time on TWIGL 40×40/160×160, so separating time diagonals from stencil setup cannot approach the promotion gate. Test fitting inside existing spatial iterations only as a separate cost experiment. | No stale fields, face couplings or workspaces after dt/frequency/material changes; charge all correction/setup work. Do not add an eigen solve solely for a frequency. Revisit caching only if a GPU profile shows a material setup share. |
| E3: neutron adaptation implemented | `adaptive_ftm` uses physical full/two-half-step defects per flux group and precursor family, and restores state, production, frequency and recurrence histories. Exact event/alignment endpoints and all-trial counters are verified. External feedback/thermal state is explicitly excluded for now. | Accepted-half-step replay, independent kinetics/Radau and localized matrix-exponential references. Keep three-solve control as the oracle for a cheaper estimator. Linear-solve-failure retry remains to be designed. |
| E3b: output restriction and history estimator implemented | Optional `output_rtol` tests physical flux/precursor and scalar-power midpoint/chord defects. The opt-in history estimator uses two half solves and actual-width quadratic physical prediction, retaining step doubling for startup/events/unsafe histories and charging discarded BE-fallback work. Same-trial audits, independent references and state-driven rejection/replay checks calibrate it. | Keep endpoint, interpolated power/shape and power-integral errors separate. Matched-budget work improves over the three-solve FTM controller, while adaptive BDF remains cheaper. Tight tails can over-resolve physical curvature; local tolerance and empirical audits are not global guarantees. Validate larger spatial output histories and two-sided power jumps separately. |
| E4: TWIGL assessed; spatial and promotion gates unmet | The 116-configuration TWIGL study compares physical power/field/shape/region/integral/final-precursor errors, tuned BE/BDF output caps, same-mesh zero-frequency controls and estimator audits. With FTM `output_rtol` also tuned (chunk 8), FTM2 ties BDF on the ramp and BDF remains 1.46-1.87x cheaper on the step at the 0.1% budget, before and after the audit's BDF fix; 40×40-to-80×80 thermal shape differs by 1.10%. A 160×160 study needs a linear tolerance above the FGMRES floor (about 1e-10). Resolve spatial error before repeated CPU/GPU timings. Strong feedback still needs its own provisional-state controller. | Retain the best acceptable BE/BDF comparator. Keep spatial and temporal errors separate. Count every constituent/rejected solve, Krylov work and rebuild; timing promotion additionally needs setup/transfer accounting and the contract below. |
| E5a: implemented; no benefit | `ftm2` accepts `precursor_production` = linear-transformed (default), linear-physical or quadratic-physical, with stable positive decay moments, unequal widths, mapped beta products, linear startup/restart and complete rollback. Nine cases with independent references: identical accepted steps and work in 44 of 46 adaptive groups; precursor errors improve in slow insertions/shutdown and worsen for near-exponential production, always at least 50x below budget. Transformed-BDF precursors at FTM2 neutron formulas were not added. | Gate unmet: no physical error/work benefit. Keep as an opt-in research control only; it supplies the frequency-independent scalar endpoint source needed by E5b. |
| E5b: conditional frequency extensions | Test a small regional/cell frequency model on a shape-changing case before full node/group rates. Distinguish PARCS physical-source interpolation, which preserves scalar endpoint fission coupling, from SIMULATE source-group exponential weights, which need family/source-group contractions. | Correct endpoint equations, old derivatives/history in the current frame, admissible spatial solves and a measured benefit over shared frequency. Different precursor families cannot simply inherit unrelated group frames. |
| E6: later composition | Only after the relevant error/cost gates, consider FTM in IQS correctors/fallbacks or a thermal-coupled controller. | Explicit rollback of every coupled state and independent heat accuracy. Speedups over IQS are measured, not multiplied from separate studies. |

## Separate matrix-exponential-action feasibility track

KIOPS evaluates matrix phi-function actions; it does not accelerate the scalar
`phi1`/`phi2` coefficients in `_exponential.py`. PARCS's BiCGSTAB is instead a
linear-system solver, a role already served by GARNET PCG/FGMRES. The
[literature review](exponential_literature_review_2026-10-26.md) gives the
equations, source distinctions, memory estimate and paper-specific caveats.

| Gate | Bounded work | Acceptance / stop condition |
|---|---|---|
| K0: physical generator oracle | Construct the full active neutron/precursor `J=M^-1*A` on tiny problems, retaining leakage, mapped kinetics, delayed spectra, `k0` and geometry weights. Compare generator actions with independent assembled matrices and existing physical RHS calculations. | Equilibrium, complete/mixed modes, stiff shutdown, six families and the localized two-group case. The current time-shifted, precursor-eliminated `MultigroupStepOperator` is not `J`. Use dense `expm` and sparse `expm_multiply` as references; the latter is a Taylor-action algorithm, not KIOPS. |
| K1: corrected polynomial action | Validate conventional Arnoldi, then KIOPS-style augmented phi combinations and length-two incomplete orthogonalization. Follow the supplied 2021 corrigendum. Sweep action tolerance, scaling, basis limits and internal substeps; test several requested output times. | Independently measured physical block errors, singular/critical and growing/nonnormal cases, masks and unit/normalization invariance. Include complete basis memory and reduction counts. Do not import negative-semidefinite guarantees into the general reactor operator. |
| K2: small time-integrator comparison | Start with frozen linear segments; then a genuinely second-order smooth prescribed-time formulation, such as a midpoint exponential, or a correctly derived nonautonomous EPI method. Add remainder/feedback treatment only with its own derivation and error tests. | Separate action, temporal, spatial and output errors; retain event alignment and physical rejection/rollback. Inner KIOPS adaptivity is not an outer transient error controller. Stop expansion if action work/memory is already uncompetitive with adaptive BDF. |
| K3: rational action, conditional | If polynomial spaces/internal substeps grow excessively under mesh/velocity refinement, prototype shifted-inverse actions with reusable monolithic neutron/precursor solves and controlled inner accuracy. Avoid production full-matrix LU. | Count every shifted solve and its inner Krylov work, rebuilds and nested basis memory. An approximate preconditioner is not the exact inverse action, and arbitrary left preconditioning changes the exponential. |
| K4: CPU/GPU retention | Compare the best action-based candidate with the best acceptable full-diffusion comparator on two resolved target cases. Reuse backend kernels selectively and measure memory/synchronization. | Same observable-error and timing contract as E4. No speedup claim from a one-cell problem, fewer macro steps or another domain's timings. |

The scalar-frequency and matrix-action tracks may be assessed independently.
Do not broaden either into a production default without its own evidence.
The circuit paper's rational Krylov/LU results and the atmosphere paper's
longer stable steps motivate experiments; neither establishes reactor/GPU
efficiency. No additional production time integrator is introduced by this
literature update.

## FTM implementation requirements

- `TimeScheme.a0()` is one scalar used for time shift and history weighting.
  Avoid encoding everything through it: zero `1 + omega*h` would make an
  artificial carried-state division fail, and group/cell frequencies do not
  fit its contract.
- Retry with smaller `h` or actual BE when estimates overflow, the group
  operator is no longer admissible for PCG, or the coupled solve fails.
  Check the physical residual and block solve; a positive diagonal alone
  does not prove SPD or full-system stability.
- Freeze frequency during the initial linear solve. If endpoint frequency
  correction is added, count every extra solve and converge the
  frequency/feedback iteration before committing state.
- Small transformed-shape defect does not bound physical amplitude error
  when the estimated frequency is wrong.

## Benchmark and acceptance contract

1. **Analytic correctness:** stationary critical equilibrium; prompt-only
   diffusion modes; one- and six-family point kinetics; prescribed
   exponential-source precursors; `omega + lambda_i -> 0`; small/large `h`;
   falling flux; temporal refinement. Check neutron balance with the new
   time derivative and precursor formula.
2. **Coverage:** per-material velocities/beta, delayed spectra, masks/blends,
   Cartesian and triangular/prismatic geometry, cylindrical RHS weighting,
   feedback retries and variable-step rollback. Start with diffusion;
   transport moments require a separate derivation.
3. **Applications:** TWIGL step/ramp and shape changes; LRA prompt peak and
   rod stop; 11-group HP-MR uniform insertion; slow/fast/asymmetric drums;
   strong-feedback coupling; then production 3-D GPU cases.
4. **Fair comparison:** time-refined and demonstrably converged reference;
   match power, peak time, energy, local/shape power and temperature error.
   Equal `rtol`, width or output count is insufficient. Use the same spatial
   model/control path and common output grid. Start exploration at declared
   0.1% maximum power and volume-weighted shape-error budgets and 0.1 K
   temperature budget, tightening where an existing benchmark requires it.
5. **Promotion:** seek at least 1.2x median march improvement on two target
   cases against the fastest acceptable full-diffusion configuration, with
   no unexplained regression at sharp events. This is a proposed retention
   gate, not a predicted gain. Report initial eigen/adjoint setup separately
   and include total application time. Warm/synchronize GPUs; retain at
   least three interleaved samples and raw work counters.

Stop broadening the implementation if gains occur only in ideal tails already
handled more cheaply by IQS, or disappear at matched physical error. An
accurate fixed-width FTM can still remain an optional reference method.

## Evidence and reproduction limits

The source/history and recorded reports were reviewed; no previous FTM
implementation was identified in the accessible checkout. Reproduce the
independent probes with:

```bash
python3 tools/assess_transient_numerics.py \
  --output benchmark-results/transient-numerics-assessment/analytic-probes.json
```

At the initial assessment, the environment had Python 3.14.4 but no NumPy,
SciPy or CuPy. Only the independent probes ran at that stage, and the timings
in this original assessment came from existing reports. NumPy, SciPy and
pytest were subsequently installed; new solver validation and CPU timings
belong to the linked prototype study. GPU performance remains unmeasured.
