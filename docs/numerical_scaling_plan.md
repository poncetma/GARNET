# Numerical scaling: assessment and separate development plan

Assessment date: 2026-10-25. Source reviewed: `b496309290d15033234b9b737ad351d4e415d306`.

**Recommendation: prioritize explicit scaling of error control and residual
diagnostics; investigate multigroup equilibration only if measurements justify
it. A wholesale conversion to dimensionless units is not a promising speed
optimization by itself.** The first benefit is interpretable accuracy and
robustness. Speed must be demonstrated at matched physical error. This work
is independent of the [exponential transformation plan](exponential_transformation_plan.md).

## What has already been done or tried

| Evidence | Interpretation |
|---|---|
| [`stencil.py`](../garnet/stencil.py): inverse diagonal; [`linalg.py`](../garnet/linalg.py): Jacobi/Neumann preconditioning | Diagonal scaling of group solves already exists. Adding another Jacobi normalization is not a new algorithmic acceleration. |
| [`transient.py`](../garnet/transient.py): unit-total initial fission source and critical adjustment by `k0` | Flux magnitudes are reduced relative to physical-power flux, and initial equilibrium is consistent. Critical adjustment is an initialization convention, not conditioning optimization. |
| [`solver.py`](../garnet/solver.py): power-iteration source normalization; [`quasistatic.py`](../garnet/quasistatic.py): amplitude/shape normalization and precursor reprojection | Eigenvector amplitudes and IQS coordinates are already managed explicitly. |
| [`stencil.py`](../garnet/stencil.py): cylindrical metric weighting and matching `rhs_weight` | Equation scaling already preserves symmetry for PCG. Arbitrary row scaling could undo it. |
| [`multigroup.py`](../garnet/multigroup.py): independent forward/test energy-mode normalization and coarse condition checks | Small coarse projection systems are scaled; this is not general production equilibration of the full system. |
| [`timescheme.py`](../garnet/timescheme.py): BDF time nodes normalized by trial width | The coefficient Vandermonde system is already dimensionless. |
| [Mixed-precision experiment](../benchmark-results/transient-acceleration/step-04a-mixed-precision-preconditioning.md) | FP32 preconditioning was accurate but 4.7% slower than equivalent FP64 on the recorded GPU. Full FP32 was slower and required more Krylov work. Scaling has not been shown to reverse that result. |
| [`Material`](../garnet/materials.py), [`Grid`](../garnet/grid.py), [`thermal.py`](../garnet/thermal.py) | Public conventions are cm, seconds, cross sections in 1/cm, temperature in K and power in watts. A broad conversion would touch boundaries, feedback, coupling and I/O. |

No systematic full-state nondimensionalization or general multigroup row/column
equilibration study was found in current sources, documentation or accessible
commit messages. This is a scoped audit result, not proof that it was never
explored outside the repository.

## Three distinct numerical questions

**Changing units.** For `du/dt = J u`, setting `u = S u_hat` and
`t = t_ref tau` gives `d u_hat/dtau = t_ref S^-1 J S u_hat`. A fixed
invertible state scaling preserves eigenvalues up to the common time factor.
It does not remove prompt/delayed timescale separation, fine-mesh diffusion
stiffness or a near-critical physical mode. Scalar scaling `A -> c A` also
leaves `cond(A)` unchanged. Scaling can improve floating-point range or
representation, but coefficients near one do not automatically add precision.

**Balancing a linear system.** Nonuniform equation/variable scales can change
singular-value conditioning and the norm seen by a Krylov solver. For the
nonsymmetric multigroup system, a possible experiment is

```text
x = D_c y,
A_hat = D_r A D_c,    b_hat = D_r b,
r_physical = b - A x.
```

Map preconditioners consistently: if `B` approximates `A^-1`, use
`B_hat = D_c^-1 B D_r^-1`. Pure column scaling with an exactly mapped right
preconditioner leaves `A_hat B_hat = A B`, so it supplies no automatic
FGMRES gain. Fixed coordinate changes likewise do not improve the contraction
eigenvalues of an otherwise equivalent fixed-point iteration.

For SPD group solves retain symmetry: use a congruence `S A S` with positive
diagonal `S`, or keep `A` and apply an SPD preconditioner. Ordinary CG on an
arbitrary left-scaled `D_r A` is not valid in general. Symmetric diagonal
normalization is closely related to GARNET's existing Jacobi PCG;
[PETSc documents both forms](https://petsc.org/release/manualpages/PC/PCJACOBI/).

**Balancing error control.** This was the clearest unresolved opportunity. *Status 2026-10-29 (audit O11):* adaptive BDF now defaults to `error_norm="blockwise"`, the simpler dimensionless variant suggested below: Eq. (44) per flux group, precursor family and external field, accepting on the largest. `"concatenated"` preserves the legacy mode. The volume-weighted componentwise RMS and `T - T_ref` thermal scaling remain unimplemented; see [adaptive-bdf-error-norm](../benchmark-results/adaptive-bdf-error-norm/) for the measured effect.
`BDF.error_norm()` computes

```text
E = sqrt(sum_blocks ||corrected - predicted||_2^2
         / sum_blocks ||atol + rtol abs(corrected)||_2^2).
```

`TransientSolver.solve()` concatenates all flux groups and precursor families.
The implicit LRA driver in `benchmarks/lra.py` also appends temperature.
This deliberately follows the paper-style ratio of norms, not a componentwise
weighted RMS. Larger-magnitude fields can dominate, and one scalar absolute
tolerance cannot express separate physical budgets for flux, precursors and
temperature. The generic split thermal coupler does not automatically append
temperature; the issue applies where `adaptive_error_state` supplies it.

For a simple source-formula probe at `rtol=1e-3`, corrected `(phi,C)=(1,1)`
and predicted `(0.99,1)` give `E=7.071`: rejection. Express the same precursor
component in units 1000 times smaller, giving stored values `(1,1000)` and
`(0.99,1000)`, and `E=0.010`: acceptance. The flux defect is still 1%.
This is concrete scale sensitivity, **not evidence that a recorded GARNET
application is wrong**. Its significance on actual trajectories needs
measurement. Common scaling of every field cancels at `atol=0`; relative
block scaling is what matters here.

The probe is preserved in
[analytic-probes.json](../benchmark-results/transient-numerics-assessment/analytic-probes.json).
SUNDIALS supplies an established alternative with componentwise absolute
tolerances and a weighted RMS norm; see its
[CVODE tolerance interfaces](https://sundials.readthedocs.io/en/v7.2.1/cvode/Usage/index.html).
Changing the norm is an intentional controller change. Preserve the legacy
mode used to reproduce the Cherezov/LRA comparisons.

## Recommended design

Keep public physical units and initially keep physical state arrays.
Introduce explicit reference scales and tolerance metadata at solver seams,
instead of converting every material, geometry and feedback formula.

For each flux group, precursor family and optional thermal field, form

```text
d_b,j = atol_b,j + rtol_b max(abs(u_old,b,j), abs(u_new,b,j)),
E_b^2 = sum_active V_j ((u_new,b,j - u_pred,b,j)/d_b,j)^2
        / sum_active V_j,
E = max_b E_b.
```

This blockwise weighted RMS with a maximum across blocks is a proposed
opt-in policy. It prevents one accurately predicted large field from diluting
another block's error, though local peaks still need observable checks.
Cell volumes and masks make its meaning stable under geometry/mesh changes.
Compare against a simpler fixed-reference, dimensionless ratio of norms.

Choose group/family references from the initial state and physically
meaningful floors. Set flux/precursor absolute tolerances relative to those
references and document behavior in initially dark regions. Never divide by
an initially zero field. Freeze scales throughout a trial step and its
retries; acceptance must not become easier just because a trial is large or
inaccurate. Use the same policy for order selection and external state.

For temperature an absolute increment tolerance in K is often clearer than
relative error on a 600 K background. If relative thermal error is needed,
base it on `T - T_ref`, transforming the reference with any change in
temperature convention. Physical feedback must still receive its expected
absolute temperature. This avoids making an arbitrary temperature zero part
of the error budget.

Report original physical linear residuals alongside any scaled residual.
A small global norm is insufficient if a group, local power peak or neutron
balance remains inaccurate. Diagonal ranges alone do not characterize
near-critical sensitivity.

## Potential speed benefit

| Proposal | Additional speed potential |
|---|---|
| Uniform time/length/flux units | Approximately none in exact arithmetic with corresponding steps/tolerances. Do not budget a speedup. |
| Another symmetric diagonal normalization for PCG | Little reason to expect gain over existing Jacobi/Neumann; establish equivalence before implementing. |
| Field-aware adaptive error control | Unknown sign: it can remove accidental over-resolution, but can also require **more** steps where the old norm hid error. Accuracy/unit invariance justify it independently. |
| Block residual weighting or equilibration for FGMRES | Plausible modest gain if norms/finite precision are unbalanced; no quantitative GARNET result yet. Existing energy-sweep preconditioning limits headroom. |
| Scaling to enable FP32 | Conditional future experiment. Existing FP32 losses argue against assuming a gain; conversion launches and additional iterations still cost time. |

If neutronics is 97% of march time, a measured 20% reduction in its cost
would give `1/(0.03 + 0.97*0.8) = 1.24x` overall. This is a hypothetical
cost calculation, not a predicted 20% reduction. A large generic speedup
from nondimensionalization is unsupported by the available evidence.

## Separate development plan

| Phase | Concrete work | Gate and deliverable |
|---|---|---|
| S0: diagnostics | Add opt-in telemetry in `timescheme.py`/`transient.py`: per-group/family/thermal state and defect norms, denominator shares, physical/weighted residuals, work and setup. Sample operator/source ranges. Audit LRA and generic coupling adapters. | Identify blocks that drive acceptance and errors that may be hidden. Preserve decisions, grids, answers and work; no per-iteration host transfers. |
| S1: error scales | Add optional references, per-block tolerances, volume/mask-aware norms and a named legacy norm. Thread identical controls through candidate orders and external adaptive state in `benchmarks/lra.py` and the solver API. Leave published comparisons on legacy controls. | Unit/normalization invariance, zero-field behavior, rollback and mesh-weighting checks; physical calibration against refined references. Useful even without a speed gain. |
| S2: diagnose the block solve | On small actual Cartesian/triangular HP-MR systems use `assemble()` only for analysis. Examine active unknowns, residual balance, singular-value/condition estimates, nonnormality, preconditioned work and tolerance sensitivity across widths. Compare existing Jacobi and energy sweeps. | Demonstrate an imbalance persisting after current preconditioning before adding a scaling path. Raw coefficient spread or unpreconditioned `cond(A)` alone is insufficient. |
| S3: minimal equilibration | If justified, try fixed per-group reference/row scales in FGMRES first. Compare equation weighting with column-only scaling. Map inverse preconditioners, guesses, RHS and residual targets consistently; only then consider spatial/material scales. Preserve symmetry in PCG paths. | Same time grid and equivalent physical stopping targets; power/final normalized flux meet the existing `5e-8` equivalence gate; work/time improve. Looser physical residuals do not count as acceleration. |
| S4: GPU decision | Precompute scales outside iterations. Fuse multipliers into existing stencil/contraction/vector kernels where possible, reuse buffers and preserve graph/workspace contracts. Repeat at matched physical error. | Seek at least 10% median march improvement on two cases for a performance-only feature; retain raw samples and reject memory/launch overhead that erases gains. This is a proposed retention gate, not a forecast. |
| S5: optional precision | Only if diagnostics identify a range/roundoff problem, revisit FP32 shadow preconditioning with FP64 residuals and fallback. Consider complete dimensionless-state storage only for a measured remaining problem. | Beat equivalent-degree FP64, not a weaker baseline, at the same physical accuracy; otherwise retain FP64. |

S0–S1 should take several focused days plus benchmark calibration. Conditional
equilibration and GPU/precision work can take another week or more. A full
units rewrite has much larger scope and is not recommended now. These are
planning estimates, not measured implementation times.

## Verification and benchmark contract

- **Coordinates/units:** rescale one group/family and its physical tolerances
  consistently; change common flux normalization; use K/mK and shifted
  temperature coordinates with a matching reference. New decisions should
  agree to rounding. Preserve existing decisions in selected legacy mode.
- **Cases:** homogeneous equilibrium/point kinetics, TWIGL, strong-feedback
  LRA, 11-group HP-MR smooth/abrupt insertions, asymmetric drums,
  heterogeneous/blended kinetics, delayed spectra, initially zero precursor
  and nonfissile regions, cylindrical/triangular geometry, falling power.
- **References:** time refinement and tightly converged spatial solves;
  maximum power/shape error, local peak/time, energy, precursor emission and
  temperature error. Calibrate field tolerances to agreed physical budgets
  before speed comparisons. A block LTE tolerance is not itself an
  observable-error guarantee.
- **Isolation:** diagnostics, controller, algebraic representation and
  precision are separate experiments. Fix the time grid for algebraic
  comparisons; match resulting physical error for controller comparisons.
- **Timing:** warmed/synchronized GPU, at least three interleaved samples,
  accepted/rejected work, setup, memory and march time. Follow the
  [existing protocol](transient_performance_plan.md). No production-sized
  matrix is assembled or transferred to the host.

Start **S0 then S1 alongside E0/E1 of the FTM plan**. Keep both workstreams
separately benchmarkable. Continue to S2–S5 only where evidence warrants it;
FTM does not depend on wholesale nondimensionalization.

## Evidence and reproduction limits

Reproduce the independent norm-formula and point-kinetics probes with:

```bash
python3 tools/assess_transient_numerics.py \
  --output benchmark-results/transient-numerics-assessment/analytic-probes.json
```

They use only the standard library and passed internal consistency checks.
They do not execute GARNET. At this initial assessment the environment lacked
NumPy, SciPy and CuPy, and no new solver conditioning or performance result
was claimed. NumPy/SciPy were subsequently installed for the separate
[FTM prototype study](../benchmark-results/exponential-prototype/README.md).
Systematic scaling experiments and GPU measurements remain future work.
