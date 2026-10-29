# Setup reuse closed; PARCS precursor control (E5a): eighth checkpoint

2026-10-28. This checkpoint closes the E2 setup-reuse item by measurement and
implements the E5a precursor experiment from the
[development plan](../../docs/exponential_transformation_plan.md): FTM2 with
PARCS-style quadratic **physical** production integration, compared at
identical transformed-theta neutron equations. The default FTM2 trajectories
are bitwise unchanged. No heat equation is transformed.

## 1. Setup reuse: measured, not implemented

Every change of width, frequency or order rebuilds the group stencils, their
Neumann/mixed preconditioners, `MultigroupStepOperator` and the energy-sweep
preconditioner. On TWIGL this happens about once per solve for every scheme.
Before separating the time diagonal from the geometric stencil, the setup
share of march time was profiled (cProfile, CPU, single runs on a shared node;
step case, adaptive rtol 1e-4, FTM2 history estimator with output_rtol 1e-4,
BDF1-5 automatic order; setup is the cumulative time of `_step_operators`,
`MultigroupStepOperator.__init__` and `EnergyGroupGaussSeidelPreconditioner.__init__`):

| TWIGL mesh | Method | March time | Solves | Rebuilds | Setup share | FGMRES share |
|---|---|---:|---:|---:|---:|---:|
| 40×40 | adaptive FTM2 | 14.9 s | 271 | 272 | 0.93% | 97.8% |
| 40×40 | adaptive BDF | 6.9 s | 107 | 92 | 0.69% | 98.2% |
| 160×160 | adaptive FTM2 | 172.6 s | 271 | 272 | 0.21% | 99.4% |
| 160×160 | adaptive BDF | 82.2 s | 108 | 93 | 0.15% | 99.6% |

Setup is O(cells) per rebuild while each solve costs O(cells × iterations), so
the share falls with refinement. Eliminating every rebuild would bound the
gain near 1.01x, far below the 1.2x promotion gate, while caching a stencil
across dt/frequency changes would add a stale-coefficient risk to the hot
path. **E2 caching is closed without implementation.** FTM's deficit is the
number of solves, not their setup: per solve, FTM2 cost 0.63 s and BDF 0.76 s
at 160×160. GPU setup was not measured; one rebuild is roughly 150-200 kernel
launches against thousands per solve, so a large GPU share is not expected.

At 160×160 the chunk-7 setting `multigroup_kwargs={"rtol": 1e-11}` is below
the FGMRES floor (residual 1.14e-15 against a 1.11e-15 target) and aborts the
march. A finer transient mesh study needs about 1e-10.

## 2. Implemented control

`ftm_kwargs={"precursor_production": ...}` on `time_scheme="ftm2"` selects:

| Value | Production model over one step | Frequency dependence |
|---|---|---|
| `"linear-transformed"` (default) | linear in the transformed frame, two endpoints | yes |
| `"linear-physical"` | linear in physical time, two endpoints | none |
| `"quadratic-physical"` | quadratic in physical time through the previous, current and new accepted endpoints (PARCS v2.6 Eqs. 2.23-2.27) | none |

With h the width, r the ratio of the previous accepted width to h, x = lambda*h
and K_m = int_0^1 exp(-x u) u^m du (u = fraction of the step remaining),

```text
C_new = exp(-x)*C_old + older*(beta*S)_older + old*(beta*S)_old + new*(beta*S)_new
new   = h*((1+r)*K0 - (2+r)*K1 + K2)/(1+r)
old   = h*((1+r)*K1 - K2)/r
older = h*(K2 - K1)/(r*(1+r))          (negative)
1 - lambda*new = ((2+r)*K0 - 2*K1)/(1+r)
```

All K_m are positive; a series (x <= 2) or the stable forward recurrence
K_m = (m*K_(m-1) - exp(-x))/x (x > 2) evaluates them, so stiff families
avoid the cancellation of equivalent phi-function forms. Mapped delayed
fractions enter as **products** at each node. The quadratic history starts
and restarts as linear-physical, never spans a declared event, and records
physical production during BE startup/fallback steps. Rollback is complete
because the scheme state, including the older node, is restored with the
existing adaptive snapshot. `result.step_precursor_productions` and each
adaptive trial diagnostic report the model actually used.

Verification: five new scalar tests in the dependency-free coefficient
file (16 there in total), with 90-digit references that include a Lagrange
basis integrated in s-monomials, independent of the u-moment formulas, up
to x = 1e12, and 26 solver tests: exact three-node mapped
production products, zero-frequency identity of the two linear models,
second order for nonequilibrium point kinetics and a two-family changing
fission source against Radau, mapped-kinetics equilibrium, groupwise versus
monolithic agreement, startup/restart/fallback sequences, accepted-half-step
replay after rejections and six-family adaptive Radau checks. Three default
TWIGL ramp trajectories (two adaptive estimators and refined fixed widths)
are **bitwise identical** to the committed source.

## 3. Matched physical comparison

The [assessment](chunk8-precursor-production.json) runs nine cases with
independent references: six-family point kinetics (+0.3$ and +0.8$ steps,
-5$ shutdown, 0.5$ ramp over 0.2 s; exact 7×7 matrix exponentials, Radau over
the ramp), a localized two-group eight-cell slab with six families, distinct
delayed spectra and a +0.40$ left-quarter thermal step (exact 48×48 matrix
exponential), and TWIGL step/ramp on 20×20 and 40×40 against the chunk-7
physical-generator Radau reference. Power integrals use augmented matrix
exponentials or Gauss integration of Radau's cubic segments.

Each production model runs adaptive FTM2 (history estimator; rtol 1e-3/1e-4,
plus 1e-5 off TWIGL; output_rtol 1e-4 and 1e-3) and shared graded meshes with
lagged and zero frequency. Point kinetics and the slab add an untuned adaptive
BDF sweep (cap 0.05 s). TWIGL reruns the chunk-7 minimum-solve BDF selection
instead; it reproduces chunk 7's solve and Krylov counts exactly, as do the
default FTM2 configurations shared with chunk 7. All 199 runs keep final
precursors nonnegative.

**Production interpolation does not change the work.** In 44 of 46 adaptive
configuration groups all three models accept the same steps with identical
solve counts, and their maximum power errors agree within 1%; the other two
differ by three solves and 6% in power error. Only final precursor errors move,
in a regime-dependent direction:

| Case (adaptive rtol 1e-4, output 1e-4) | Solves (all models) | Maximum power error | Final precursor error: linear-transformed / linear-physical / quadratic-physical |
|---|---:|---:|---:|
| +0.3$ step | 250 | 1.108e-04 | 1.79e-06 / 1.20e-06 / **3.19e-07** |
| +0.8$ step | 233 | 9.345e-05 | **4.57e-06** / 1.54e-05 / 2.01e-05 |
| -5$ shutdown | 530 | 1.213e-04 | 4.80e-07 / 6.08e-07 / **1.92e-07** |
| 0.5$ ramp | 214 | 6.707e-05 | 1.19e-06 / **1.07e-06** / 2.23e-06 |
| Localized slab | 277 | 7.635e-05 | 1.40e-06 / **1.11e-06** / 2.11e-06 |
| TWIGL step 40×40 | 271 | 7.147e-05 | **1.31e-09** / 1.02e-08 / 1.18e-07 |
| TWIGL ramp 40×40 | 180 | 8.696e-05 | 1.18e-07 / 2.24e-07 / **7.71e-08** |

Precursor errors are relative maxima over families (RMS over cells for spatial
cases). Quadratic physical production helps slow insertions and shutdowns
(25x at rtol 1e-5 in shutdown), and loses up to two orders of magnitude where
production is nearly exponential, as the transformed model is exact there.
On shared lagged meshes, quadratic production is worse in seven of nine cases
(e.g. 1.68e-05 versus 2.45e-06 on the slab; the largest is 2.97e-04 on the
+0.8$ step). In the adaptive runs every precursor error is at least 50x below
the 0.1% budget: production integration never limits the controller.

**E5a gate outcome: no physical error/work benefit.** The option stays an
opt-in research control, as the plan prescribes before an option is retained.
It is the only route in the code to the PARCS frequency-independent scalar
endpoint source that E5b would need for regional neutron frequencies.

## 4. Correction to the chunk-7 matched-work comparison

Chunk 7 tuned BDF width caps but held FTM's `output_rtol` at 1e-4, which kept
FTM2 about ten times inside the power budget. Allowing output_rtol 1e-3 changes
the 0.1% (all seven chunk-7 categories) minimum-solve comparison:

| TWIGL case | Best FTM2: solves / group-PCG iterations | Chunk-7 BDF: solves / iterations | Chunk-7 best adaptive FTM2 solves |
|---|---:|---:|---:|
| Step 20×20 | 202 / 59,614 | **109 / 37,221** | 257 |
| Ramp 20×20 | **87 / 39,032** | 92 / 41,310 | 180 |
| Step 40×40 | 202 / 119,688 | **138 / 85,431** | 257 |
| Ramp 40×40 | 93 / 84,989 | **92 / 84,233** | 180 |

The ramp gap closes: FTM2 is 5% cheaper on 20×20 and within 1% on 40×40. The
step remains 1.46-1.85x cheaper with BDF; its FTM2 rtol 1e-3/output 1e-3
candidate fails only the regional power gate (1.34e-3). The ramp FTM2 margin
is thin (power error 9.05e-4 and 8.03e-4). Neither method has been tuned
exhaustively, so a 5% difference is not a ranking and nothing approaches the
1.2x promotion gate. Off TWIGL, FTM2 needs 82-266 solves; the BDF sweep there
has no width caps and misses the budget on two point-kinetics cases, so those
rows are not a fair comparison and no FTM advantage is claimed.

Shared-mesh controls again show no uniform benefit from the lagged frequency:
zero frequency is more accurate on the +0.8$ step, the shutdown and the slab.

## 5. After merging `audit-fixes`

This checkpoint was prepared for a merge with `audit-fixes` (`aa266f7`, which
includes the readability refactor `5f5be77`). The audit left the exponential
transformation out of scope. On the merged tree every FTM2 record of the
assessment is bitwise identical ([post-merge report](chunk8-postmerge-precursor-production.json)).
The audit's fix H12, an explicit-Euler startup estimator for adaptive BDF,
removes most early rejections and so changes the comparator. Repeating
chunk 7's BDF search on the merged tree (rtol 1e-3/1e-4/1e-5 at a 0.02 s cap;
caps 0.01 and 0.005 at rtol 1e-3/1e-4; same seven-category 0.1% gate) gives
the [post-merge BDF selections](chunk8-postmerge-bdf.json):

| TWIGL case | Best FTM2: solves / iterations | BDF before audit | BDF after audit |
|---|---:|---:|---:|
| Step 20×20 | 202 / 59,614 | 109 / 37,221 | **108 / 36,885** |
| Ramp 20×20 | **87 / 39,032** | 92 / 41,310 | 101 / 44,466 |
| Step 40×40 | 202 / 119,688 | 138 / 85,431 | **137 / 83,443** |
| Ramp 40×40 | **93 / 84,989** | 92 / 84,233 | 101 / 90,759 |

On the ramp, the former 92-solve selection (rtol 1e-3, cap 0.01) now takes
76 solves but misses the power budget by 0.7% (error 1.0065e-3 and 1.0055e-3),
so the cheapest qualifying tested BDF run needs 101. That ranking rests on a
near miss, so the ramp remains a tie within the tuning resolution, not an FTM2
advantage. BDF stays 1.47-1.87x cheaper on the step. Off TWIGL, the untuned
BDF sweep on the localized slab falls from 1356 to 101 solves (14,657
iterations), now below FTM2's 138 (20,329); point-kinetics BDF needs 77 (+0.3$
step) and 104 (0.5$ ramp) solves against FTM2's 151 and 82.

## Reproduction and evidence

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_exponential_precursor_production
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest tests/verification/test_exponential_precursor_production.py tests/verification/test_exponential_coefficients.py -q
```

The [report](chunk8-precursor-production.json) contains every configuration,
error category and work counter, with source hashes. The setup profile is a
throwaway measurement recorded in the table above. The
[default CPU suite](chunk8-cpu.xml) gives **951 passed, 26 skipped, 13
deselected (slow) and 1 failed**: the example `--help` subprocess in
`test_examples_current.py` exceeded its 10 s timeout while the assessment ran
concurrently. That import takes 8-12 s on this node even unloaded; the
[separate retest](chunk8-examples-retest.xml) passes all four example tests.
The run used Python 3.12, NumPy 2.3.5 and SciPy 1.16.3 on the merlin7 CPU
node. GPU execution was not run.

On the merged tree, the [default CPU suite](chunk8-postmerge-cpu.xml) gives
**1240 passed, 26 skipped, 13 deselected and 1 failed**, the same example
timeout: that cold import took 11.4 s against the 10 s limit with no other
load, and the [retest](chunk8-postmerge-examples-retest.xml) passes 4/4.
`audit-fixes` alone failed three triangular-mesh tests in
`tests/validation/test_noise_point_kinetics.py` (k = 0.52-0.54 instead of 1):
its fix H9 makes mask-free tri lattices apply `bc` on their outer edges, and
these tests relied on the old implicit reflection. They now request
`bc="reflective"` explicitly and pass both before and after the merge.

**Next:** with production interpolation ruled out as a lever and setup cost
negligible, FTM's remaining TWIGL deficit is the step case's solve count. Tune
both comparators along the same output-width dimensions (BDF caps and FTM
output_rtol) before any further frequency-model work, and run the 160×160
transient study with a reachable linear tolerance.
