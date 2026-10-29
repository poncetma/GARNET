# Verification and validation

GARNET's QA combines exact mathematical checks, reactor benchmark comparisons,
and studies of coupled transients and GPU execution. This report links the
evidence to the core features exposed in the [README](../README.md). Test
assertions are the executable acceptance criteria; benchmark reports retain
the mesh, time step, physics data, and hardware behind their results.

## Physical-state adaptive exponential check

The [eighth checkpoint](../benchmark-results/exponential-second-order/CHUNK8.md)
adds FTM2 `precursor_production` controls (PARCS-style quadratic and linear
physical production). Five new scalar coefficient tests use 90-digit
references and 26 solver tests use exact products, Radau and matrix-exponential
references; default FTM2 trajectories are bitwise unchanged. A 199-run
assessment over nine cases finds no error/work benefit: all production
models accept the same steps, while final precursor errors move by regime and
stay at least 50x inside the 0.1% budget. Setup reuse was closed by
measurement (below 1% of CPU march time). With FTM `output_rtol` also tuned,
FTM2 ties tuned BDF on the TWIGL ramp; BDF stays cheaper on the step. The
default CPU suite records 951 passed, 26 skipped and one example-import
timeout under concurrent load, which passes on retest. Merged with
`audit-fixes`, FTM2 results are bitwise unchanged and the suite records 1240
passed, 26 skipped and the same import timeout; the tri noise tests now state
their reflective boundary explicitly, as audit fix H9 requires.

The [seventh checkpoint](../benchmark-results/exponential-second-order/CHUNK7.md)
assesses TWIGL step/ramp on 20×20 and 40×40 meshes, with independent sparse
physical-generator references and a finer spatial check. **118 targeted
checks pass**; the later 12-case independent-reference check overlaps.
The 116-configuration comparison retains output-width-tuned adaptive BDF as
the cheaper tested method at the common 0.1% temporal power/shape/region/
integral/final-precursor budget. A 40×40-to-80×80 thermal-shape difference
of **1.10%** exposes a separate unresolved spatial budget. The
[QA record](qa/garnet-exponential-twigl-2026-10-26.json) links the references,
mesh diagnosis, tests and combined work selections. This assessment does
not pass spatial-accuracy or performance-promotion conditions; full-suite
and GPU records below remain earlier evidence.

The [sixth checkpoint](../benchmark-results/exponential-second-order/CHUNK6.md)
adds an opt-in two-solve physical-history estimator and a reference audit
mode. **363 distinct CPU checks pass** across a 361-case regression and two
smooth-insertion stress cases; a separate 101-case focused batch overlaps.
At the 0.1% power/integral/precursor budget, tested history configurations
use **31.5–33.1% fewer solves** than full/fine FTM, while adaptive BDF still
uses less work. Audits cover 20,410 paired output-aware trials and 5,482
state-only/smooth-insertion trials, including 13 failed reference state
checks that the history estimate also flags. Tight exponential tails can
make this physical predictor more restrictive. The default stays step
doubling, output control remains independent, and no runtime speedup is
claimed. The [QA record](qa/garnet-exponential-estimator-2026-10-26.json)
links the work/error studies, tests and isolated archive validation.

The [fifth checkpoint](../benchmark-results/exponential-second-order/CHUNK5.md)
adds optional physical midpoint/chord width control for linear output.
**314 regression checks pass**, and an updated **58-case focused batch**
passes with four additional output-aware physical-reference configurations;
the batches overlap. The rapid-rise FTM2 common-grid power error falls from
**2.20658% to 0.006738%** at unchanged state tolerance, and its power-integral
error falls from **0.910182% to 0.004572%**. The 160-run comparison still
finds adaptive BDF cheaper at matched power/integral/precursor accuracy.
Verification covers constant-power spectral redistribution, localized
multigroup output, six families and output-driven rollback on three
geometries. The [QA record](qa/garnet-exponential-output-2026-10-26.json)
links the tests, benchmark and isolated source-archive check. Full-suite,
slow-study and GPU results below remain the preceding checkpoint's records.

On 26 October 2026, **259 targeted checks passed**, including 33 new adaptive
FTM cases. The default CPU run recorded **801 passed, 12 skipped, 10 slow
cases excluded**, with one example subprocess import failure because this
checkout was not installed. All four example checks passed on retest with
the absolute checkout path in `PYTHONPATH`; no example code changed. Across
the run and retest, all **802 distinct executable CPU checks pass**. The
original full-run failure remains in the evidence rather than being hidden.

The new `adaptive_ftm` controller compares physical full/two-half-step flux
and precursor fields, commits the fine state once per macro endpoint, and
restores all temporal history on rejection. Verification includes exact
accepted-half-step replay on three geometries, six-family Radau references,
and a localized two-group/two-family diffusion matrix exponential. The
masked-grid checks also led to explicit inactive-state/derivative masking
for FTM population histories.

The [fourth checkpoint](../benchmark-results/exponential-second-order/CHUNK4.md)
records 80 accuracy/work configurations. Adaptive BDF remains the cheapest
tested method at matched output accuracy. Long exponential steps expose a
separate linear-output-interpolation limit; neither a speedup nor a coupled
thermal accuracy claim is made. GPU execution was not repeated.

The refreshed source archive was installed into an isolated temporary
directory and passed an adaptive cylindrical event/rejection/callback check.
Evidence: [focused tests](../benchmark-results/exponential-second-order/chunk4-verification.xml),
[original full CPU run](../benchmark-results/exponential-second-order/chunk4-cpu.xml),
[example retest](../benchmark-results/exponential-second-order/chunk4-examples-verification.xml),
and [environment, source fingerprints and archive check](qa/garnet-exponential-adaptive-2026-10-26.json).

## Exponential damping check

On 26 October 2026 the default CPU suite passed: **777 passed, 0 failed,
7 skipped, 13 slow cases excluded**, in 687.55 s. All **43 new method checks**
also passed in a final focused run. A later source edit only extends a
docstring; the executable AST is unchanged. GPU execution and the excluded
slow studies were not repeated.

The experimental `ftm-bdf2` recurrence transforms neutrons and precursors in
the current step's frame, using one endpoint solve before any requested
frequency refinement. The [damping checkpoint](../benchmark-results/exponential-second-order/CHUNK3.md)
records its derivation, guards and 196 error/work configurations. Adaptive
BDF remains the strongest tested comparator; no runtime speedup or positivity
guarantee is inferred from stiff damping.

The [new tests](../tests/verification/test_exponential_damping.py) compare
Cartesian/r-z/triangular solves with an independent matrix recurrence, check
second-order refinement and six-family non-equilibrium precursors against
Radau, and verify distinct delayed spectra against a matrix exponential.
They also cover mapped material kinetics, BE fallbacks, frequency-corrector
accounting, event continuity, a stiff shutdown mode and unchanged thermal
exchanges/progress callbacks. The two-stage SDIRK2 candidate stays a dense
research driver because its extra stages did not earn an FV implementation.

The refreshed Colab source archive was installed in isolation. It ran an
r-z transient with an event restart, reproduced ordinary BDF2 at zero
frequency, and integrated a power-density field to 2 MW.

Evidence: [full pytest record](qa/garnet-exponential-cpu-2026-10-26.xml),
[new verification record](../benchmark-results/exponential-second-order/chunk3-verification.xml),
and [environment, source fingerprints and archive checks](qa/garnet-exponential-damping-2026-10-26.json).

## Transient API and cylindrical check

The transient API follow-up on 23 October 2026 passed the default CPU suite:
**666 passed, 0 failed, 7 skipped, 13 slow cases excluded**, in 714.27 s.
The final focused run passed **105 cases**, including three additional
parameterized guards for public steady-state reuse with untracked spatial
cross-section hooks. The full-suite snapshot predates that guard; the final
focused run covers both changed Python files. The optional-dependency and
hardware skips are the same as in the earlier CPU record below.

The [transient workflow checks](../tests/verification/test_transient_model_api.py)
exercise common Cartesian/cylindrical/triangular reports, named material and
cached-state controls, event alignment, adaptive-step rejection, accepted-step
callbacks, and time-zero compatibility. Cylindrical checks include exact
annular volumes, full-core power integration, and full/quasi-static thermal
coupling. See the [transient API guide](transient_api.md) and
[r-z example](../examples/cylindrical_transient.py).

Six existing modules now exercise the public API where it expresses the same
physics calculation. All **58 test functions and 149 assertions** were
preserved. Their original files are retained byte-for-byte in
[`tests/legacy/`](../tests/legacy/), inactive by default; all **62 legacy cases
passed** separately in 168.81 s with `--run-legacy`.

The refreshed Colab source archive was installed in isolation and ran the r-z
example with fixed and adaptive BDF, including off-grid event times and a
power density integrating to 2 MW. CUDA execution and slow studies were not
repeated. These checks establish API and numerical regression coverage; the
demonstration cross sections are not a validated reactor design.

Evidence: [full pytest record](qa/garnet-transient-cpu-2026-10-23.xml),
[preserved legacy record](qa/garnet-transient-legacy-2026-10-23.xml), and
[environment, source fingerprints, focused results and archive checks](qa/garnet-transient-api-2026-10-23.json).

## Diffusion API check

The diffusion API audit on 23 October 2026 added 44 workflow cases. The full
default CPU suite passed: **639 passed, 0 failed, 7 skipped, 13 slow cases
excluded**, in 712.10 s. A final focused run passed all **75 cases**, covering
the original model/TWIGL checks, new workflows, example imports, and the last
mesh-boundary input guard. The skips are the same optional dependencies,
external mesh inputs and CUDA cases listed in the earlier CPU record below.

The new checks verify named material replacement, constructor-option forwarding,
power integration on all three model types, snapshot consistency, Cartesian
thermal feedback and coupled equilibrium, invalid-input errors, and a one-cell
Gmsh mesh. Acceptance criteria and remaining API gaps are in the
[audit report](api_audit.md); the
[workflow tests](../tests/verification/test_diffusion_workflow_api.py) are the
executable specification.

The refreshed `dist/garnet-src.zip` was installed in isolation. It reproduced
the README eigenvalue `1.1352040511640966`, the `(2, 24, 24, 24)` flux shape,
and a power field integrating to 2 MW, and ran the new Cartesian coupled
example successfully. Syntax was checked in 177 Python files. GPU execution
and the excluded slow studies were not repeated.

Evidence: [full pytest record](qa/garnet-api-cpu-2026-10-23.xml) and
[environment, source fingerprints, focused checks and archive validation](qa/garnet-api-2026-10-23.json).
The record distinguishes the full-suite source snapshot from the two files
with the subsequent boundary guard and its final focused checks.

## GARNET rename check

The renamed checkout was checked on 23 October 2026 in a local staging
directory. **595 passed, 0 failed, 7 skipped, and 13 slow cases were
excluded** in 774.36 s. The earlier example-launch timeout did not
reproduce in this run; its original record is retained below. The staged
files are the ones applied to the renamed checkout.

The checks also covered all 17 Colab notebooks (111 code cells parsed), syntax
in 180 Python files and 11 shell scripts, and installation from
`dist/garnet-src.zip`. The installed package loaded its bundled 11-group
cross sections, honored `GARNET_FUSED=0`, and ran the README example with the
same eigenvalue and flux shape as before. The library's 56 source files
contain only the expected name substitutions.

The LRA figures and PDF were rebuilt from the preserved numerical records.
Historical JSON, CSV, cross-section data, and earlier raw QA records were
checked for byte-for-byte preservation. Hardware GPU studies were not rerun.
See the [rename validation record](qa/garnet-rebrand-2026-10-23.json) for the
source fingerprint, environment, archive checks, and per-module counts, and
the [pytest record](qa/garnet-cpu-2026-10-23.xml) for individual cases.

## Earlier CPU check

Checked on 23 October 2026 against revision `9241d54`. Pytest collected 614
cases, selecting 601 across 57 test modules: 511 verification cases and 90
validation cases. The configured default excluded 13 expensive cases.

**Result: 594 passed, 1 failed, 7 skipped, 13 deselected in 838.82 s.**

The failure is
[`test_cross_example_import_works_from_documented_entry_point`](../tests/verification/test_examples_current.py):
its subprocess running `hpmr_adaptive_coupled_benchmark.py --help` exceeded the
10 s timeout. A focused recheck reproduced the timeout. Direct invocation of
the same command without that limit completed successfully. The numerical
checks reported no failures; the example-launch timeout remains an outstanding
QA failure in this recorded run.

The seven skips comprise unavailable preCICE bindings, two unavailable
external VVER mesh cases, two CUDA-device kernel cases, an optional `pyamg`
case, and a CUDA-less error-path test that skips when CuPy is installed.
The preCICE skip occurs at module import, so pytest reports it in addition to
the 601 selected test cases. Slow cases and GPU studies were not rerun here.

The environment was Linux x86_64, Python 3.12.12, NumPy 2.3.5, SciPy 1.16.3,
and pytest 8.4.2, with single-threaded OpenMP/OpenBLAS and pytest plugin
autoload disabled. The five FEMFFUSION cross-section reader tests used local
reference inputs through an `examples/` symlink to the checkout's `Examples/`
directory. The separate fixed VVER mesh path was absent.

Artifacts: [environment and per-module results (JSON)](qa/cpu-2026-10-23.json),
[full pytest record (JUnit XML)](qa/cpu-2026-10-23.xml), and
[focused timeout recheck (JUnit XML)](qa/cpu-example-recheck-2026-10-23.xml).
The JSON records the source fingerprint and command. The README quickstart
also converged on CPU and returned the documented `(2, 24, 24, 24)` flux shape.

## Core numerical verification

| Feature | Reference and acceptance criteria | Automated checks |
|---|---|---|
| Cartesian multigroup diffusion | Exact bare-box eigenvalues and sine-mode flux; mesh doubling reduces eigenvalue error by a factor between 3 and 5. Reflective boundaries recover the infinite-medium eigenvalue within `1e-6`. | [Analytic diffusion](../tests/verification/test_diffusion_analytic.py), [boundary conditions](../tests/verification/test_boundary_conditions.py), [neutron balance](../tests/verification/test_neutron_balance.py) |
| Cylindrical r-z diffusion | Bare-cylinder Bessel-mode eigenvalue: finest-mesh absolute error below `5e-6`; successive errors decrease by about four. Changing the boundary specification at the zero-area axis leaves the answer unchanged within `1e-12`. | [Cylindrical tests](../tests/verification/test_cylindrical.py) |
| Triangular and prismatic geometry | Reflective tri-prisms recover infinite-medium `k` within `1e-9`; axial slab errors decrease by `4 ± 0.3` under refinement. Reflective extrusion of VVER-440 preserves its 2-D eigenvalue within `1e-7`. | [Tri-prisms](../tests/verification/test_tri_prisms.py), [triangular SP3](../tests/verification/test_tri_sp3.py), [hexagonal geometry](../tests/verification/test_hex_lattice.py) |
| Unstructured diffusion | Compare structured and unstructured finite volumes on matching meshes; check reflective limits and boundary treatments. The full-core 3-D HP-MR comparison is a separate slow test. | [Mesh verification](../tests/verification/test_unstructured_mesh.py), [3-D HP-MR comparison](../tests/validation/test_hpmr_3d_mesh.py) |
| SPN and SDPN | SP3 analytic eigenvalues and second-order refinement; independent angular-coefficient derivation; symmetry and transformed-operator equivalence. Coupled Marshak SP3/SDP1 eigenvalues extrapolate to the Brantley–Larsen literature values within `1e-4`. | [SP3 analytics](../tests/verification/test_sp3_analytic.py), [SDPN derivation](../tests/verification/test_sdpn_derivation.py), [operator symmetry](../tests/verification/test_sdpn_symmetric.py), [Brantley–Larsen](../tests/validation/test_brantley_larsen_2d.py) |
| Adjoint and perturbation theory | Forward and adjoint eigenvalues agree within `1e-7`. The adjoint-weighted absorption perturbation predicts a separately re-solved reactivity change within 2%. This also checks the importance flux. | [Adjoint](../tests/verification/test_adjoint.py), [perturbations](../tests/verification/test_perturbation.py) |
| SPH and homogenization | Reaction-rate preservation, material consistency, improved leaky-colorset eigenvalues, and preservation of fission-energy coefficients. Slow HP-MR checks require corrected diffusion to recover each selected transport-family reference within 15 pcm. | [SPH verification](../tests/verification/test_sph.py), [HP-MR SPH](../tests/validation/test_hpmr_sph.py), [reference families](../tests/validation/test_hpmr_sph_reference_families.py) |
| Cross-section input | Griffin material balances and conventions; FEMFFUSION files compared with independent benchmark transcriptions, including scattering direction and upscatter. | [Griffin readers](../tests/verification/test_griffin_xs.py), [FEMFFUSION readers](../tests/verification/test_femffusion_io.py) |
| Delayed-neutron transients | Uniform perturbations agree with an independent point-kinetics solution to relative error below `5e-4` in the specified diffusion and SDPN cases. Equilibrium, precursor hand-off, and restart invariants are checked separately. | [Point kinetics](../tests/verification/test_transient_point_kinetics.py), [material-dependent kinetics](../tests/verification/test_transient_per_material.py), [kinetics theory](../tests/verification/test_transient_kinetics_theory.py) |
| BDF time integration | Orders 1–6 differentiate polynomials exactly to floating-point tolerance; nonuniform history, precursor elimination, convergence order, event alignment, rejected-step rollback, and automatic order selection have explicit checks. | [Time schemes](../tests/verification/test_timescheme.py), [transient integration](../tests/verification/test_transient_point_kinetics.py) |
| Quasi-static kinetics | Exact homogeneous projections, normalization invariance, stationary equilibrium, and comparisons with full spatial transients. Reduced HP-MR ramps check power and temperature errors; guards check early shape corrections and full-diffusion fallback. | [Quasi-static tests](../tests/verification/test_quasistatic.py) |

The per-case tolerances above belong to those meshes and inputs. They are not
general accuracy guarantees for arbitrary geometries or cross-section sets.

## Reactor benchmark comparisons

Problem builders, cross sections, and reference constants live in
[`garnet/benchmarks/`](../garnet/benchmarks/). These cases test several levels
of agreement: published solutions, independent code results, and expected
physical behavior. They should be read with their modeling assumptions.

| Benchmark | What is compared | Evidence and scope |
|---|---|---|
| OECD/NEA C5G7, 2-D | Seven-group eigenvalue, geometry, homogenization, and SPN/SDPN hierarchy against the transport reference `k = 1.18655`. | [Tests](../tests/validation/test_c5g7.py). Pin-cell-homogenized calculations: the coarse diffusion test allows 2% relative error; the angular-family comparison allows 0.5%. This is not a pin-resolved transport validation. |
| BIBLIS and IAEA-3D | Eigenvalues, active masks, material symmetry, and Robin/albedo boundaries. | [Tests](../tests/validation/test_biblis_iaea.py). BIBLIS uses a coarse-mesh envelope; IAEA-3D checks absolute eigenvalue error below `1e-3` against `1.02903`. |
| VVER-440 | Body-fitted triangular geometry, eigenvalue, and a bulk transient; optional unstructured solve on the FEMFFUSION mesh. | [Tests](../tests/validation/test_vver440.py). The coarsest triangular test allows `6e-3` absolute eigenvalue error; the external-mesh comparison allows `1e-3`. |
| TWIGL step and ramp | Static eigenvalue, transient power samples, and the public Model API. | [Model tests](../tests/validation/test_model_twigl.py), [solver tests](../tests/validation/test_twigl_langenbuch.py), [worked example](model_api.md#worked-example-the-twigl-benchmark-validated-against-literature). Fine static `k` is within `1e-4` of `0.91321`; API transient samples are within `0.02` in normalized power. |
| Langenbuch/LMW | Three-dimensional moving rod-bank maps and transient peak/decline. | [Tests](../tests/validation/test_twigl_langenbuch.py). A coarse regression with a peak-power envelope, rather than a fine convergence study. |
| ANL-7416 8-A1 | Cylindrical static and transient results, including alternative Krylov solvers and divergence-form operators. | [Tests](../tests/validation/test_anl7416_8a1.py). Early power samples allow 3%; later published traces use a wider envelope, with an independent 4 s comparison. |
| LRA-2D with thermal feedback | Original ANL specification, reflector-data audit, static mesh convergence, power/temperature histories, adaptive BDF, and matched-error work. | [Tests](../tests/validation/test_lra2d.py), [study and data](../benchmark-results/lra2d-adaptive-bdf-report/), [PDF](../benchmark-results/lra2d-adaptive-bdf-report/adaptive-bdf-lra-benchmark.pdf). The archived 3.75 cm run reports first-peak power 3.48% above the original reference; finer-mesh and feedback-zone effects are documented. |
| HP-MR | Assembly counts, symmetry, absorber volume mixing, drum-worth ordering, mesh stability, 3-D extrusion, and SPH reference recovery. | [Geometry and worth tests](../tests/validation/test_hpmr.py), [SPH study tests](../tests/validation/test_hpmr_sph_reference_families.py). The default two-group data are placeholders. Checks against GARNET transport references establish numerical consistency, not independent validation of a physical microreactor. |

## Thermal and coupling checks

The conduction equation is checked against an exact hyperbolic-cosine slab
solution with both imposed-temperature and Robin boundaries. Measured spatial
orders must lie between 1.9 and 2.1. Cartesian, cylindrical, and triangular-prism
tests also check sink limits and energy conservation; the masked HP-MR steady
case requires a relative balance residual below `1e-10`.
See [conduction tests](../tests/verification/test_conduction.py).

[Power normalization](../tests/verification/test_power_density.py),
[feedback](../tests/verification/test_feedback.py), and
[coupled-solver tests](../tests/verification/test_coupling.py) check rated-power
normalization, feedback sign, zero-power and zero-feedback limits, stationary
coupled transients, thermal subcycling, and agreement between accelerators.
The coupled steady energy balance must close below `1e-10`; accelerators must
reach eigenvalues within `2e-8` of each other.

External coupling has a separate [preCICE comparison](../tests/validation/test_coupled_precice.py):
identity data mapping, iteration-by-iteration agreement under identical
relaxation, and agreement at the fixed point with different accelerators.
The [coupling report](coupling.md#6-verification) records a maximum eigenvalue
difference of `5e-13` over 33 lockstep iterations and a fixed-point difference
below `2e-8`. These are archived results; the tests require preCICE and its
Python bindings to reproduce.

## GPU and transient studies

CPU verification exercises the shared numerical implementation. CUDA kernel
execution, memory use, and performance need hardware measurements as well.

| Study | Recorded result | Scope |
|---|---|---|
| [Guarded IQS on GH200](../benchmark-results/hpmr-iqs-gh200/) | Maximum sampled power error of 0.173%, versus 5.708% for fixed-cadence IQS. The guarded 200 s run made 131 shape updates and two full-diffusion fallbacks. | 105,600 active cells, 11 groups, 20 axial layers. Accuracy was compared every 10 s through 110 s, where the full-diffusion reference ends. [Machine-readable record](../benchmark-results/hpmr-iqs-gh200/adaptive-fallback-r4-nz20.json). |
| [LRA adaptive-BDF study](../benchmark-results/lra2d-adaptive-bdf-report/) | Temporal/spatial convergence, controller histories, and a 5.18× CPU speedup over adaptive backward Euler at matched error. | An LRA-specific result. The report also records a coupled 11-group HP-MR case where BDF did not outperform matched adaptive backward Euler. |
| [T4 3-D adaptive/monolithic study](../benchmark-results/transient-acceleration/step-08a-adaptive-bdf-gpu-readiness.md) | Accuracy, energy-group batching, persistent Krylov memory, state-update microbenchmarks, and repeated timing comparisons. | Quick case: 13,200 active cells and 145,200 flux unknowns. Larger production and reference-convergence checks remain pending. |
| [Energy-sweep acceleration study](../benchmark-results/transient-acceleration/step-10a-energy-sweep-anderson.md) | CPU and T4 configurations were evaluated separately; the T4 fixed-polynomial configuration gave a 1.101× median gain over the plain configuration. | Three interleaved T4 samples. Tolerance-based energy Anderson was rejected on T4 despite its CPU benefit. |

The [optimization index](../benchmark-results/transient-acceleration/) retains
accepted and rejected experiments. Performance claims belong to the stated
hardware, precision, mesh, and accuracy target.

## Additional solver checks

The lower-level noise solver has exact zero-power transfer-function tests:
complex amplitude and phase errors below `1e-6`, with spatial flatness checked
separately. A stored FEMFFUSION slab reference checks both diffusion and SP3
noise, requiring relative field L2 error below 1% on the 60-cell mesh.
See [point-kinetics noise](../tests/validation/test_noise_point_kinetics.py)
and [FEMFFUSION noise](../tests/validation/test_noise_femffusion.py).

Discrete-ordinates and hybrid solvers have their own
[Cartesian](../tests/verification/test_sn.py),
[triangular](../tests/verification/test_tri_sn.py), and
[hybrid-interface](../tests/verification/test_hybrid_tri_sn.py) tests. Their
coverage is separate from the core diffusion/coupled workflow described above.

## Reproducing the checks

From an installed checkout:

```bash
python -m pip install -e '.[test]'
python -m pytest tests -ra
python -m pytest tests -m slow -ra     # expensive cases only
python -m pytest tests -o addopts= -ra # all cases, including slow
```

The default marker selection is defined in [pyproject.toml](../pyproject.toml).
`FEMFFUSION_DIR` must point to reference inputs with an `examples/` directory
for the file-reader tests. Some mesh tests use a separate fixed external path;
inspect the skip reasons with `-ra`. The FEMFFUSION noise comparison embeds
its reference and does not require a live FEMFFUSION solver.

Install the CUDA extra and use actual GPU hardware for CUDA-only kernel tests.
The [GPU regression job](../slurm/run_garnet_gpu_regression.sh) checks that CuPy
can see a device and clears the default slow-test exclusion; many physics
tests still explicitly select the CPU. For CPU/GPU solve comparisons use the
[benchmark notebook](../notebooks/colab_cpu_gpu_benchmarks.ipynb); for coupled
transients use the [GPU walkthrough](../notebooks/colab_coupled_transient_gpu_latest.ipynb)
and [3-D adaptive-BDF notebook](../notebooks/colab_hpmr3d_adaptive_bdf_gpu.ipynb).
preCICE installation and its test command are in the
[coupling guide](coupling.md#running-it).

The [July theory report](report.html) retains the original discretization
derivations and benchmark snapshot. Use this report and the
[user guide](user_guide.md) for the current coverage and supported workflows.
