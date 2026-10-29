# Diffusion transients through the model API

`Model`, `CylindricalModel`, and `TriReactor` share the same neutron-transient
interface and return `TransientModelResult`. They also support thermal
feedback through `configure_thermal` and the coupled methods. The underlying
diffusion and time-integration solvers remain available for specialized work.

## A timed material change

```python
from garnet import CylindricalModel, Kinetics, PWR_TWO_GROUP

fuel = PWR_TWO_GROUP.replace(name="fuel")
core = (CylindricalModel(radius=60, height=120, cells=(12, 24))
        .fill(fuel).set_boundary(r="vacuum", z="vacuum")
        .set_kinetics(Kinetics([1e7, 3e5], [0.0065], [0.08])))
steady = core.steady(device="auto")
poisoned = fuel.replace(sigma_a=fuel.sigma_a * [1.0, 1.003])

run = core.transient(
    t_end=0.1, dt=0.01, time_scheme="bdf2", device="auto",
    initial_steady=steady,
    materials_at=lambda t: {"fuel": poisoned} if 0.035 < t <= 0.08 else {},
    event_times=[0.035, 0.08],
)
print(run)
print(run.power_at([0.02, 0.06, 0.1]))
```

The initial eigenvalue `k0` critically adjusts the fission source, so an
unperturbed calculation stays at `P/P0 = 1` to solver accuracy.
`materials_at(t)` accepts a partial
name/index mapping or a complete ordered list; each mapping is relative to the
base model. Return cached material objects while the constants are unchanged.
The operator cache is keyed on object identity, so when the constants change,
return a new `Material` rather than mutating the arrays of one already
returned: an in-place edit to the same object is not detected and the solve
keeps the stale operator.

For moving assignments, `state_at(t)` returns a cached model with the same
grid, physical dimensions, boundaries, active cells and material count. Use
`at(t)` for the advanced two/four-array problem specification. These three
callbacks are mutually exclusive. The
[complete example](../examples/cylindrical_transient.py) includes a reflector
and accepts `--device cpu`, `--device gpu`, and `--adaptive`.

## Time steps, events and adaptive BDF

`dt` is a scalar fixed width, an explicit list of widths summing to `t_end`,
or the initial width for adaptive BDF. Backward Euler remains the default.
Select `time_scheme="bdf2"` through `"bdf6"` explicitly for higher orders.

`event_times` must lie strictly inside `(0, t_end)`. For fixed BDF steps, the
API splits the schedule at those times and includes a final short interval if
needed. A grid endpoint within `1e-3` of the local step width of an event is
moved onto the event instead, so no sliver step is created (`t_end` itself is
never moved). Adaptive BDF hits the same times exactly. This prevents a control
jump from being silently crossed by a long step.

The event endpoint belongs to the old smooth segment: the callback's value at
`t_e` must be the pre-event state, as its value at t=0 is. Write `t > event`
when the new state begins there, as in the examples. BDF history is discarded
before the step *leaving* the event, which restarts at order one; this holds
for fixed and adaptive steps and for low-level `bdf_restart_times`. Before
29 October 2026 fixed schedules restarted the step ending at the event and
recommended `t >= event`; the observed order then fell to about one.

```python
adaptive = core.transient(
    t_end=0.1, dt=0.005, time_scheme="bdf3", device="auto",
    materials_at=lambda t: {"fuel": poisoned} if t > 0.035 else {},
    event_times=[0.035],
    adaptive_bdf={"rtol": 1e-3, "min_dt": 1e-7, "max_dt": 0.01},
)
```

Supplying `adaptive_bdf` selects monolithic stepping unless `step_solver` was
explicitly supplied. If the time scheme is omitted, it uses adaptive BDF1
(backward Euler). Existing low-level `bdf_restart_times` and
`step_alignment_times` are still available. Use `event_times` or
`bdf_restart_times`, not both. Nonuniform/event schedules require BDF or FTM.
Controller tolerances govern a local defect estimate; assess time convergence
for the observable and transient of interest. On the first step, and after
each restart, no history exists. There the defect is taken against an
explicit-Euler predictor that uses the materials just after the start (the
post-event state). This gives an O(h²) defect, like later steps, so a fast
prompt jump is resolved rather than rejected down to `min_dt`. A coupled
driver's own `adaptive_error_state` prediction is not covered by this: a
constant startup prediction there is still an O(h) defect.

**Formal order.** Fixed-step BDF3–BDF6 converge only at second order, although
with a smaller error constant than BDF2. Their BDF1/BDF2 startup steps, at t=0
and after each restart, run at the full width, and the startup error persists.
Third order needs startup widths that shrink faster than `dt`. The adaptive
controller selects such widths. Alternatively, pass an explicit `dt` list
graded from about `dt*sqrt(dt/t_end)` by a ratio of at most 1.5 (measured order
3.25 for BDF3). `time_scheme="crank-nicolson"` is second order for data that are
continuous at t=0, and it stays second order across restarts, which re-seed
its explicit half from the post-event state. For a perturbation applied at t=0
it is first order, because its first explicit half uses the t=0 steady state.

Experimental `time_scheme="ftm2"` and `"ftm-bdf2"` also use this interface
with fixed or explicit widths. The latter supplies a damped, full-state
transformed BDF2 recurrence. Both refuse `adaptive_bdf`; see the
[damping checkpoint](../benchmark-results/exponential-second-order/CHUNK3.md)
for safeguards, precursor differences and the measured limits.

For neutron/precursor adaptation, use the separate opt-in controller:

```python
transformed = core.transient(
    t_end=0.1, dt=0.005, time_scheme="ftm-bdf2", device="cpu",
    adaptive_ftm={"rtol": 1e-4, "output_rtol": 1e-4,
                  "min_dt": 1e-10, "max_dt": 0.01},
    materials_at=lambda t: {"fuel": poisoned} if t > 0.035 else {},
    event_times=[0.035],
)
```

`adaptive_ftm` supports `ftm2` and `ftm-bdf2`; omitting the scheme selects
`ftm-bdf2`, and omitting the spatial solver selects `monolithic`. It compares
a full step with two half steps in physical flux and precursor fields, then
commits the two-half-step state once per macro endpoint. Rejections restore
the physical state, frequency samples and recurrence histories. Callbacks
and reported times include only committed macro endpoints. FTM events follow
the same convention as BDF: use `t > event` to change the material in the
interval leaving that endpoint, as in the example above.

Each group/family has a separate active-volume weighted RMS error; the
maximum controls acceptance. Scaling uses
`atol + rtol * max(abs(start), abs(fine))`, with scalar `atol=0` by default.
The raw full/fine difference has no Richardson divisor. The width controller
defaults to `max_factor=1.2` and `rejection_strategy="error"`; other width
controls match `adaptive_bdf`, excluding automatic order selection.
`min_dt` and `max_dt` bound macro widths; internal half widths are half that.
Existing adaptive BDF retains its original error norm and defaults.

`local_error_blocks` and `rejected_error_blocks` list flux-group errors then
precursor-family errors. `ftm_trial_diagnostics` and
`rejected_ftm_trial_diagnostics` retain each full/half trial's width, order,
frequency, fallback and correction counts. An early invalid-population retry
can stop after the first half. Solve/iteration counters include discarded
full steps, both halves, refinements and rejected attempts. `time_orders`
uses the minimum active trial order; `step_frequencies`, `step_frequency_log_defects`
and `ftm_fallback_reasons` describe the final half, while
`step_frequency_refinements` totals every trial of an accepted attempt,
including any discarded work when an estimator fallback replays the attempt.
BE startup counts internal accepted half steps.

Set `output_rtol` to additionally restrict widths for linear output. The
controller compares the physical first-half-step state with the average of
the starting and final states. Each group and precursor family has a separate
active-volume weighted RMS chord defect, scaled by
`output_atol + output_rtol * abs(midpoint)`. A separate scalar check applies
the same scale to midpoint relative power, including changing fission
coefficients. `output_atol` defaults to zero and requires `output_rtol`;
it is an absolute floor in each field's units and in P/P0 for power.
Initially zero fields follow the same zero-scale rules as the state defect.
This check uses the existing first half step and adds no spatial solve.
The state and output checks each propose a width; the smaller proposal wins.
The chord defect uses square-root scaling because it is O(h²).

With this option, `local_errors` and `rejected_errors` contain the maximum
of state and output defects. `local_error_blocks` and `rejected_error_blocks`
retain the full/fine physical state defects. `output_error_blocks` and
`rejected_output_error_blocks` contain the chord defects in flux-group,
precursor-family, then scalar-power order. Output arrays stay empty when
the option is omitted or `output_rtol=None`. Macro times, callbacks and the
linear `power_at` formula keep their existing meaning.

An experimental `adaptive_ftm['estimator']` selects the state check:

| Value | State check and solve cost |
|---|---|
| `"step-doubling"` (default) | Full step versus two half steps; three endpoint solves before frequency refinements. |
| `"history"` | Quadratic physical-history prediction versus the two-half-step endpoint; normally two endpoint solves. Uses step doubling when history is unavailable or unsafe. |
| `"history-audit"` | Records both estimates on the same eligible trials, retaining step-doubling decisions and its three-solve trajectory. |

The history prediction uses the previous accepted midpoint, the macro start
and the current midpoint, with their actual widths. It stores physical flux
and precursor fields, so a changed frequency does not mix exponential frames.
The raw prediction/correction defect uses the same per-field norm and has
no Richardson divisor or universal error constant. It can be more restrictive
on a nearly exact exponential tail because it still sees physical curvature.
`output_rtol` remains an independent midpoint/chord check in all modes.

Startup, explicit events, BE safeguards and half-width ratios outside
`[0.25, 2]` use step doubling. If a provisional half step unexpectedly enters
a BE safeguard, the controller rolls back and replays that macro attempt
with step doubling. Its discarded half solves and frequency refinements
remain charged, so an attempt can exceed three solves. Error order uses
the replayed trajectory; work diagnostics retain all discarded trials.

`ftm_estimator_diagnostics` and `rejected_ftm_estimator_diagnostics` contain
`method`, `reason` and `history_error_blocks` for each macro attempt in the
two opt-in modes; default-mode lists are empty. A `None` history defect means
there was insufficient safe second-order history. In `"history"` mode,
`local_error_blocks` contains the selected history or fallback full/fine
defects; in audit mode it always contains the full/fine reference defects.
Trials, callbacks, rollback and accepted two-half-step states keep their
existing meaning. The [estimator checkpoint](../benchmark-results/exponential-second-order/CHUNK6.md)
records same-trial calibration and matched physical error/work comparisons.

The [TWIGL assessment](../benchmark-results/exponential-second-order/CHUNK7.md)
also checks spatial fields and region powers. Its ramp shows that a smaller
`adaptive_bdf['max_dt']` can satisfy the linear-output budget with less work
than tightening `rtol` at a larger cap. Select output resolution independently
of state tolerance for every scheme, and keep mesh error separate from
temporal error. This study does not establish an FTM performance promotion.

This controller currently excludes implicit feedback, external error state
and thermal coupling. Local midpoint checks do not certify every interior
point or accumulated integration error. Keep physical output budgets separate
from either local tolerance. A jump in the recorded power itself needs
two-sided event output, which is not implemented by this continuous chord.
The [adaptive checkpoint](../benchmark-results/exponential-second-order/CHUNK4.md)
records the original rapid-growth interpolation limit, and the
[output checkpoint](../benchmark-results/exponential-second-order/CHUNK5.md)
records its resolution and the remaining cost comparison.

Constructor settings such as `dtype` belong in `solver_kwargs`. Common
`precond_degree`, `precond_dtype`, and `linear_solver` controls are also direct
arguments. `tol_step`, `time_scheme`, `adaptive_bdf`, `adaptive_ftm`, `linsolve_kwargs`, and
`multigroup_kwargs` go to the time integrator. Mixed preconditioning is still
opt-in and is not supported by monolithic steps.

## Results and progress

The common result provides:

| Field or method | Meaning |
|---|---|
| `steady`, `k0` | Public time-zero steady report and critical-adjustment eigenvalue |
| `times`, `power` | Accepted times in seconds and normalized `P/P0`, including t=0 |
| `step_widths`, `n_steps` | Actual accepted widths and step count, for fixed or adaptive runs |
| `flux`, `flux_numpy` | Final scalar flux as NumPy; its grid axes are preserved |
| `precursors_numpy` | Final precursor fields copied to NumPy on request |
| `peak_power`, `peak_time`, `final_power` | Recorded extrema and final power |
| `power_at(times)` | Linear interpolation inside the recorded span; no extrapolation |
| `raw` | Original solver result, including device-resident flux and precursors |

Solver telemetry such as `time_orders`, `local_errors`, `rejected_steps`,
`step_iterations`, and `initial_state_reused` is forwarded from `raw` and
appears in tab completion. `dt` retains the first recorded width for
compatibility; use `step_widths` when the schedule varies. `power_at` samples
the recorded history; it is not a high-order dense-output formula.

```python
accepted = []
run = core.transient(
    t_end=0.1, dt=0.01, device="auto",
    on_step=lambda t, flux, power: accepted.append((t, float(power))),
)
```

`on_step` runs once after each accepted neutron step and never for rejected
attempts or the initial state. The flux is a private copy on its current
device: a `(G, ...)` array on the batched GPU path, otherwise a list of
per-group arrays. It may be retained across steps, and writing to it does not
affect the solver. Ordinary
result reports transfer the final scalar flux to NumPy; device workflows can
use `run.raw.flux` and `run.raw.precursors`.

Public `initial_steady` results must be converged forward diffusion solutions
matching the actual t=0 grid, boundary conditions, cross sections, maps, and
volume mixing. This is checked against saved metadata, including after the
original builder has been edited. Raw solver results retain their existing
caller-owned physical-compatibility contract. Reusing an initial eigenpair is
not a general transient restart/checkpoint facility.

Public steady results built with raw `xs_update`/`xs_update_at` field hooks are
rejected for automatic reuse: their effective spatial cross sections are not
captured by material metadata. The same restriction applies when the new
transient uses `xs_update_at`. Express the t=0 constants as model materials,
or use the low-level result with its explicit caller-owned compatibility check.

## Cylindrical r-z geometry

`CylindricalModel(radius, height, cells=(nr, nz))` represents a full body of
revolution. `add_region(material, r=(...), z=(...))` paints cell centres in an
annular axial region. `add_box` is an alias. `r="vacuum"` sets the cylinder
wall; `z=("reflective", "vacuum")` sets the lower and upper axial faces.
The r=0 face has zero area and is a natural symmetry axis. The y faces of an
r-z grid are the azimuthal seam, not a surface, so they are always reflective.
This also holds for a scalar `bc` passed to a low-level cylindrical `Grid`
solver.

`material_map` has shape `(nr, nz)` and scalar flux has shape
`(groups, nr, 1, nz)`. `radial_centers` and `axial_centers` provide coordinates.
`grid.cell_volumes` and steady `result.cell_volumes` contain full annular
volumes `π(r_outer² − r_inner²) Δz`. Thus `result.power_density(total_power=...)`
integrates to the requested full-core watts; there is no slice-power factor.
The existing scalar `Grid.cell_volume` retains its rectangular stencil meaning.

To use an existing structured benchmark definition without repainting:

```python
from garnet import Model

core = Model.from_grid(grid, materials, material_map, bc=bc, kinetics=kinetics)
run = core.transient(t_end=1.0, dt=0.01, at=problem_at, device="cpu")
```

`Model.from_grid` returns `CylindricalModel` for a cylindrical grid and copies
the material map. On Cartesian input, trailing one-cell axes are collapsed as
in a model built in 1-D or 2-D, so a single `bc` spec leaves them reflective.
An explicit per-axis or per-face `bc` that closes such an axis keeps it as a
physical one-cell slab. Before 2026-10-29 a scalar `"vacuum"` closed the
collapsed z axis of a 2-D grid too. Physical extents are preserved either way.

## Thermal coupling

The [thermal configuration](model_api.md#thermal-coupling) works for all three
grid types. `coupled_transient` and `quasistatic_transient` now accept named
`materials_at` as well as cached `state_at` and the raw `problem_at` callback.
Only one perturbation callback may be supplied. Material count/order and the
configured thermal/feedback indexing stay fixed across states.

For coupled progress, `on_step(t, flux, power, temperature)` observes an
accepted neutron step after any due thermal exchange. Temperature is held
between those exchanges. Arrays remain on their current device and should be
treated as read-only.

Adaptive neutron controls enter through
`coupled_transient(transient_kwargs={"time_scheme": "bdf2",
"adaptive_bdf": {"rtol": 1e-3}})`. Monolithic/BDF1 defaults are filled in as
for neutron-only runs. Thermal exchange times are aligned by the coupled
driver without restarting BDF history. For fixed stepping, `dt_thermal` must
remain an integer multiple of `dt`.

`quasistatic_transient` with the default `shape_method="iqs"` iterates each
shape interval as a predictor–corrector. The interval is replayed from its
start, with the end-shape correction to ρ, Λ and β interpolated in time,
until the end power changes by at most `iqs_iteration_tol` (default `1e-4`).
It is rejected and retried with half the width after `iqs_max_iterations`
replays (default 4). It is also rejected when the corrector's own end power
differs from the accepted power by more than `iqs_predictor_tol`. Before
29 October 2026 that tolerance only shortened the *next* interval, so guard
settings calibrated earlier need re-checking. `result.counters` reports
`iqs_corrector_iterations` (replays), `iqs_max_iteration_change_ppm`,
`iqs_interval_rejections`, split into `iqs_predictor_interval_reductions` and
`iqs_unconverged_rejections`, and `iqs_predictor_interval_recoveries`. It also
reports `iqs_unconverged_accepts` and `iqs_predictor_tol_accepts`, which count
one-fine-step intervals that cannot be halved further. `thermal_redos` counts
thermal windows that were re-stepped after a power correction. See the
[quasi-static plan](quasistatic_acceleration_plan.md#predictor-corrector-iqs-2026-10-29-audit-fix)
for the algorithm and measured accuracy.

## Test migration and evidence

Six existing test modules were reworked where the public diffusion API can
express the same calculation: cylindrical analytics, ANL-7416, TWIGL/LMW,
point-kinetics/BDF, per-material kinetics, and triangular prompt-jump/inhour
checks. Their 58 test functions and 149 assertion statements/calls are
preserved. Specialized solver-internal tests remain active at the lower level.

Original modules are retained byte-for-byte under `tests/legacy/` and excluded
from default collection. Run them explicitly:

```bash
python -m pytest tests                         # current API and solver checks
python -m pytest tests/legacy --run-legacy      # 62 preserved low-level cases
python -m pytest tests --run-legacy             # current and preserved checks
```

The [new workflow tests](../tests/verification/test_transient_model_api.py)
cover common results, callbacks, event schedules, adaptive-step rejection,
time-zero compatibility, physical annular volumes, and full/quasi-static
cylindrical coupling. Recorded results and reproduction details are in the
[V&V report](verification_validation.md#transient-api-and-cylindrical-check).
