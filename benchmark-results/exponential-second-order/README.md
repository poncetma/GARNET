# Second-order exponential neutrons: first development checkpoint

2026-10-25. Experimental CPU implementation, selected with `time_scheme="ftm2"`.
This checkpoint develops neutron and delayed-neutron time integration only.
The heat-transfer equation is outside the exponential-transformation scope.

Follow-up: [prompt-event and SIMULATE-inspired development](CHUNK2.md) adds
optional BE start-up and bounded frequency refinement, with rapid-growth
and shutdown comparisons. The measurements below retain the first checkpoint.
The [damping checkpoint](CHUNK3.md) adds transformed BDF2, a two-stage research
candidate and a four-case comparison; neither changes the heat equation.
The [adaptive checkpoint](CHUNK4.md) adds physical-state error control and
rollback for FTM2/transformed BDF2, with a localized multigroup reference.
The [output checkpoint](CHUNK5.md) adds optional midpoint/chord width control,
resolving the rapid-growth linear-interpolation discrepancy while retaining
the existing physical-state controller and adaptive BDF comparator.
The [estimator checkpoint](CHUNK6.md) adds an opt-in two-solve physical-history
check, reference audit mode and explicit startup/event/fallback safeguards.
The [TWIGL checkpoint](CHUNK7.md) assesses spatial time integration against
independent physical references and tuned BE/BDF controls. It identifies
an unresolved thermal-shape mesh error and retains BDF as the cheaper
tested comparator at the declared 0.1% temporal budget.
The [eighth checkpoint](CHUNK8.md) closes setup reuse by measurement and
adds PARCS-style quadratic physical precursor production as an opt-in control.
It finds no error/work benefit, and shows that tuning FTM's output tolerance
makes FTM2 tie BDF on the TWIGL ramp, while BDF stays cheaper on the step.

## Implemented method

The scheme applies the trapezoidal rule after removing one shared frozen
frequency from all neutron groups. For `M*dphi/dt=R(phi,C,t)` and width h,
the physical endpoint equation is

```text
M*(2/h+omega)*phi_new - R_new
    = exp(omega*h)*(M*(2/h-omega)*phi_old + R_old).
```

The existing diffusion, scattering and fission operators remain in physical
variables. The time scheme carries the accepted physical derivative, not a
derivative in the previous exponential frame. Initialization and explicit
restarts evaluate the physical neutron RHS, including leakage, geometry
weights and supplied non-equilibrium precursors. Subsequent accepted
derivatives follow from the solved neutron equation. No frequency-corrector
spatial solves are added.

Delayed precursors integrate a linear transformed production history:

```text
z_j = -(lambda_j+omega)*h,
B_j = h*phi_2(z_j),
A_j = exp(omega*h)*h*(phi_1(z_j)-phi_2(z_j)),
C_j_new = exp(-lambda_j*h)*C_j_old
          + A_j*(beta_j*S)_old + B_j*(beta_j*S)_new.
```

Old and new **production products** are kept separately, including mapped
delayed fractions when materials change. Stable moment formulas avoid
cancellation at zero and for stiff precursor decay. This is a second-order
approximation for general smooth production and exact for constant
transformed production; it is not a full matrix-exponential integrator.

The existing accepted-population frequency predictor is reused. The initial
estimated frequency is zero. `ftm_kwargs={"frequency": value}` prescribes a
shared rate for verification. Widths may be fixed or explicitly nonuniform;
`adaptive_bdf` continues to require a BDF scheme. Rates exceeding the existing
bound use actual backward Euler and report time order one in that step.

FTM2 returns the ordinary endpoint source weight h to coupling callbacks.
It adds no exponential heat quadrature and changes no heat-transfer solver.
Second-order neutron integration alone does not establish second-order
accuracy of a coupled thermal march. LRA driver support and coupled efficiency
measurements are deferred to a later checkpoint.

## Verification

The targeted checks cover:

- Scalar moments against 90-digit references, including removable
  singularities and very fast precursor decay.
- Exact positive/negative complete kinetics eigenmodes and second-order
  convergence of mixed modes, with prescribed or lagged frequencies.
- Smooth time-dependent neutron problems, unequal widths and restarts.
- Actual non-equilibrium precursor initialization against matrix exponentials.
- Both endpoint production products against independent numerical quadrature.
- Two-group, two-family changing-fission problems against an independent
  Radau ODE integration with distinct delayed spectra.
- Agreement of groupwise and monolithic spatial solves, material-specific
  kinetics, equilibrium preservation and actual BE fallback.
- Existing BE/BDF/theta, multigroup and example regression checks.

Two targeted test batches pass: **41 tests** in
`chunk1-exponential-verification.xml` and **116 tests** in
`chunk1-verification.xml`, with 12 overlapping tests (145 distinct cases).
The entire repository suite was not rerun for this checkpoint. GPU execution
and performance have not been measured.

## Initial error versus work

The [reproducible report](chunk1-neutrons.json) records 24 configurations on
the existing reflective 4 by 4 by 1 smooth-ramp FV benchmark. All schemes use
the same independent Radau reference and common-grid linear interpolation.
The reference-refinement discrepancy is recorded. Counts include rejected
attempts; these are single executions for numerical work, with no wall-time
speedup claim.

Representative configurations meeting the same 0.1% maximum power-error budget:

| Scheme | Width/control | Maximum power error | Neutron block solves | PCG iterations |
|---|---|---:|---:|---:|
| Backward Euler | h=0.002 s | 0.057645% | 500 | 4,972 |
| First-order FTM | h=0.002 s | 0.067026% | 500 | 4,933 |
| Second-order FTM2 | h=0.01 s | 0.059831% | 100 | 1,159 |
| Crank–Nicolson | h=0.01 s | 0.059831% | 100 | 1,096 |
| Adaptive BE | rtol=1e-4 | 0.038318% | 160 | 1,434 |
| Adaptive BDF1–5 | rtol=1e-5 | 0.031374% | 96 | 845 |

Second order removes the earlier fivefold block-solve deficit in this case.
The transform has not improved on CN or adaptive BDF here. The peak
interpolated error occurs in the initial interval, before a lagged frequency
is available; FTM2 and CN therefore have almost identical maxima. Their
endpoint errors do differ: 0.020338% for FTM2 versus 0.019538% for CN at
h=0.01 s. Both metrics are retained in the report.

## Next small chunk

Check prompt-event start-up, stiff-mode oscillations and spatial shape changes
against BE, CN and adaptive BDF. Transformed trapezoidal integration does not
inherit BE's stiff damping. Event alignment/restarts alone do not supply a
start-up damping policy. Do not infer general transient robustness or a
thermal speedup from this smooth-ramp result. Keep group-dependent rates,
frequency correction and adaptive FTM2 as separate later changes.

## Reproduction

With GARNET and its test dependencies installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/assess_exponential_second_order.py
python3 -m pytest tests/verification/test_exponential_coefficients.py tests/verification/test_exponential_theta.py tests/verification/test_exponential_transient.py -q
python3 -m pytest tests/verification/test_exponential_theta.py tests/verification/test_timescheme.py tests/verification/test_transient_point_kinetics.py tests/verification/test_transient_per_material.py tests/verification/test_multigroup_step.py tests/verification/test_examples_current.py -q
```

The JSON contains source hashes for the implementation and comparison script.
The earlier first-order prototype and proposal-review measurements are
preserved in their original reports.
