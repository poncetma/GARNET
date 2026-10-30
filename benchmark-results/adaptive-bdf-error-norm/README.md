# Adaptive BDF error norm: concatenated vs blockwise (audit item O11)

Measured 2026-10-29 on CPU (merlin7 `cn000`, one thread per process).

The adaptive BDF controller accepts a step when the predictor–corrector
defect passes Eq. (44) of Cherezov et al.:
‖Δu‖₂ / ‖atol + rtol·|u|‖₂ ≤ 1. Until now `u` was every field joined: flux
groups, precursor families, and any external state such as the LRA
temperature. Those fields carry different units, so the field with the
largest values decided:

- at the LRA 2 s restart the temperature was 1e-16 of the defect and 6e-12 of
  the scale;
- at t = 0 the temperature was about 100% of the scale, so the first step's
  flux defect was judged about 950× more loosely than against its own scale.

`error_norm="blockwise"` applies Eq. (44) to each field and accepts on the
largest, so every field meets the tolerance relative to its own scale. With a
single field the two norms are identical. The default `"concatenated"`
implementation is bitwise unchanged: 0 differences in 3000 random cases, and
the adaptive LRA run reproduces P(3 s) to the last bit.

**Decision: blockwise is the `adaptive_bdf` default.** `error_norm=
"concatenated"` reproduces the paper's Eq. (44) and every adaptive record
made before 2026-10-29. `BDF.error_norm()` itself keeps the published default.

## Neutron-only kinetics with exact references

The cores are the 2×2 reflective infinite-medium cores of
[test_kinetics_reference.py](../../tests/verification/test_kinetics_reference.py),
with six families and T_end = 0.2 s. Steps use `expm`; the ramp uses Radau at
rtol 1e-12. The error is the maximum relative power error over every
accepted step.

| Case | Scheme | rtol | Concatenated: error / steps / rejected | Blockwise: error / steps / rejected |
|---|---|---|---|---|
| 1-group step | BDF2 | 1e-3 … 1e-6 | identical | identical |
| 1-group step | BDF5 auto | 1e-3 … 1e-6 | identical | identical |
| 2-group step | BDF2 | 1e-4 | 1.92e-4 / 85 / 9 | 1.90e-4 / 85 / 9 |
| 2-group step | BDF2 | 1e-6 | 9.21e-6 / 355 / 12 | 9.17e-6 / 362 / 15 |
| 2-group step | BDF5 auto | 1e-5 | 1.37e-5 / 73 / 15 | 6.64e-6 / 73 / 14 |
| 2-group step | BDF5 auto | 1e-6 | 3.83e-7 / 102 / 15 | 3.19e-7 / 103 / 18 |
| 1-group ramp | BDF2, BDF5 auto | 1e-3 … 1e-6 | — | within ±2 steps, same error |
| fast step (v = 2.2e9) | BDF2 | 1e-3 | 7.87e-4 / 49 / 3 | 7.87e-4 / 53 / 3 |
| fast step | BDF2 | 1e-6 | 9.20e-6 / 367 / 8 | 9.20e-6 / 412 / 8 |
| fast step | BDF5 auto | 1e-4 | 1.24e-4 / 56 / 11 | 1.24e-4 / 61 / 10 |

Every row is in [kinetics.json](kinetics.json). On neutron-only problems the
norms are close:

- blockwise takes 0–13% more steps, and the most on the fast step, where the
  precursor families now get their own control;
- the power error is the same or lower.

The H12 regression case (BDF2, default rtol, initial dt 1e-3) takes 53
accepted steps instead of 49, and its P(t_end) error falls from 1.8e-4 to
3.7e-5.

## LRA-2D with adiabatic feedback (coupled temperature)

The case is `run_lra2d_cpu(refine=1, implicit_feedback=True)`: initial dt
1e-6, factor-two rejection. The reference is BDF5 with automatic order at
rtol 1e-7, blockwise. The concatenated run at 1e-7 agrees to 5e-10 in P(3 s)
and 3.3e-5 in the sampled peak. The peak is the largest accepted-step sample,
so its error includes the sampling of a ~0.05 s wide prompt peak.

| Scheme | rtol | Norm | Peak power error | Peak time error [s] | P(3 s) error | T(3 s) error | Steps | Rejected | t [s] |
|---|---|---|---|---|---|---|---|---|---|
| BDF2 | 1e-3 | concatenated | −1.23e-2 | −1.9e-3 | −2.90e-4 | +1.0e-4 | 453 | 16 | 14.8 |
| BDF2 | 1e-3 | blockwise | **−5.65e-3** | −8.7e-4 | −3.03e-4 | +1.0e-4 | 539 | 11 | 16.7 |
| BDF2 | 1e-4 | concatenated | −2.46e-3 | −4.4e-4 | −7.13e-5 | +2.5e-5 | 960 | 16 | 21.1 |
| BDF2 | 1e-4 | blockwise | **−1.09e-3** | −1.2e-4 | −7.12e-5 | +2.4e-5 | 1158 | 11 | 24.9 |
| BDF2 | 1e-5 | concatenated | −5.13e-4 | −1.0e-4 | −1.63e-5 | +5.8e-6 | 2054 | 15 | 30.0 |
| BDF2 | 1e-5 | blockwise | **−2.30e-4** | −1.4e-4 | −1.65e-5 | +5.7e-6 | 2488 | 8 | 36.4 |
| BDF5 auto | 1e-3 | concatenated | −4.63e-3 | −9.9e-4 | +9.3e-6 | −2.6e-7 | 223 | 27 | 9.0 |
| BDF5 auto | 1e-3 | blockwise | **−9.46e-4** | −7.5e-4 | +8.6e-6 | −3.0e-6 | 246 | 19 | 9.4 |
| BDF5 auto | 1e-4 | concatenated | −1.52e-3 | −8.2e-4 | −6.3e-7 | −6.5e-8 | 315 | 18 | 9.2 |
| BDF5 auto | 1e-4 | blockwise | **−1.21e-4** | +1.4e-4 | −8.8e-7 | +5.9e-7 | 368 | 20 | 10.6 |
| BDF5 auto | 1e-5 | concatenated | −1.37e-4 | +8.6e-5 | −2.0e-7 | +3.0e-8 | 467 | 22 | 11.3 |
| BDF5 auto | 1e-5 | blockwise | −1.56e-4 | −3.4e-4 | +1.4e-7 | −1.4e-7 | 547 | 18 | 12.8 |

Rows are in [lra2d.json](lra2d.json).

## Reading

- **What blockwise costs.** At the same rtol it takes 10–21% more LRA steps
  (4–21% more time) and has fewer rejections.
- **Peak power improves at matched work.** Interpolating the concatenated
  error–steps curve to the same step count gives about 1.5× lower peak error
  for BDF2 at every rtol, and 3.6× and 4.9× lower for BDF5 at rtol 1e-3 and
  1e-4. At BDF5 1e-5 both norms sit near the peak-sampling floor (the two
  references differ by 3.3e-5), and blockwise is slightly worse there.
- **End-of-transient values don't move.** P(3 s) and T(3 s) agree at the same
  rtol, so for those quantities alone blockwise costs about 15% more work.
- **Neutron-only problems:** neutral to +13% work, with equal or lower error.

## Limits

- The per-field norm is still an unweighted Euclidean ratio over cells. The
  volume-weighted componentwise RMS proposed in
  [numerical scaling plan](../../docs/numerical_scaling_plan.md)
  is not implemented.
- Temperature is scaled by its absolute value in kelvin, not by `T − T_ref`.
- The stored [LRA adaptive-BDF report](../lra2d-adaptive-bdf-report/),
  including its 5.18× matched-error speedup over adaptive backward Euler, was
  made with the concatenated norm. It has not been regenerated and reproduces
  with `adaptive_bdf={..., "error_norm": "concatenated"}`.
- No GPU timing; the extra work per step is one more scalar pair per field in
  the existing single host transfer.
