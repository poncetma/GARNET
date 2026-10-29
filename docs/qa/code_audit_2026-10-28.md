# Code audit of the mature solver features

Audit date: 28 October 2026; fixes applied 28–29 October 2026 on branch
`audit-fixes`.

Scope: the features with benchmark V&V. These are Cartesian and r-z diffusion,
SPN/SDPN, the Krylov solvers and preconditioners, space-time kinetics,
quasi-static acceleration, thermal coupling and feedback, SPH and the
cross-section readers, the `Model` API, and triangular, hexagonal and
unstructured meshes.

Out of scope: S_N, tri-S_N, the hybrid solvers, ADF (audited separately in
[adf_audit_2026-10-27.md](adf_audit_2026-10-27.md)), the exponential
transformation and the 2D-1D work.

Seven independent reviews covered these areas. Each finding was reproduced
with a script or confirmed by direct derivation before it was accepted. GPU
execution was not available: the fused-kernel and other device-only paths
were reviewed by reading the code, and none of them was executed.

Severity: **C** = physics results silently wrong; **H** = wrong in a
realistic configuration, or a crash; **M** = medium; **L** = low.

## Findings and status

### Critical

| ID | Finding | Status |
|---|---|---|
| C1 | SDP1/SDPN/SPN transients put the implicitly eliminated delayed fission into the prompt `chi` instead of `chi_delayed`. With no perturbation, P(0.5 s) was 5.5–7.0. | Fixed. These solvers now use the same `DelayedEmission` weights as `TransientSolver`. |
| C2 | SPN-family transients on r-z grids did not apply the metric weight to the time source; an unperturbed core lost 45% of its power in one step. | Fixed. |
| C3 | `first_order_reactivity` left out the r-z source weight, giving −34.7 for a zero perturbation. | Fixed. It now shares the weighted helpers in `quasistatic`; SP3/SPN solvers are rejected with a clear error. |
| C4 | Floor divisions in the VVER-440 map placement shifted odd rows by one pitch; k converged about −420 pcm off the reference. | Fixed. The error is now −111, −92, −37, −14 pcm at refine 1, 2, 4, 8. The map is tested for 60° symmetry. |

### High

| ID | Finding | Status |
|---|---|---|
| H1 | A scalar `bc` on a cylindrical `Grid` applied boundary terms on the non-existent y faces (k = 0.0067 against 1.105). | Fixed. y faces are always reflective in r-z. |
| H2 | The power edit fell back to νΣf for materials without κΣf, which mixes units (a ~10¹¹ error in the power split). SPH homogenization and `read_xsec` also dropped κΣf. | Fixed. Mixing fissile materials with and without κΣf raises an error. SPH keeps κΣf. `read_xsec` stores κΣf = `energy_per_fission`·Σf (default 200 MeV). |
| H3 | Mixed-cell heat sink: h and T_sink were blended separately and then multiplied (750 K → 375 K). | Fixed. h·T_sink is blended as one product. |
| H4 | The SDPN adjoint used the forward spatial block, so adjoint k was 286 pcm off for SDP2 with Marshak. The same defect affected r-z diffusion with `symmetric_operator=False`. | Fixed. The adjoint now uses the transposed block with the source and read-out weights swapped. |
| H5 | Default SDP3 on BIBLIS did not converge (BiCGStab combined with Anderson). | Fixed. Non-symmetric SDPN blocks default to GMRES; explicit `bicgstab` uses tighter inner solves; Anderson falls back to plain iteration on stagnation. |
| H6 | IQS was explicit in the shape, and its guard never rejected a step, so the error was first order in `shape_dt`. | Fixed. IQS is now a predictor–corrector with replay and rejection; see [quasistatic_acceleration_plan.md](../quasistatic_acceleration_plan.md). |
| H7 | Unstructured two-point flux was inconsistent on non-orthogonal meshes (about −320 pcm, not converging). | Fixed. A normal-distance face weight plus a lagged non-orthogonal correction gives second-order convergence. `nonorthogonal_correction=False` restores the old scheme. |
| H8 | The Gmsh reader assumed exactly 2 tags per element and contiguous node ids. | Fixed. It handles any tag count and node ids through a map; MSH 4, binary and higher-order files are refused with a clear message. |
| H9 | Tri/hex lattices without a mask silently treated in-plane edges as reflective. | Fixed. Without a mask the outer edges use `bc`. With a mask, exposed faces use `mask_bc` and no one-cell void border is needed. |
| H10 | `Model.from_grid` put vacuum on the collapsed z axis of a 2-D grid (k = 0.0175 against 1.193). | Fixed. |
| H11 | `read_xsec` read `PLANE n` headers as map rows. | Fixed. 3-D files return per-plane maps (`n_planes`). |
| H12 | The adaptive BDF error estimate on the first step and after restarts was O(h); fast-spectrum steps failed at `min_dt`. | Fixed. It now uses an explicit-Euler predictor; see open item O2 for the LRA driver. |
| H13 | `TransientSNSolver` crashed with BDF3–6. | Fixed. |

### Medium and low

| Finding | Status |
|---|---|
| Doppler and expansion coefficients were blended separately from the cross sections in mixed cells (50× the correct ΔΣa). | Fixed. Each material's law is evaluated separately and then blended. |
| Steady coupling converged on the relaxed step rather than the residual, and returned an inconsistent (T, k, power). | Fixed. |
| κΣf did not follow feedback applied to νΣf. | Fixed. |
| `TabulatedFeedback` missed fuel used as the blend partner; `t_ref` ignored `reference`; `uncoupled_k` used the maximum `t_ref`. | Fixed. |
| `criticality_search` never confirmed convergence; `coupling_vertices` returned nothing without a mask. | Fixed. |
| `hpmr_feedback` ignored the value of `expansion_pcm_per_K`. | Fixed. `fuel_density_worth_pcm` is now required. |
| The coupled transient is explicit Lie splitting. | Documented in [coupling.md](../coupling.md). |
| Odd-degree Neumann preconditioners were indefinite on SP5/SP7 blocks. | Fixed. `damping="auto"` applies Lanczos-estimated Jacobi damping. |
| The dedicated SP3 "vacuum" boundary was a per-moment Robin condition, not the coupled Marshak condition (120–250 pcm). | Fixed. Marshak is the default for `sp3` on Cartesian grids; `marshak_vacuum=False` restores the old behaviour. |
| `marshak_vacuum` ignored active-mask faces. | Fixed. |
| Krylov solvers had inconsistent signatures; a zero right-hand side failed; BiCGStab used absolute breakdown thresholds. | Fixed. |
| The eigenvalue stopping test bounded the per-iteration change, not the error. | Fixed for plain power iteration and the adjoint; tolerances are scaled by (1 − ρ̂). |
| Excised cells kept their fission and scatter sources. | Fixed, including in `TransientSolver`. |
| Fixed-step event and BDF restarts happened on the wrong step (order 0.96). | Fixed. Restarts happen on the step leaving the event; `problem_at(t_e)` returns the pre-event state. |
| A Crank–Nicolson restart dropped Fⁿ (order 0.94). | Fixed. |
| Fixed-step BDF3–6 are only second order. | Documented, and tested as a documented cap; see open item O1. |
| Events close to grid points created sliver steps. | Fixed. Step ends within 1e-3 of a step width are snapped onto the event. |
| Per-material β crashed on the GPU (host conversion). | Fixed; not executed, since no GPU was available. |
| `on_step` received arrays that are overwritten by the next step. | Fixed. It now receives copies. |
| SDP1/SDPN rejected a per-family `chi_delayed`. | Fixed. |
| SPH: `_mixed_entries` crashed on per-cell volumes; `region_average` ignored volumes; Anderson returned a mismatched pair when not converged; empty regions crashed; `d_scaling` was not validated. | Fixed. |
| SPH D scaling depends on the normalisation convention. | Kept as-is by decision; documented as an open issue in [numerical_implementation_notes.md](../numerical_implementation_notes.md). See open item O5. |
| `HexLattice.add_axial_region` kept the drum absorber blend. | Fixed. |
| Drum absorber area fractions were biased (−17% to +4%). | Fixed. Areas are now computed exactly by sector intersection. |
| 3-D prismatic flux gave garbage pin powers. | Fixed. A `layer=` argument was added. |
| The Griffin reader zero-filled missing required tags. | Fixed. |
| `add_box` silently painted a different extent than requested. | A warning is now issued. |
| Hanging nodes other than 2:1 cut cells off from their neighbours. | Fixed in 2-D and 3-D. |
| The tri adjoint with discontinuity factors used the forward operator. | Fixed. |
| Axial faces between active and inactive cells had no Robin term. | Fixed (tri and hex). |
| `alpha_boundary=inf` gave NaN; quad centroids were wrong; 2-D meshes outside the xy plane were mishandled. | Fixed. |
| NaN albedo was accepted and NumPy scalar albedos were rejected. | Fixed. `albedo=inf` is now rejected, by decision. |
| The documented r-z power-density recipe was 2π/dy too large. | Fixed. |
| The preCICE example's `--warm-start` did nothing. | Fixed; not executed, since preCICE is not installed. |

## Changes that move results

| Change | Effect |
|---|---|
| Coupled Marshak boundary for SP3 | Brantley–Larsen SP3 k +53 pcm (now +114 pcm from the paper, was +61); BIBLIS SP3 +0.3 pcm; C5G7 unchanged. The SPH colorset test limit was loosened from 0.25 to 0.3, because SPH error rose from 18 to 65 pcm against the Marshak reference. |
| Exact drum absorber area | HP-MR 2-D drum worth 2642 → 2649 pcm; 3-D 2652 → 2658 pcm. |
| Non-orthogonal correction | Unstructured HP-MR drum worth 2462 → 2653 pcm, now matching the structured-triangle result. The locally refined mesh error went from about −10 pcm to +3 to +10 pcm. Orthogonal meshes are unchanged. |
| Error-based stopping for plain power iteration and the adjoint | Adjoint shape error about 100× smaller, at 25–80% more outer iterations. Forward solves with Anderson are unchanged. |
| Adaptive BDF startup estimator | Adaptive LRA-2D BDF2 P(3) changed by −3.6e-5 relative. TWIGL and fixed-step LRA are bit-identical. |
| Predictor–corrector IQS | HP-MR drum ramp error 1.52% → 0.46%, for 5–12% more wall time. `iqs_predictor_tol` now rejects the current interval instead of shortening the next one. |
| Tri/hex lattices without a mask | A non-reflective `bc` now applies on the outer in-plane edges. |

## Open items

- **O1.** Fixed-step BDF3–6 are second order, because their startup ramps the order at full step width. An order-preserving startup needs sub-steps, which would break coupled drivers that mirror the solver one step at a time (e.g. `benchmarks/lra.py`). Graded steps give order 3.25.
- **O2.** `benchmarks/lra.py` `adaptive_error_state` still predicts temperature with a constant on the first step, so that part of the LRA adaptive error estimate is O(h).
- **O3.** The calibrated guard settings in [benchmark-results/hpmr-iqs-gh200](../../benchmark-results/hpmr-iqs-gh200/) (e.g. `iqs_predictor_tol=0.02`) were chosen under the old IQS semantics. They need re-checking on the GH200.
- **O4.** GPU-only paths were not executed: fused kernels (including the transposed SDPN stack), the per-material β precursor start, and the graph-capture and CGS2 paths.
- **O5.** SPH D scaling (`D_SCALING="inverse"`) makes k depend on the normalisation convention (355 pcm in the test case, against 27 pcm for "direct"). Evaluate "direct" scaling with per-group flux-volume normalisation.
- **O6.** `benchmarks/hpmr_assembly._barycentric_samples` samples only the upright sub-triangle centroids, which biases pin material fractions.
- **O7.** `TransientSDPNSolver` still defaults its inner solver to BiCGStab. No trouble was seen there, but the eigen solver moved to GMRES for the same blocks.
- **O8 (resolved).** Brantley–Larsen SP3 now sits +114 pcm from the published value on the test's 80×80 mesh, against +61 pcm before. This is mesh error: the coupled Marshak SP3 extrapolates to the paper under refinement (`test_marshak_converges_to_paper`). The old per-moment Robin offset partly cancelled the mesh error, and the other methods, which still default to per-moment Robin, sit at +45 to +87 pcm.
- **O9.** The VVER-440 Gmsh validation tests and the FEMFFUSION reader tests are skipped without external files, so they were not rerun.
- **O10 (development).** Review the Brantley–Larsen cases and prepare a proper V&V report for them. The validation test compares each method on a single 80×80 mesh. Only SP3 and SDP1 are extrapolated to the paper, and the methods use different default vacuum treatments: coupled Marshak for the dedicated SP3, per-moment Robin for the others. The report should give a mesh-convergence study and Richardson-extrapolated k for every SPN/SDPN order under both vacuum treatments. It should compare these against Carreno et al. (2024) Table 6 and Fig. 3, separate discretization error from boundary-condition and angular-approximation error, and justify the test tolerances. Record it with the other benchmark studies under `benchmark-results/`, and link it from [verification_validation.md](../verification_validation.md).

## Regression tests added

All of these are in `tests/verification/`.

| Test file | Covers |
|---|---|
| `test_kinetics_reference.py` | Point-kinetics problems checked against `expm`; temporal orders; events; Crank–Nicolson restarts; BDF3 order cap; S_N with BDF3 |
| `test_transient_spn_equilibrium.py` | Unperturbed equilibrium for every SPN-family solver, Cartesian and r-z, with and without `chi_delayed` |
| `test_transient_events.py` | Event and restart handling |
| `test_sdpn_adjoint.py` | Adjoint against a dense left eigenvector |
| `test_solver_convergence.py` | Eigen solver convergence (H5, stopping test) |
| `test_spn_marshak.py` | Marshak boundary on masked geometries |
| `test_quasistatic_accuracy.py` | IQS convergence order and accuracy |
| `test_mesh_skewed.py` | Skewed-mesh convergence |
| `test_mesh_gmsh.py` | Gmsh reader |
| `test_tri_boundaries.py` | Tri/hex boundaries |
| `test_feedback_mixed_cells.py` | Feedback in mixed cells |
| `test_coupling_convergence.py` | Steady coupling convergence |
| `test_kappa_consistency.py` | κΣf consistency |
| `test_thermal_mixed_cells.py` | Heat sink in mixed cells |
| `test_model_api_audit.py` | Model API fixes |
| `test_drum_absorber.py` | Drum absorber area |
| `test_pin_power_3d.py` | 3-D pin power |

Existing test files were also extended. Together the new files run 231 tests in about 26 s on CPU.
