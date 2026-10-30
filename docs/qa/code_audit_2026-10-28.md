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
| Blockwise adaptive BDF error norm (O11, 2026-10-29) | Every adaptive BDF run changes. LRA-2D refine 1, BDF2 at rtol 1e-3: 453 → 539 steps, peak-power error −1.23e-2 → −5.7e-3, P(3 s) unchanged to 1.3e-5. The H12 fast-step case: 49 → 53 steps, P(t_end) error 1.8e-4 → 3.7e-5. The stored LRA adaptive-BDF report (concatenated norm) is not regenerated. |
| Exact HP-MR assembly fractions (O6, 2026-10-29) | Periodic S_N assembly k∞ 1.24523 → 1.20632 at refine 4 (−3891 pcm) and 1.18232 → 1.18414 at refine 8 (+183 pcm); the refine-4/8 spread fell from about 6300 to 2200 pcm. No stored result used the assembly k∞. |
| Diffusion transient step convergence (O12, 2026-10-30) | Every run on the fixed-point diffusion step changes: `coupled_transient`, `Model` transients, and IQS shape solves and fallbacks. On the HP-MR at 2-D refine 2 and the default `tol_step`, the P(end) error went from 2.8e-4 to 8.3e-6 (fast step) and from 6.4e-4 to 2.0e-6 (slow insertion), with 5–19× fewer sweeps. Unperturbed cores now drift by their steady state's residual (1.9e-7 on the HP-MR from the default steady tolerance) instead of holding at exactly 1. The monolithic step and adaptive BDF are unchanged. |
| Marshak vacuum default for SPN/SDPN (after O10, 2026-10-30) | SP5, SP7, SDP2 and SDP3 eigen, transient and noise solvers now apply the coupled Marshak vacuum by default on Cartesian grids with a vacuum face and no hybrid mask (the dedicated SP3 already did). Brantley–Larsen k +53 pcm; BIBLIS +0.9 pcm; problems without vacuum faces are unchanged. SDP2 on such problems loses its CG path and runs GMRES (5.8× slower on Brantley–Larsen at n = 160). The dedicated SDP1 solver keeps the per-moment Robin term, which `marshak_vacuum=False` restores everywhere else. |

## Open items

- **O1.** Fixed-step BDF3–6 are second order, because their startup ramps the order at full step width. An order-preserving startup needs sub-steps, which would break coupled drivers that mirror the solver one step at a time (e.g. `benchmarks/lra.py`). Graded steps give order 3.25.
- **O2 (resolved 2026-10-29).** `benchmarks/lra.py` `adaptive_error_state` still predicts temperature with a constant on the first step, so that part of the LRA adaptive error estimate is O(h). With one accepted thermal state (t=0 and the 2 s restart) it now predicts T + h·(α/E_f)·q(φⁿ) (`LRAAdiabaticState.predicted`). The rate depends only on the flux, which is continuous at the rod-law corner, so no right-limit probe is needed. At the restart this defect is 35–2000× below the constant one (`test_adaptive_lra_restart_uses_the_thermal_startup_predictor`). Results do not move: adaptive LRA-2D (refine 1; BDF2 at rtol 1e-3, 1e-4 and 1e-5; automatic-order BDF5 at 1e-4) keeps the same accepted and rejected step counts, and P(3 s) changes by at most 1.5e-12 relative. The reason is O11.
- **O3 (resolved 2026-10-30).** The calibrated guard settings in [benchmark-results/hpmr-iqs-gh200](../../benchmark-results/hpmr-iqs-gh200/) (e.g. `iqs_predictor_tol=0.02`) were chosen under the old IQS semantics. They need re-checking on the GH200. The re-check repeated the August command on gh-daily. It ran guarded IQS at `iqs_predictor_tol` 0.005, 0.01 and 0.02, fixed cadence, and a full-diffusion reference, this time to 200 s. The drums now insert +0.309 $ (August: +0.248 $). The profile is kept. Where the reference is reliable, the guarded maximum error is 0.181%, against 0.173% in August; fixed cadence is 8.2% low at the end of the ramp. Guarded costs 2.0× the fixed-cadence time (August: 1.26×), and full diffusion costs 12.7× the guarded time. `iqs_predictor_tol` never binds at 0.01 or 0.02: those two runs are bitwise identical, with a largest disagreement of 0.64%. At 0.005 it costs 14% more shape solves and changes the power by 5.3e-6. The residual guard drives all 63 early updates and all 12 fallbacks, which fall during the drum ramp. After the peak, guarded drifts to −0.18% while fixed cadence stays within 0.054%; the cause was not investigated. The reference itself stalls in places (O12). The run records and `tools/assess_hpmr_iqs_guards.py` (which now flags reference stalls) are in the same directory.
- **O4 (resolved 2026-10-29).** GPU-only paths were not executed: fused kernels (including the transposed SDPN stack), the per-material β precursor start, and the graph-capture and CGS2 paths. [test_gpu_paths.py](../../tests/verification/test_gpu_paths.py) now checks each against an independent CPU or uncaptured path. On a GH200 it passed with the other GPU-gated modules (123 passed, 0 skipped), and the full suite with slow tests passed on the same node type (1283 passed, 4 skipped for missing optional packages). See [benchmark-results/gpu-paths-gh200](../../benchmark-results/gpu-paths-gh200/). The A100 has not run them.
- **O5.** SPH D scaling (`D_SCALING="inverse"`) makes k depend on the normalisation convention (355 pcm in the test case, against 27 pcm for "direct"). Evaluate "direct" scaling with per-group flux-volume normalisation.
- **O6 (resolved 2026-10-29).** `benchmarks/hpmr_assembly._barycentric_samples` sampled only the upright sub-triangle centroids, which biased pin material fractions. The sampler is gone: per-cell constituent areas are now exact (`pin_layer_fractions`, the closed-form intersection behind the drum fix), exposed as `AssemblyProblem.cell_fractions`, and `samples` is ignored. They sum to the analytic pin areas to 1e-12 on any mesh (`test_hpmr_assembly_pin_power.py`). Still open, and not part of this fix: the solver sees two constituents per cell, so cells holding core, shell and graphite lose their smallest part (achieved-fraction error ≤0.0074 at refine 4, ≤0.0030 at 8, ≤5e-5 at 16). See [numerical_implementation_notes.md](../numerical_implementation_notes.md#exact-pin-resolved-assembly-fractions).
- **O7 (resolved 2026-10-29).** `TransientSDPNSolver` still defaults its inner solver to BiCGStab. No trouble was seen there, but the eigen solver moved to GMRES for the same blocks. Measured on non-symmetric SDP3 transients (a Brantley–Larsen-style core at 20² and 40², BIBLIS at 34² and 68²): BiCGStab converged every step, matched GMRES in trajectory error at every `tol_step`, and took 27–32% less CPU time. GMRES saved up to 30% of applies, but each apply cost 1.4–2.0× more. The eigen solver's stall came from a loose inner tolerance (`0.1·src_err`); the transient's is `1e-3` of the source change. BiCGStab therefore stays the default, without the eigen solver's ×0.1 tightening (no accuracy gain measured), and a new `linear_solver` keyword selects GMRES or any other solver. See [benchmark-results/transient-sdpn-inner-solver](../../benchmark-results/transient-sdpn-inner-solver/) and `test_transient_explicit_gmres_matches_default_bicgstab`. GPU cost was not measured.
- **O8 (resolved).** Brantley–Larsen SP3 now sits +114 pcm from the published value on the test's 80×80 mesh, against +61 pcm before. This is mesh error: the coupled Marshak SP3 extrapolates to the paper under refinement (`test_marshak_converges_to_paper`). The old per-moment Robin offset partly cancelled the mesh error, and the other methods, which still default to per-moment Robin, sit at +45 to +87 pcm.
- **O9 (resolved 2026-10-29).** The VVER-440 Gmsh validation tests and the FEMFFUSION reader tests were skipped without external files, so they were not rerun. Both now locate the checkout through [tests/_femffusion.py](../../tests/_femffusion.py) (`$FEMFFUSION_DIR`, then a sibling `FEMFFUSION`, then `~/FEMFFUSION`) and skip per missing file. Against a FEMFFUSION checkout all 7 pass: 2 in `test_vver440.py` (Gmsh mesh k within 1e-3 of the reference; loose-source regression) and 5 in `test_femffusion_io.py`.
- **O10 (resolved 2026-10-30).** Review the Brantley–Larsen cases and prepare a proper V&V report, extended to the paper's C5G7 case. The report is [carreno2024-sdpn-vv.pdf](../../benchmark-results/carreno2024-sdpn-vv/carreno2024-sdpn-vv.pdf), built from `tools/assess_carreno2024.py` runs on CPU and GH200. Brantley–Larsen converges at second order; with the paper's Marshak vacuum the mesh limits equal Table 6 within 0.6 pcm for SP1–SP7, SDP1 and SDP2. The paper's SDP3 values were computed with the tables in the authors' FEMFFUSION code; with those tables garnet reproduces them within 0.2 pcm. garnet's re-derived SDP3 tables give −9.9 pcm, and the paper's printed Eq. (A.6) gives +23.6 pcm. Pin-resolved C5G7 (new opt-in `exact_area` and `outer_bc` in `build_c5g7_2d`) reproduces the paper's pin-power AVG and RMSE against the NEA MCNP reference within 0.04 points. Its eigenvalues are 11–22 pcm below the paper's, a model difference whose source is not identified. The 80×80 test offsets are mesh error (+87 to +132 pcm) partly cancelled by the per-moment Robin default (−52 to −73 pcm). The mesh-limit test now covers every method at 4 pcm, and a slow pin-resolved C5G7 test was added. Following the report, the coupled Marshak vacuum is now the default of every SPN/SDPN eigen, transient and noise solver wherever it is available, with the per-moment Robin term kept as `marshak_vacuum=False` (see Changes that move results). One recommendation awaits a decision: make exact pin areas the default in `build_c5g7_2d` (the sampled default moves k by +226 pcm at 4 cells per pin).
- **O11 (found and resolved 2026-10-29, while closing O2).** The adaptive BDF error norm is one Euclidean norm over the concatenated state, in absolute units (Cherezov et al. Eq. 44 as implemented). In the coupled LRA run the flux (normalized to P/P0) and the temperature (K) differ by orders of magnitude, so neither the temperature nor the precursors take part in step control after the excursion: at the 2 s restart the temperature is 1e-16 of the defect and 6e-12 of the scale. At t=0 the temperature is about 100% of the scale, so the first step's flux defect is judged against rtol·‖T‖, about 950× looser than against rtol·‖φ‖. Resolved: `adaptive_bdf` now defaults to `error_norm="blockwise"`, Eq. (44) per field with the largest deciding; `"concatenated"` reproduces the old behaviour bitwise. On LRA-2D (refine 1) blockwise takes 10–21% more steps at the same rtol; at matched work its peak-power error is about 1.5× lower for BDF2 and 3.6–4.9× lower for BDF5 at rtol 1e-3 and 1e-4. P(3 s) and T(3 s) are unchanged. On neutron-only kinetics with exact references it costs 0–13% more steps with equal or lower power error. See [benchmark-results/adaptive-bdf-error-norm](../../benchmark-results/adaptive-bdf-error-norm/) and `test_blockwise_bdf_error_norm_is_per_field_and_unit_invariant`. Not done: volume-weighted componentwise RMS and `T − T_ref` thermal scaling.
- **O12 (found and resolved 2026-10-30, while closing O3).** In the 200 s HP-MR full-diffusion reference of O3 (refine 4, nz 20, 11 groups, dt 0.05 s, default `tol_step=1e-6`), the power stayed flat between the 0.5 s feedback updates and caught up in jumps over 31–50 s and 165–200 s. That reference is therefore good only to about 0.4% near the peak and 0.1% in the tail. Two defects in the fixed-point step of `TransientSolver` caused it. First, a step's first sweep solved its group equations at `rtol` 1e-6 of the group source, whatever `tol_step` was. When the step's own source change was smaller than that, PCG returned the warm start, and the step was accepted unchanged at any `tol_step`. Second, the stopping test was the source change between sweeps alone. Near criticality that understates the amplitude error by 1/(1 − k_prompt), about 150× on the HP-MR, and the rebalance that removes that mode switched off below 10·`tol_step`. Fixed: the first sweep solves at the inner floor; a step must also pass |f − 1| < `tol_step`, the amplitude error from the whole-core balance the rebalance already forms; and the rebalance runs on every sweep. On the 11-group HP-MR at 2-D refine 2, against an independent monolithic solve at the default `tol_step`, the P(end) error fell from 2.8e-4 to 8.3e-6 for a fast step (96 → 20 sweeps a step) and from 6.4e-4 to 2.0e-6 for a slow insertion (117 → 6). On the slow insertion the old solver was still 4.0e-4 off at `tol_step` 1e-10. Wall time fell 3.4× and 18× (best of 3). The error is now about 8×`tol_step`. On an exact point-kinetics reference at 0.002 $, the old solver was 2.2e-4 off at any `tol_step`; the new one leaves only the time-discretization error. Six equilibrium tests had held P = 1 only because their steps went unsolved (one sweep, no inner iterations); they now start from a tight steady state. IQS shape solves and fallbacks use the same step, so every O3 run predates the fix. The monolithic step, and so adaptive BDF, is unaffected. See [benchmark-results/transient-step-convergence](../../benchmark-results/transient-step-convergence/) and `test_slow_steps_are_solved_not_accepted_unchanged`.
- **O13 (found 2026-10-30, while fixing O12; open).** `TransientSDP1Solver`, `TransientSDPNSolver` (SPN and SDPN) and `TransientSNSolver` share the first-sweep inner tolerance and the change-only stopping test. On a homogeneous 0.002 $ step (dt 0.05 s, `tol_step` 1e-6), SDP1 and SDP3 accepted 58 and 51 of 80 steps with no change and were about 1e-4 off a tight solve. The diffusion fix does not carry over directly, because these solvers have no whole-core balance over their moment rows yet. S_N was not characterized: its step does not converge below about 1e-7 on the test slab.
- **O14 (found 2026-10-30, while closing O10; open).** With Marshak vacuum the SDP1, SDP2 and SDP3 eigenvalue blocks are non-symmetric and solved by GMRES with a Jacobi preconditioner. On pin-resolved C5G7 at 32 cells per pin (2.7 million cells, GH200), all three stopped in the first outer iteration: GMRES did not reach its tolerance in 5000 iterations (residual 1.3–4.8 times the target). Up to 24 cells per pin they converged, with 121–161 outer iterations against 57–73 for the CG-solved SPN blocks. Options: precondition with `spn_precondition` (available, not tried here) or a stronger preconditioner, and decide whether an inner solve that misses its tolerance in an early outer iteration should abort the eigenvalue solve. See the solver-cost section of the [O10 report](../../benchmark-results/carreno2024-sdpn-vv/carreno2024-sdpn-vv.pdf).

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
