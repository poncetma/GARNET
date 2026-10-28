# Test suite layout

The default suite contains verification and validation. Preserved low-level
tests under `legacy/` are inactive unless `--run-legacy` is supplied.

## `verification/` — is the *mathematics* solved correctly?

Checks against **exact references**: analytic eigenvalues (bare-box buckling,
infinite-medium k∞, the 1D slab limit), discretization *order* (error must
fall 4× per mesh doubling), exact invariants (symmetry planes, reflective ⇒
k = k∞, extruded-2D ⇒ 3D equality), cross-solver equivalence on identical
meshes, spec parsing, and exact file transcription (FEMFFUSION readers).
These tests would fail if the numerics were wrong, regardless of any physics
data. Fast; no external data needed (the FEMFFUSION reader tests skip
without a checkout).

| File | Pins down |
|---|---|
| `test_diffusion_analytic.py` | Cartesian FV vs exact bare-reactor solutions, 2nd-order convergence |
| `test_cylindrical.py` | cylindrical (r-z) FV vs the exact bare-cylinder (Bessel-mode) eigenvalue, 2nd-order convergence, inert axis bc |
| `test_sp3_analytic.py` | SP3 vs the exact analytic SP3 eigenvalue |
| `test_boundary_conditions.py` | bc spec forms; symmetry-plane invariant |
| `test_hex_lattice.py` | hex topology, k = k∞ invariant, BC ordering |
| `test_tri_prisms.py` | tri-z extrusion: slab limit, order, extruded-2D = 3D |
| `test_axial_diffusion.py` | independent column matrices, diffusion-operator equivalence, Robin slab convergence, mixed boundaries, and conservative axial currents |
| `test_sn_2d1d.py` | planar S_N / axial diffusion: independent fixed-source/angular eigenproblems, dimensional limits, upscatter, signed sources, step/SCB currents, thin layers, blended fission spectra, normalized currents and convergence reporting |
| `test_unstructured_mesh.py` | unstructured FV ≡ structured FV on the same mesh |
| `test_transient_point_kinetics.py` | transient stack vs the exact point-kinetics ODE |
| `test_delayed_spectra.py` | shared and family-specific delayed spectra vs independently assembled two-group neutron/precursor equations, with BDF1/BDF2 and both diffusion step solvers |
| `test_transient_model_api.py` | shared Cartesian/r-z/triangular histories, event schedules, accepted-step callbacks, initial-state compatibility and cylindrical coupling |
| `test_exponential_damping.py` | damped transformed BDF2, independent kinetics/stiff-mode references, six precursor families, unequal widths, event/fallback guards and coupled API regression |
| `test_exponential_adaptive.py` | physical-state FTM error control, complete trial rollback, volume/mask norms, events, six-family Radau and localized multigroup matrix-exponential references |
| `test_exponential_output.py` | physical midpoint/chord width control, rapid-growth power/integral refinement, constant-power spectral redistribution, changing-fission power output and accepted-half-step replay on three geometries |
| `test_exponential_estimator.py` | unequal-width physical-history prediction, unchanged audit trajectories, two-solve work, event/rejection replay, guarded alignment and discarded BE-fallback work |
| `test_exponential_estimator_ramp.py` | state-driven history rejection on a smooth insertion, same-trial oracle failures, independent Radau error and accepted-half-step replay |
| `test_exponential_twigl.py` | independent spatial generator vs kernel actions, augmented-exponential state/integral reference, ramp-corner continuity, conservative mesh restriction and all physical gates for six transient methods |
| `test_linalg_krylov.py` | GMRES/BiCGStab options: match CG on SPD systems through every solver entry point; solve a convection-dominated non-symmetric system where CG provably diverges; the non-symmetric divergence-form cylindrical stencil (`symmetric_operator=False`) equals the SPD-weighted one |
| `test_cartesian_adf.py` | Cartesian ADF face law vs independently assembled matrices (masks, transpose, currents, traces, factor scaling, invalid input); dense two-group forward/adjoint eigenpairs; ADF kernel-source edits; on a GPU, fused vs unfused vs CPU stencils and GPU vs CPU solves |
| `test_adf_slab_reference.py`, `test_dallosso_adf_reference.py` | reference-factor equivalence against analytical heterogeneous one- and two-group slabs |
| `test_colorset_adf.py` | C5G7 colorset ADF workflow: homogenized isolated assemblies keep k∞; symmetric library factors; node-level reference XS + DF reproduce the fine reference exactly, and inadmissible (negative-trace) nodes are reported |
| `test_femffusion_io.py` | .xsec/XML readers vs the files' exact contents |
| `test_conduction.py` | conduction vs the exact sinked-slab `cosh` solution (2nd order, Dirichlet and Robin, Cartesian and tri-prism); the discrete energy balance as an exact identity |
| `test_power_density.py` | power normalizes to the rated watts; independent of the eigenvalue flux's arbitrary scale; kappa- vs nu-weighting; mixing |
| `test_feedback.py` | temperature feedback: negative sign, linearity in the coefficient, Doppler/expansion separability, and a zeroed feedback bit-identical to no hook |
| `test_coupling.py` | the coupled fixed point: exact limits (zero feedback, zero power), contraction, conservation, accelerator-independence |

## `validation/` — does it reproduce *published reactor problems*?

Community benchmark cores solved end-to-end and compared to literature /
reference-code results. The problem definitions **and their reference
values** live together in `garnet/benchmarks/` (e.g.
`garnet.benchmarks.twigl.P_REFERENCE`); tests import them from there, so
there is a single source of truth for every published number.

| File | Benchmark | Reference |
|---|---|---|
| `test_c5g7.py` | OECD/NEA C5G7 MOX 2D | transport k = 1.18655 |
| `test_biblis_iaea.py` | BIBLIS 2D, IAEA 3D | published k's, FEMFFUSION |
| `test_vver440.py` | VVER-440 2D (tri + Gmsh mesh) | FEMFFUSION k = 1.00349 |
| `test_twigl_langenbuch.py` | TWIGL 2D, Langenbuch/LMW 3D kinetics | published power histories |
| `test_anl7416_8a1.py` | ANL-7416 Problem 8-A1 (r-z kinetics) | book k's + Exhibit A power trace |
| `test_hpmr.py` | HP-MR microreactor 2D/3D | behavioural (placeholder XS): symmetry, drum worth, mesh stability |
| `test_coupled_precice.py` | HP-MR neutronics/thermal coupled through preCICE | the internal `CoupledSolver` — exact mapping, lockstep iterates, shared fixed point. Skips without `pyprecice`; run it in the env that has it (see `docs/coupling.md`) |

Related directories: `garnet/benchmarks/` holds the benchmark *problem
builders + reference constants* (importable library code);
`examples/` holds runnable demo scripts (`speed_benchmark.py` is the
CPU-vs-GPU performance harness, not a physics benchmark); `dev-refs/` holds
third-party reference inputs used to derive data, never imported.

## API migration and preserved tests

The cylindrical, transient point-kinetics, per-material kinetics, triangular
kinetics-theory, TWIGL/LMW and ANL-7416 modules use model methods wherever the
calculation fits the diffusion API. Specialized low-level engine tests remain
active. All original test functions and assertions are retained.

Their pre-migration versions are stored byte-for-byte under
`legacy/verification/` and `legacy/validation/` (62 parameterized cases).
These are a reproducible comparison with the earlier entry points, not a
second default copy of each test:

```bash
python -m pytest tests                         # current API and engine checks
python -m pytest tests/legacy --run-legacy      # preserved low-level checks
python -m pytest tests --run-legacy             # both sets
```

The usual `slow` marker still applies. `-o addopts=` includes slow cases;
`--run-legacy` controls archived collection independently. See the
[transient guide](../docs/transient_api.md#test-migration-and-evidence) and
[V&V report](../docs/verification_validation.md#transient-api-and-cylindrical-check).
