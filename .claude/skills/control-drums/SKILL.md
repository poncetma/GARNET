---
name: control-drums
description: Default modelling approach for any reactor with rotating control drums in GARNET (HP-MR, heat-pipe and gas microreactors, SIMBA/Empire-like cores, any HexLattice with set_drum). Use it whenever building, meshing, homogenizing, SPH-correcting, tabulating or running worth curves and transients for a core that has control drums, including drum-angle sweeps, drum worth, assembly-level SPH, pin-power reconstruction and cross-section tables in drum angle, even when the user does not mention the drums explicitly.
---

# Control drums in GARNET

A control drum puts a thin absorber arc (HP-MR: 1 cm of B4C on a 13.25 cm
drum) into a mesh whose cells are several centimetres across, and then rotates
it. Two things go wrong when that is treated casually, and both are silent:

- the absorber is lost or misplaced when it is thinner than a cell, which
  shows up as a wrong worth;
- the discretization error changes *irregularly* with the drum angle, because
  the arc cuts the cells differently at every angle. A worth curve, a
  cross-section or SPH table interpolated in angle, or a drum transient then
  inherits an error that no table spacing removes.

The defaults below are what the 2026 SPH work on the 2-D HP-MR measured
(docs/qa/sph_audit_2026-11-13.md §10.4; data in docs/qa/sph_audit_evidence/,
tools/assess_sph_angle_tables.py, tools/assess_sph_form_functions.py).

## 1. Volume-mix the arc at its exact area

Each cell the arc crosses is blended with the drum body at the exact area
fraction of the annular sector, computed in closed form. The absorber area is
then exact on any mesh, and the blend changes continuously as the drum turns.

| Core | Call |
|---|---|
| any hex core | `HexLattice.set_drum(rc, body, absorber, inner_radius, outer_radius, arc_deg, angle_deg)` (always volume-mixed) |
| HP-MR raster | `build_hpmr2d(..., absorber="polar")`, `build_hpmr3d(..., absorber="polar")` |
| any hex core, drum band refined | `HexLattice.build_mesh(levels, absorber="polar")` (`garnet.drum_mesh.refine_drum_band`) |
| HP-MR, drums locally refined | `hpmr_locally_refined_problem(..., absorber="polar")` |

The alternatives cost accuracy that tables cannot recover:

- **Centroid painting** (`absorber="raster"`, or `"centroid"` on the refined
  mesh) moves the arc in steps. On a mesh that resolves the drums it still
  left 39–55 pcm and about 2 % assembly power at every table spacing, where
  volume mixing left 6–20 pcm and 0.2–0.5 %. Below raster refine 4 the arc
  disappears entirely.
- **The hex-homogenized drum** (`absorber="smeared"`, or flux-weighted drum
  constants with `merge_drum=True`) makes the rotation invisible to the mesh.
  Its constants interpolate well only on a 15° table (13 pcm); at 45° it was
  111 pcm and 40 % worst pin error, at 90° 312 pcm. Kumar et al. (Ann. Nucl.
  Energy 230 (2026) 112111) needed 5° tables for the same reason.

Solvers that take one material per cell (`UnstructuredDiffusionSolver`) get the
blend through `HpmrMeshProblem.solver_materials()` or
`garnet.blend.per_cell_blended_materials`, which apply the raster solvers' rules
(linear cross sections, harmonic D).

## 2. Resolve the drums, locally

Drum cells of about 2 cm or less against a 1 cm arc make the model smooth in
drum angle. Refine only the band the arc sweeps:
`hpmr_locally_refined_problem(refine=4, levels=1 or 2, absorber="polar")`
splits every cell in that band into 4**levels triangles for all rotations,
with conservative hanging nodes. Measured on the 2-D HP-MR against refine-16
references, fuel-only SPH, linear 45° tables, worst between table points:

| Node model | Cells | `k`, assembly power | `k`, pin RMS, worst pin (with form functions) |
|---|---|---|---|
| uniform refine 4 (3.9 cm) | 5,280 | 95 pcm, 1.7 % | 174 pcm, 1.9 %, 15 % |
| uniform refine 8 (1.9 cm) | 21,120 | 14 pcm, 0.31 % | 13 pcm, 0.52 %, 4.0 % |
| uniform refine 16 (0.97 cm) | 84,480 | 5.5 pcm, 0.13 % | 4.3 pcm, 0.18 %, 2.2 % |
| refine 4, drum band x2 (1.9 cm) | 7,116 | 14 pcm, 0.32 % | 18 pcm, 0.79 %, 7.3 % |
| refine 4, drum band x4 (0.97 cm) | 14,460 | 6.5 pcm, 0.17 % | 4.9 pcm, 0.52 %, 3.8 % |

Read it as follows:

- For `k`, drum worth and assembly powers, a refine-4 core with the drum band
  refined x2 or x4 matches the uniform mesh with the same drum resolution at
  a third to a sixth of the cells. Uniform refine 4 stays at 68–95 pcm
  whatever the table spacing.
- For pin powers the rest of the core counts too: the form functions see the
  fuel at the base resolution. Refine the fuel to about 2 cm (uniform
  refine 8 or more) when pin RMS below 0.5 % matters.
- A coarser base does not compensate: refine 3 with the band x4 (6,210 cells)
  gave 48 pcm.

For any `HexLattice` core, `build_mesh(levels=...)` does the same (2-D,
vacuum boundary; `levels=0` reproduces the raster exactly), and
`garnet.equivalence.drum_core_zones` gives its equivalence zones (§4).

## 3. Check every drum model before quoting it

When you write any of these checks as a test, break it on purpose first and
confirm it fails (see the verification-tests skill).

1. **Absorber area**: the sum of `mix_weight * cell area` equals
   `n_drums * (arc_deg/2 in radians) * (R_out**2 - R_in**2)` to round-off.
2. **Angle convention**: 0° points the arc at the core centre (inserted),
   180° away (withdrawn); `HexLattice` measures from the lattice origin.
   Confirm with the eigenvalue, k(0°) < k(180°).
3. **Smoothness in angle**: sweep in steps of 7.5° or less and compare with a
   reference that is *finer* than the model near the drums. An error that
   alternates in sign between steps (+70, −126, +44, −160 pcm at 15°, 30°,
   45°, 60° on 7.7 cm cells) means the mesh does not resolve the arc.
   Comparing against a reference on the same mesh hides this: identical
   drum discretizations cancel, and refine-4 nodes looked smooth (2.8 pcm)
   against a refine-4 reference but were 68–95 pcm off against refine 16.
4. **Worth convergence**: quote a drum worth only with its mesh sequence.

## 4. Assembly-level homogenization and SPH

The workflow of Kumar et al. 2026, implemented on branch SPH-dev:

1. **Reference**: full core, vacuum boundary, finer than the node model near
   the drums; pin-resolved when pin powers are wanted (`build_hpmr2d_pins`).
   Scatter anisotropically as the data allow: `TriSNTransportSolver(...,
   scattering_order=2)` with the library's moments (`Material.sigma_s_moments`).
   Griffin ISOXML `Total` is the TRUE total, so isotropic S_N on it, or
   `D = 1/(3 Total)`, has no transport correction. Light moderators and
   reflectors (hydride, Be, graphite) scatter strongly forward: on HP-MR,
   isotropic S_N is +2400 pcm off P2, out-scatter-corrected isotropic S_N
   +210 pcm, and P3 differs from P2 by 1 pcm. Collapse zone constants from
   transport-corrected data (`transport_corrected`, `pin_reference(
   homogenize_tc=...)` in `tools/sph_hpmr_nodes.py`). Check the balance
   `Σa + ΣP0 = Σt` of every material: Be reflectors and drums count (n,2n)
   twice in P0, and GARNET's removal then loses the gain unless the
   absorption is made net (`net_absorption`; on HP-MR the drum worth moves by
   about 430 pcm at refine 4).
2. **Zones** (baseline since 2026-11-18), in `garnet.equivalence`:
   - **Any `HexLattice` core:** `drum_core_zones(lattice, model, n_sectors=8)` on its `build()` raster or `build_mesh()` drum-band mesh. It returns the zones, the node cells, the boundary zones and the drum specs.
   - **HP-MR:** `garnet.benchmarks.hpmr.hpmr_drum_sector_zones`; in the study tools, `sector_zoning_hooks(8, aligned=True, outer_arc_to_sector=True)`, or `SPH_DRUM_SECTORS=8 SPH_OUTER_BE=1 SPH_SECTOR_ALIGN=1` for `tools/assess_sph_df_tables.py`.
   - **Reference zones:** `inherit_zones` carries the node zones to a nested reference by containment; it is exact when the reference refines the node mesh by an integer factor.
   - **Node solve:** `SphDfNode.raster` or `SphDfNode.mesh` is the solve that `sph_bcf_correct` iterates.
   - **Fuel and Be hexes:** one zone per fuel assembly and per Be reflector hex.
   - **Inner/outer split:** every hex on the vacuum boundary is split into an inner and an outer zone (the first row of node cells).
   - **Drum hexes, inside:** each drum is one merged hex whose inner part becomes a core (within 10.25 cm of the drum centre) and 8 ring sectors fixed in space. The sector boundaries lie on the drum's outward line and every 45° from it, so the arc edges cross them at drum angles that are multiples of 45°. The arc moves through the sectors, so their absorber content changes continuously with angle.
   - **Drum hexes, outer layer:** it holds Be only; the absorber fraction of its cells goes to the ring sector at the same azimuth.
   - **Zone maps:** identical on node and reference rasters (exact nesting).

   **No zone with a vacuum face may hold B4C.** A uniform boundary zone that holds part of the arc cannot reproduce where the reference leaks, and SPH-DF has no solution at some angles (`tools/probe_sph_drum_zonings.py`).

   The older per-arc zoning, with a separate absorber-arc zone per drum that moves with the drum, has three problems:
   - its absorber factors swing between 60 and 590 with angle;
   - it has no solution at 165°;
   - interpolated in angle, it keeps a floor of about 200 pcm and 2 % pin RMS at any table spacing.
3. **Factors: SPH-DF** (Labouré et al., Ann. Nucl. Energy 128 (2019) 443), `sph_bcf_correct`:
   - SPH factors on every zone plus a boundary discontinuity factor on each outer zone, flux-volume normalized;
   - reference outgoing currents from `TriSNTransportSolver.region_partial_currents` (`load_or_solve_currents`), coarse leakage from `tri_boundary_leakage`.

   On HP-MR, with the sector zones, it reproduces `k` and every zone rate at all 13 table angles 0°–180°. Fuel factors are 0.84–1.18, drum factors 0.3–3.6, and boundary factors 0.1–4.2.
   - **Check feasibility first** (`tools/probe_sph_bcf_feasibility.py`, `tools/probe_sph_drum_zonings.py`). Without the inner/outer split, or with B4C in a boundary zone, the zero-flux leakage limit of the coarse model is below the reference in some groups. No solution exists then, and the boundary factors run to their upper cap.
   - **Small boundary factors are legitimate.** A boundary zone shadowed by the arc needs γ ≈ 1e-6, which is why the floor is exp(−30).
   - **Reference currents** must come from the S_N problem of the method (`load_or_solve_currents` builds it). Never use the problem returned by `pin_reference`, which carries transport-corrected homogenization data.
   - **Do not use flux-volume SPH alone** with a vacuum boundary: +400 to +770 pcm.
   - **Avoid fixed-k SPH** (`normalization="none"`): it reproduces `k` by distorting the fuel factors through the fuel–reflector interface (0.55–0.75; 0.06–20 with smeared drums).
4. **Node model**: drums resolved and volume-mixed as in §1–2, on a uniform
   raster (HP-MR: refine 8, 1.9 cm) or a drum-band mesh (`build_mesh`).
   Factors on coarse drum cells absorb the drum discretization error, which
   then does not interpolate in angle.
   - **With SPH-DF, a finer drum band adds no accuracy.** On HP-MR with 5° tables,
     `local:4:1:polar` (7,116 cells, band 1.9 cm) and `local:4:2:polar` (14,460,
     band 0.97 cm) match raster:8 (21,120 cells) to 5.1 pcm and 0.02 % pin RMS at
     all 20 test angles: the interpolation in angle dominates.
   - **Default node with SPH-DF: the locally refined drum band**
     (`build_mesh(levels=1)` on a refine-4 core, HP-MR `local:4:1:polar`).
     Its benefit is performance: the same accuracy at a third of the cells.
     Use it by default, above all for transients and 3-D cores. Keep a uniform
     raster only as a cross-check.
5. **Tables in drum angle**: zone cross sections, SPH factors (`apply_sph_factors`), boundary factors and per-group form functions (`pin_footprints` or `mesh_pin_footprints`, `group_form_functions`, `reconstruct_group_form_functions`), linearly interpolated. Tool: `tools/assess_sph_df_tables.py` (`generate`, `evaluate`, `attribute`).
   - **Use a 5° table with the sector zoning.** With the 1.9 cm node mesh and SPH-DF on the HP-MR, interpolating to the midpoints between table points kept `k` within 38 pcm, assemblies within 0.28 %, pin RMS 0.22 % and the worst pin within 2.0 %.
   - **Coarser tables lose accuracy quickly:** 10° gives 121 pcm and 1.0 % pin RMS, 15° gives 171 pcm and 1.2 %, 30° gives 670 pcm and 4.7 %.
   - **The coarse-table error is the drum sectors' cross sections.** The arc fills a 45° sector within 45° of rotation and the constants go from Be to near-black, so linear interpolation over-predicts the absorption between table points. Fuel, reflector and boundary factors interpolate essentially exactly.
   - **What did not help:** interpolating μΣ instead of μ and Σ, and a self-shielding rebuild of the sector constants.
   - **Judge interpolation errors against the reference's own non-linearity** in angle (`tools/reference_angle_ripple.py`). On HP-MR, between 5° table points, the SPH-DF `k` error correlates at −0.95 with the refine-16 reference's own interpolation residual. The off-midpoint worst pins (3–5 %) match the reference's own residual as well.
   - **Refine-16 S_N is too coarse for absolute accuracy.** Against refine 32 its `k` is 5–178 pcm low, the worth is 162 pcm high, and assembly powers are off by up to 2.7 % (`tools/compare_reference_refinement.py`). Its `k(θ)` also carries a mesh ripple of ±15–25 pcm. Generate production tables against refine 32 or finer, or against an arc-conforming mesh.
   - **Validate the spacing for each new core** against references at the table midpoints, and use `attribute` to see which class of quantity carries the error.
6. **Construction check**: at every table angle the reconstruction reproduces
   the reference pin rates to round-off; anything else is a bug.
7. **Transients**: `sph_kinetics(kinetics, factors)` gives the speeds v/mu;
   the uncorrected 1/v biased the generation time by +20.7 %.
   - **Tabulated model in time:** `SphDfTable.load(dir, geometry_at)`
     (`hpmr_sphdf_geometry` for HP-MR). `table.problem_at(theta_of_t)` returns
     the corrected zone materials, the absorber blend at θ(t) and the
     boundary factors as `bcf`, and `table.kinetics(base, θ0)` gives v/mu.
   - **On the drum-band mesh:** `TransientSolver(MeshGrid(mesh, alpha), ...,
     group_operator=MeshGroupOperator, eig_solver=MeshDiffusionEigenSolver,
     linear_solver="gmres")`. The hanging-node correction makes the operator
     non-symmetric, so CG and monolithic steps are rejected.
   - **Step settings on HP-MR-like cores:** `rebalance=True, anderson_depth=1,
     max_sweeps=4000, scatter_subsweeps=6, tol_step=1e-6`, as in
     `hpmr_transient_bench`. The defaults stall near criticality.
   - **Reference:** quasi-static S_N, with `sn_effective_kinetics` (forward and
     `adjoint=True` solves) and `adiabatic_point_kinetics`. Plan and status:
     `docs/sph_drum_transient_plan.md`.
   - **Kinetics data:** `hpmr_kinetics_11g()`, the library's own (β = 680 pcm).

Transport effects concentrated in the drums can also be handled by the hybrid
solvers (`hpmr_transport_mask`, `hybrid_mask=`, `sn_mask=`); see the benchmark
skill for their mask-window caveats.

## 5. Run the studies on GPU

Drum sweeps and full-core references are GPU jobs; the login node kills long
processes. Stage the checkout into its own directory under
`/data/scratch/shared/poncet_m/`, then

```bash
GARNET_REPO=<stage>/repo \
GARNET_PYTHON_BIN=/data/scratch/shared/poncet_m/garnet-gh-py313-v1/bin/python \
GARNET_EXAMPLE=<stage>/repo/tools/assess_sph_angle_tables.py \
GARNET_ARGS="16 local:4:1:polar gpu" \
sbatch --partition=gh-hourly --time=00:59:00 <stage>/repo/slurm/run_garnet_gh.sh
```

Node models for both tools: `raster:R`, `smeared:R`, `local:R:L:polar|centroid`
(tools/sph_hpmr_nodes.py). Use `gh-daily` for runs over an hour (a 25-angle
table on an 84k-cell node mesh took 49 minutes). Logs land in
`/data/scratch/shared/poncet_m/garnet-slurm-debug/garnet-gh-<jobid>.log`.
