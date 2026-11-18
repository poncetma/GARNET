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

The local refinement and the assembly zone map are written for HP-MR (its
site lists and drum geometry) and in 2-D. For another drum core, port them to
`HexLattice` first, or use a uniform `refine` that meets the cell size.

## 3. Check every drum model before quoting it

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
2. **Zones** (baseline since 2026-11-17):
   - one per fuel assembly, one per Be reflector hex, and one each per drum body and drum absorber arc (`hpmr_assembly_zones(per_hex=True)`, same numbering on every mesh);
   - every hex on the vacuum boundary split into an inner and an outer zone (`split_boundary_layer`, carried to the nested reference raster by `transfer_boundary_layer`, `tools/sph_hpmr_nodes.py`).

   Per-drum zones matter: corner and mid-edge drums need absorber factors a factor of 2–2.5 apart.
3. **Factors: SPH-DF** (Labouré et al., Ann. Nucl. Energy 128 (2019) 443), `sph_bcf_correct`:
   - SPH factors on every zone plus a boundary discontinuity factor on each outer zone, flux-volume normalized;
   - reference outgoing currents from `TriSNTransportSolver.region_partial_currents` (`load_or_solve_currents`), coarse leakage from `tri_boundary_leakage`.

   On HP-MR it reproduces `k` and every zone rate at both drum states, with fuel factors of 0.88–1.17.
   - **Check feasibility first** (`tools/probe_sph_bcf_feasibility.py`). Without the inner/outer split the zero-flux leakage limit of the coarse model is below the reference in some groups, no solution exists, and the boundary factors run to their cap.
   - **Do not use flux-volume SPH alone** with a vacuum boundary: +400 to +770 pcm.
   - **Avoid fixed-k SPH** (`normalization="none"`): it reproduces `k` by distorting the fuel factors through the fuel–reflector interface (0.55–0.75; 0.06–20 with smeared drums).
4. **Node model**: drums resolved and volume-mixed as in §1–2. Factors on
   coarse drum cells absorb the drum discretization error, which then does
   not interpolate in angle.
5. **Tables in drum angle**: zone cross sections, factors
   (`apply_sph_factors`) and per-group form functions (`pin_footprints` or
   `mesh_pin_footprints`, `group_form_functions`,
   `reconstruct_group_form_functions`), linearly interpolated. With resolved
   drums, 45° tables kept `k` within 5.5–14 pcm and 15° tables within
   1.2–11 pcm. Validate the spacing for each new core with the two tools
   above, on a reference angle step finer than the table.
6. **Construction check**: at every table angle the reconstruction reproduces
   the reference pin rates to round-off; anything else is a bug.
7. **Transients**: `sph_kinetics(kinetics, factors)` gives the speeds v/mu;
   the uncorrected 1/v biased the generation time by +20.7 %.

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
