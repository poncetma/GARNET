# Plan: arc-conforming S_N reference mesh for the HP-MR control drums

Status: proposed 2026-11-20, not started. Until it exists, drum-transient
comparisons use a smooth fit of the refine-32 S_N k(θ) over each manoeuvre
window (`tools/hpmr_drum_transient.py compare`, audit report §10.8).

## Why

The HP-MR S_N reference is the pin-resolved core on the structured
triangular raster (`sn-p2:4:16`, refine 32). Each drum's 1 cm B4C arc
(12.25–13.25 cm, 90°) is volume-mixed into the raster cells it crosses.

- **Mesh size against the arc:** triangles of 0.48 cm side, against a 1 cm arc.
- **Mostly smeared:** of the 3,438 cells the arc touches at 90°, 3,121 are only partly covered.

As the drum turns, the covered fractions change cell by cell, and with them the
error of homogenizing a black absorber with beryllium (lost self-shielding).
The result is a mesh ripple on k(θ):

- **Over the 5° grid:** 7.3 pcm RMS about a smooth curve, worst −20 pcm.
- **Against the 2.5° chord:** +27 pcm at 88.75°, inside the operating window of
  cases A–C. The 5° chord gives about the same, so this is not curvature.
- **Effect on the comparison:** the SPH-DF model with 2.5° tables is within
  about 3–7 pcm of a smooth S_N curve, so the reference's own ripple now
  dominates SPH-DF − S_N. A $0.25 manoeuvre turns 10 pcm into several percent
  of power.

Refining the raster reduces the ripple only slowly: refine 16 → 32 halved it.
Each refinement also multiplies the cost by 4.

An arc-conforming mesh removes the cause. Cells are pure B4C or pure Be, and the
geometry the drum sees is the same at every angle up to a rigid rotation.

## Design

### Geometry: a rotating polar disk inside each drum hex

- **Outside the drum hexes:** keep the refine-32 raster unchanged. Fuel, heat
  pipes, moderator, reflector, pin footprints and zones are all as now.
- **Inside each drum hex:** replace the raster triangles with three regions.
  1. **The rotor:** a polar disk of radius R_rot, between the drum's outer radius
     13.25 cm and the hex's inscribed radius.
     - Rings at exactly 12.25 and 13.25 cm (the arc), several rings across the
       B4C (graded, for self-shielding), coarser rings in the Be body, and a
       small unstructured or fan core at the centre.
     - Sectors every Δφ, triangulated.
     - The disk turns rigidly with the drum.
  2. **The stator ring:** between R_rot and the hex boundary, a fixed mesh. Its
     inner boundary carries the same N_s = 360°/Δφ equally spaced nodes as the
     rotor's rim. Its outer boundary uses exactly the raster's nodes on the hex
     edges, so the coupling to the rest of the core is conforming.
  3. **The rotation rule:** for drum angles that are multiples of Δφ, the rotor
     rim nodes coincide with the stator nodes, so the whole mesh is conforming
     at every such angle. No sliding interface and no remeshing are needed;
     the rotor's cells are only renumbered by the rotation.
- **Arc ends:** the arc's end faces at ±45° fall on sector lines whenever the
  drum angle is a multiple of Δφ.
  - Δφ = 0.625° (N_s = 576) covers every table and test angle used so far,
    1.25° multiples included.
  - Rim edges are 0.15 cm, finer than the raster.
- **Mesh generation:**
  - The rotor and the stator's ring are structured polar layers, written directly.
  - The transition from the stator's polar ring to the hexagon boundary needs a
    constrained triangulation. Options: `pip install gmsh` (GARNET already reads
    gmsh 2.x meshes) or `triangle`. Alternatively, a hand-written advancing-front
    transition, since the geometry is one fixed shape reused for every drum.

### Unstructured S_N engine

The triangular S_N engine works only on the structured lattice:
- its neighbours, normals and SCB corner geometry are lattice formulas;
- its level-scheduled GPU sweep and the CMFD read the lattice edge tables.

The new engine, `MeshSNTransportSolver` on `garnet.mesh.Mesh` triangles, keeps
everything that does not depend on geometry:
- quadrature;
- P_L moments and the spherical-harmonic basis;
- the within-group source iteration;
- the adjoint;
- the transient adapter (`t_*`), including `t_update_materials`, which on this
  mesh means renumbering the rotor cells.

**New pieces:**
1. **Corner geometry from the mesh:** per cell, corner sub-volumes, external
   half-edges (length, normal, neighbour corner across the face) and internal
   faces. The SCB corner blocks per (ordinate, cell) are then precomputed and
   inverted exactly as now.
2. **Sweep order from the mesh adjacency.** `_topological_sweep_levels` already
   takes generic dependency lists. Conforming triangle meshes can in rare cases
   produce a cycle for some ordinate, so the build detects cycles (strongly
   connected components) and lags the minimal set of faces, with a test.
3. **DSA on the mesh:** the existing `MeshGrid` diffusion operator, including
   its non-orthogonal correction, used as the DSA preconditioner. It only needs
   to be effective, not consistent.
4. **CMFD on mesh faces:** currents folded from ψ per face; drift-corrected
   coupling per face; the aggregated variant with aggregates kept to the rotor
   and stator rings.
5. **Rotation without rebuild:** the rotor's corner blocks, level tables and
   material assignment are permuted when the drum turns by k·Δφ. Only the
   rotor-to-stator rim links change. This replaces volume mixing in transients:
   drum angles off the Δφ grid are not representable, and a transient uses
   θ(t) on that grid (0.625° steps at 90°/s are 7 ms, about the transient's
   time step).

### Reference outputs unchanged

- **Pin footprints:** `mesh_pin_footprints`.
- **Zone transfer to the SPH-DF nodes:** `inherit_zones`, by containment.
- **Kinetics:** the forward/adjoint kinetics (`sn_effective_kinetics`) is
  geometry-independent.

So `tools/hpmr_pin_reference.py`, `assess_sph_df_tables.py` and
`hpmr_drum_transient.py` switch reference by a method tag.

## Verification (each check broken on purpose)

1. **Engine equivalence.** The unstructured engine on a mesh made from the
   refine-8 raster reproduces `TriSNTransportSolver` k and fluxes to round-off
   (step and SCB, P0 and P2). Broken: a corner normal flipped.
2. **Infinite medium and one group:** k∞ exact; self-adjoint flux.
3. **Rotation invariance.** A drum disk with a symmetric arc gives the same k at
   θ and θ + k·Δφ, to solver tolerance. Broken: rim links not updated.
4. **Ripple test** on the HP-MR: k(θ) at every 0.625° over 85–95°, and residual
   from a smooth fit. Target ≤ 2 pcm RMS, against 7.3 pcm for refine 32.
5. **Convergence in the rotor resolution** (rings across B4C, Δφ) and in the
   raster refine outside, separately. This shows which one limits.
6. **Against the refine-32 raster:** agreement in k and pin powers away from the
   arc's edges, to show the conforming mesh changes only the drum treatment.

## Cost and milestones

| # | Milestone | Estimate |
|---|---|---|
| 1 | Drum-hex mesh generator (rotor, stator ring, transition), hybrid HP-MR mesh | 2–4 days |
| 2 | Unstructured S_N engine: corner geometry, cycle-safe sweep order, levels on GPU, DSA via `MeshGrid`, P_L | 5–8 days |
| 3 | CMFD (steady and transient) on mesh faces; rotation by renumbering | 2–3 days |
| 4 | Verification 1–6 | 3–5 days |
| 5 | New references on the 2.5° grid plus window angles (about 80 solves) and re-evaluation of the tables and transients | about 40 GPU-hours |

**Cell count.**
- Each rotor at Δφ = 0.625° with about 30 rings has roughly 35k triangles.
- Twelve drums add about 0.4 M cells and replace their raster cells, so the
  reference grows from 0.34 M to roughly 0.6–0.7 M active cells.
- A solve costs about 2× today's (about 25 min on a GH200).
- Coarser rotor sectors are possible if milestone 4 shows Δφ is not limiting.

## Risks and decision points

- **Sweep cycles on thin polar triangles near the rotor centre.** Mitigation:
  the fan or unstructured core, plus cycle detection with lagging.
- **Transient angles limited to the Δφ grid.** At 90°/s and Δφ = 0.625° the drum
  moves one sector per 7 ms. Case D's step is 5 ms, so θ(t) is quantized;
  halving Δφ doubles the rotor cells.
- **Choice of mesh generator** for the transition region: a new dependency
  (gmsh or triangle) against writing our own.
- **Decision after milestone 4:** whether the ripple is gone. If not, the
  remaining source is the raster outside (refine 32), and the plan extends to a
  conforming mesh of the whole core.
