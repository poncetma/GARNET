# Benchmarks for planar S_N with axial diffusion

Assessed on 28 October 2026 for the
[implementation milestones](sn_2d1d_implementation_plan.md). The selection
criterion is a specified multigroup transport problem with available material
constants: no nuclear-data evaluation, resonance processing, energy collapse,
or new equivalence-factor generation before a solver run.

**Recommendation:** use synthetic dimensional/conservation checks first,
**Takeda Model 4** as the first published triangular/prismatic benchmark, and
the **C5G7 3D Extension** as the later seven-group heterogeneous target.
Keep Takeda-1 and Kobayashi as tests of the limits of axial diffusion.

## Shortlist and data readiness

| Problem | Groups and available data | What it tests | Readiness / work remaining |
|---|---|---|---|
| Manufactured and homogeneous extrusions | Explicit 1/2-group constants; existing C5G7 tables for heterogeneous variants | Balance, dimensional limits, axial interfaces, iteration stability | No external data; construct small independent reference problems |
| **Takeda Model 4, hexagonal-Z FBR** | 4 groups; transport, absorption, production, spectrum, full scattering tables in the public report | Hexagonal geometry, material jumps, partial/full rods, axial-diffusion error | Tables verified; transcribe arrays and build the published layout; not implemented in GARNET |
| **C5G7 3D Extension, Unrodded / Rodded A / Rodded B** | 7 groups; total, scattering, fission, nu-fission, spectrum; all eight materials already checked in | Heterogeneous pins, upscatter, reflector/rod interfaces, local fission rates | No XS work; needs 3D geometry, suitable planar geometry/boundaries, and complete reference-rate arrays |
| Takeda Models 2 and 3 | 4 groups; shared tabulated data, separate from Model 4 | Larger Cartesian core and internal axial blanket | No XS generation; defer geometry work until a Cartesian adapter exists |
| Takeda Model 1 | 2 groups; four material tables | Strong rod/low-density-channel transport effects | Small data set; Cartesian geometry and boundary work; a stress case |
| Kobayashi void/duct problems | 1 group; source and cross sections tabulated | Streaming, ray effects, diffusion-model limitations | Very small data set; optional fixed-source stress tests, not a reactor accuracy gate |

## Takeda Model 4: first published geometry target

The original study defines a small hexagonal-Z fast-reactor model with four
energy groups and rods withdrawn, half-inserted, or fully inserted. Its
published comparisons include eigenvalues, rod worth, region/group fluxes,
and midplane assembly production rates.
[Takeda and Ikeda, original paper][takeda-paper].

The full [NEACRP-L-330 report][takeda-report] is downloadable without a data
library request. Appendix 5 contains the Model-4 spectrum and twelve material
tables on printed pages 95–96 (**PDF pages 102–103**). Appendix 1 describes the
cases. The scanned tables were downloaded and visually checked; they include
the scattering matrices as well as transport, absorption, and production data.

Implementation choices for GARNET:

- Resolve the hexagons with `TriRaster`/`TriGrid`, using the full radial layout
  and its physical outer vacuum boundary. This avoids depending on a new
  triangular reflective-boundary implementation.
- Use the reported transport coefficient as the isotropic solver's collision
  coefficient and `D = 1/(3 Sigma_transport)` for the axial closure, following
  the benchmark's corrected-data convention. Do not apply another correction.
- Check the table's scattering arrows explicitly: source groups label the
  columns in the inspected Appendix-5 layout. GARNET arrays use source groups
  as rows, so transcription must account for that orientation.
- Compare **production rates** using the supplied nu-fission coefficient;
  do not invent a fission cross section or call production heat power.
- Align layer boundaries with the specified material and rod-tip elevations.
  First compare on the same mesh with GARNET's full 3D S_N solver, then make
  the comparison with the published transport results.

This is the best geometry match, not a promise of small axial-diffusion error.
Rods and axial interfaces are useful precisely because they can expose that
approximation. The original paper also discusses transport-method differences,
so use a specified reference and its uncertainty rather than an average over
incompatible calculations. [Original benchmark comparison][takeda-paper].

## C5G7: data already available, geometry still to build

GARNET's [`_c5g7_data.py`](../garnet/benchmarks/_c5g7_data.py) already contains
UO2, three MOX enrichments, fission chamber, guide tube, water, and control rod.
Each has seven-group total, scattering, fission, nu-fission, and spectrum
arrays. The source file records extraction from the
[OpenMOC material-table script][openmoc-xs]. That script packages numerical
benchmark constants into HDF5; it does not generate cross sections.

The official [3D Extension report][c5g7-report], Appendix A, supplies the
geometry and corrected isotropic tables. It defines Unrodded, Rodded A, and
Rodded B. Importantly, even Unrodded includes rods in the upper reflector;
the reflector is not simply a slab of pure water. The prescribed outputs
include the eigenvalue and a 34 x 34 x 3 array of pin/slice fission rates.
The full reference arrays are assigned to the separate CD-ROM appendix D,
not embedded in the downloadable PDF. The report and material tables were
downloaded and inspected; a complete electronic reference array was **not**
located during this review. [Specification and reference appendix][c5g7-report].

What can run without a new XS workflow:

1. **Early derived tests:** extrude a small assembly or colorset using the
   checked-in material arrays and insert axial absorber layers. These are
   reproducible code-verification problems; the published C5G7 eigenvalue is
   not their reference.
2. **Official 3D Extension:** implement its exact configuration and reuse the
   tables. The existing `build_c5g7_2d()` defaults to pin-cell volume
   homogenization; it must not be treated as the official heterogeneous
   transport geometry. Its `pin_resolved=True` option is a useful starting
   point, with explicit radial geometry refinement.
3. **Planar backend:** a Cartesian adapter can reuse `SNTransportSolver` and
   its reflective boundaries while sharing the new axial/coupling machinery.
   Alternatively, a full-domain triangular raster needs a demonstrated
   geometry-convergence study. Equilateral triangles do not exactly represent
   every square core/pin-cell boundary.
4. **Independent comparison:** the published report provides eigenvalue and
   summary comparisons. Retrieve the complete rate arrays before claiming
   their error norms; otherwise generate a documented independent solution
   using the same supplied multigroup constants. OpenMOC maintains
   [a 3D Rodded-B input][openmoc-rodded] that can help cross-check the geometry.
   Running an independent solver is reference calculation, not XS generation,
   and is not required for small regular tests.

Keep **2D C5G7**, the **original 3D problem**, the **2005 3D Extension**, and
**C5G7-TD** identified separately. Their axial layouts and reference values
are not interchangeable. Existing diffusion/ADF reference runs in this
repository are useful regression evidence for their own models, not full
transport references for the new solver.

## Other readily specified cases

**Takeda Models 2 and 3:** Model 2 is a four-group Cartesian fast-reactor
problem with withdrawn and half-inserted rods. Model 3 adds an internal axial
blanket and uses the same cross-section set. They are useful follow-ons for
testing axial material changes without adding a data-generation workflow.
Model 1 is two-group, with empty-channel and inserted-rod cases.
[Original model definitions][takeda-paper]. The report places Model-1 data in
Appendix 3, printed p. 92 (PDF p. 99), and Models-2/3 data in Appendix 4,
printed pp. 93–94 (PDF pp. 100–101). [Cross-section appendices][takeda-report].

**Kobayashi:** the [NEA specification][kobayashi-report] defines one-group
fixed-source problems with near-void regions, including a dog-leg duct.
Table 1 gives the material constants and source strength. Both pure-absorber
and 50%-scattering variants have reference results. These are cheap to
specify but deliberately difficult streaming problems. Use them to quantify
where an axial-diffusion closure loses directional transport physics, not to
require diffusion to reproduce a void-channel transport solution.

**Existing HP-MR:** useful as an application/performance case once verified.
The repository includes eleven-group pin/core NPZ tables, so external library
downloads are unnecessary. However, the default core builder uses placeholder
two-group constants, and assembly fuel preparation can involve homogenization
or SPH choices. The bundled material constructors also use the solver's
`Sigma_t = 1/(3D)` fallback rather than explicit total arrays. Freeze and
document one existing material set before comparison; do not make a new
homogenization workflow part of initial solver qualification.

**Existing diffusion benchmarks:** IAEA-3D, Langenbuch, and similar builders
can check the diffusion limit, but their diffusion references do not validate
full transport physics. Likewise, a benchmark specified through nuclide
compositions and continuous-energy reference calculations is not automatically
ready for a fixed-multigroup comparison. Prefer the tabulated-data problems
above for this development.

## Data handling and reproducibility

Store small benchmark arrays in the repository, with citation and provenance,
so normal tests never download or generate nuclear data. Preserve the
published precision and record any spectrum renormalization or rounding
residual rather than silently adjusting the tables. Check:

- Group ordering, units, scattering orientation, and finite/nonnegative data.
- Consistency of collision, absorption, and outgoing scattering to the
  precision actually printed; distinguish a transpose from table rounding.
- Nu-fission versus fission rates, and the exact reference normalization.
- Material volumes, symmetry/full-domain equivalence, axial elevations, rod
  occupancy, and vacuum versus reflective boundaries.
- Source/report checksum, table location, transcribed-array checksum, solver
  revision, mesh/angles, convergence tolerances, and reference uncertainties.

Verification should first compare against the **same hybrid equations**.
Comparison with converged full 3D S_N then measures the effect of the axial
closure, alongside measured mesh and angle errors. This lets us start useful
tests immediately without fitting new cross sections to conceal model bias.

[takeda-report]: https://www.oecd-nea.org/upload/docs/application/pdf/2020-01/neacrp-l-1990-330.pdf
[takeda-paper]: https://www.jstage.jst.go.jp/article/jnst1964/28/7/28_7_656/_article
[c5g7-report]: https://www.oecd-nea.org/upload/docs/application/pdf/2019-12/nsc-doc2005-16.pdf
[openmoc-xs]: https://github.com/mit-crpg/OpenMOC/blob/develop/sample-input/c5g7-mgxs-hdf5.py
[openmoc-rodded]: https://github.com/mit-crpg/OpenMOC/blob/develop/sample-input/benchmarks/c5g7/c5g7-3d-rodded-B.py
[kobayashi-report]: https://www.oecd-nea.org/upload/docs/application/pdf/2020-01/nsc-doc2000-4.pdf
