# Published benchmarks usable as references for an assembly-level SPH diffusion workflow (SMR / microreactor)

STATUS: partial (survey in progress; will be overwritten as checks complete). Date: 2026-11-12.

Legend: **[CONFIRMED]** = I opened the document/file at the URL given and saw the item (location noted). **[UNCONFIRMED]** = reported by a search snippet, abstract, secondary source or memory; not seen in the primary document. **[RESTRICTED]** = access requires registration/participant status/Data Bank request.

Working copies of everything I opened are in `/tmp/claude-49466/-data-user-poncet-m-GARNET/4e761c73-416a-483b-82be-63f260842d4a/scratchpad/dl/` (PDF/raw files) and `.../scratchpad/txt/` (extracted text, grep-able).

---

## Headline finding

No open publication was found that tabulates **assembly-averaged multigroup fluxes or reaction rates from Monte Carlo together with the homogenized multigroup cross sections** for a hexagonal SMR/microreactor core. The closest items are (i) the OECD/NEA PWR MOX/UO2 transient benchmark (assembly-homogenized 2G/4G XS with ADFs + heterogeneous-transport reference k_eff and assembly powers, Cartesian, no fluxes), (ii) the INL SPH-HTR report INL/EXT-17-41516 (hex prismatic 2-D core: Serpent block-wise one-group fluxes/reaction rates + DRAGON 8-group block fluxes/reaction rates + SPH-corrected INSTANT results), and (iii) the VTB HP-MR/GC-MR/KRUSTY/S8ER packages (open Serpent inputs + Serpent-generated ISOXML multigroup libraries, but **no reference fluxes in the libraries**). Everything else is either a heterogeneous transport benchmark (Tier 2) or a code-to-code SPH comparison (Tier 3).

---

## TIER 1 — assembly/region-level Monte Carlo (or transport) data already published

### T1-A. OECD/NEA + US NRC PWR MOX/UO2 Core Transient Benchmark (NEA/NSC/DOC(2006)20; NEA-1780) — Cartesian PWR, thermal, 2-D (Part 1) and 3-D (Parts 2–4)
- (a) Spec: Kozlowski & Downar, *Final Specifications, Rev. 2*, Dec 2003 — open PDF https://www.oecd-nea.org/upload/docs/application/pdf/2020-10/mox_benchmark_specifications_2003.pdf **[CONFIRMED, downloaded, 18 pp]**. Final report: NEA/NSC/DOC(2006)20, NEA No. 6048 — open PDF https://www.oecd-nea.org/upload/docs/application/pdf/2019-12/nea6048-mox.pdf **[CONFIRMED, downloaded, 71 pp]**. NEA page: https://www.oecd-nea.org/jcms/pl_32218 (returns 403 to scripted fetch).
- (b) 17x17 PWR, assembly pitch 21.42 cm (spec Table, line "Assembly pitch (cm) 21.42") **[CONFIRMED]**; 1/4 core with reflector; MOX (weapons-grade) + UO2 assemblies; Part 1 = 2-D fixed T/H (rho_M=752.06 kg/m3, T=560 K, 1000 ppm); Part 2 = 3-D HFP; Part 3 = 3-D HZP; Part 4 = rod withdrawal transient **[CONFIRMED, spec lines 536–600]**.
- (c) Data supplied: "2G assembly homogenized XS with discontinuity factors and pin power form functions; 4G assembly homogenized XS with discontinuity factors and pin power form functions; 8G pin-cell homogenized XS", NEMTAB-like format, branches in fuel temperature / moderator density / boron, kinetic parameters **[CONFIRMED, spec §"XS homogenization levels", lines 380–396; ADF table format at line 746]**. Group structure Table 8 (2G cut 0.625 eV; 4G; 8G) **[CONFIRMED]**. Reference results (final report): Part I eigenvalue and assembly powers (Table 3, Figs 2–3 "reference assembly power density at ARO/ARI"), rod worths (Tables 4–5), pin-power %PWE/%EWE (Tables 6–9); Part II critical boron + assembly power (Table 10); Part III (Table 13) **[CONFIRMED, final report table list lines 138–153]**. The reference is the **DeCART 47-group heterogeneous transport** solution (HELIOS-based library), with MCNP also used for Part I **[CONFIRMED, final report lines 251, 329–342, 535]**. **No assembly-averaged fluxes are published** (none in table list) **[CONFIRMED absent in table list; UNCONFIRMED whether figures show fluxes]**.
- (d) Access: spec and final report open. The XS library files were distributed via https://engineering.purdue.edu/PARCS/MOX_Benchmark (spec line 479–481) **[CONFIRMED the spec says so; site availability today UNCONFIRMED]** and via NEA-1780, which a search snippet of the NEA abstract page describes as restricted to participants **[UNCONFIRMED — NEA tools page returns 403 to me]**.
- (e) Mapping: single-assembly nodes (21.42 cm), 2-group or 4-group with ADFs → directly a nodal-diffusion input; fits an assembly-level SPH/ADF workflow one-to-one (the ADFs could be converted to SPH factors only if reference fluxes were available — they are not; so this validates the *diffusion core solver with equivalence data* and the assembly-power/k_eff end points, not an SPH-factor computation).
- (f) Effort: low if the library can be obtained (Purdue site or NEA-1780); otherwise one must regenerate XS with a lattice code (moderate).
- Caveat on tier: lacks reference region fluxes, so strictly "Tier 1-partial".

### T1-B. INL/EXT-17-41516, Sen, Hummel, Hiruta, "Super-Homogenization-Corrected Cross-Section Generation for High-Temperature Reactors" (INL, March 2017) — hexagonal prismatic HTR (SPHTR), thermal (graphite), 2-D full core
- (a) https://inldigitallibrary.inl.gov/sites/sti/sti/7323693.pdf **[CONFIRMED, text extracted; header and table list seen]**.
- (b) Hypothetical 2-D prismatic HTR full core ("SPHTR") modelled in Serpent; supercells (7 blocks) in DRAGON5; INSTANT diffusion with SPH **[CONFIRMED, lines 87–109, 334–369]**. Table 1 = SPHTR characteristics; Table 3 = geometry for DRAGON SC calculations; Table 4 = material number densities **[CONFIRMED in table list, lines 167–171]** → the problem appears fully specified in the report (details of Tables 1/3/4 not yet read line by line **[UNCONFIRMED completeness]**).
- (c) Published: Table 2 "Serpent one-group neutron flux and reaction rates for different blocks" (central block, inner fuel, outer fuel, outer CR, graphite reflector; flux, fission, nu-fission, absorption; normalized to 1 fission/s) **[CONFIRMED, values seen lines 339–350]**; Tables 5–7 DRAGON volume-integrated fluxes and homogeneous fission/absorption reaction rates per block (8-group structure, Table 9 lower boundaries) **[CONFIRMED in table list; group-wise content UNCONFIRMED]**; Tables 10–15 INSTANT vs DRAGON/Serpent reaction rates with/without SPH, k comparisons rodded/unrodded **[CONFIRMED in table list]**.
- (d) Open (INL digital library).
- (e) Mapping: block = hex assembly; the Serpent block fluxes are one-group only, the multigroup (8-g) block fluxes are from DRAGON (deterministic); so an SPH check at block level is possible against DRAGON (multigroup) and against Serpent (one-group reaction rates).
- (f) Effort: low-moderate (re-type geometry/compositions from tables; verify completeness).

### T1-C. NEA/NSC/R(2019)6, MHTGR-350 "Results for the Lattice Physics Exercises" — hex prismatic block (infinite lattice), thermal
- (a) https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_3913.pdf **[CONFIRMED, downloaded, 56 pp]**.
- (c) Tables 5–8: homogenized few-group cross-sections for cases 1a/1b (block level) from participants including Serpent v2 (INL) and another Monte Carlo code, plus k_inf (Table 4) and four factors (Table 9); flux per lethargy shown only as figures (Figs 4, 7) **[CONFIRMED table/figure list lines 254–262, 210–213, 401–440]**. Region fluxes **not** tabulated **[CONFIRMED absent from table list]**.
- (e) Lattice-level only (k_inf and block-homogenized XS); useful to check an XS-generation/condensation step for a hex TRISO block, not an SPH core step.

### T1-D (near miss). NEAMS Virtual Test Bed HP-MR ("mrad") data files — hexagonal heat-pipe microreactor, thermal (graphite + YH), 3-D full core
- (a) Repo folder https://github.com/idaholab/virtual_test_bed/tree/main/microreactors/mrad **[CONFIRMED via GitHub API listing]**: `Serpent_Model/serpent_input.i` (17,701 B) **[CONFIRMED, downloaded]**, `Serpent_Model/TRISO_U900_PF40_R100.inp` (133 B = git-LFS pointer) **[CONFIRMED size]**, `isoxml/fullcore_xml_G11_endfb8_ss_tr.xml` (1,474,125 B) **[CONFIRMED, downloaded]**, `isoxml/HP-MR_XS.xml`, `HP-MR_XS_Na_HPMR.xml`, `Na-HPMR_Serpent_ENDFB80_g11_u804.xml` (132–133 B = LFS pointers) **[CONFIRMED sizes]**.
- (c) ISOXML: 11 groups (`NGroup="11"`), 10 libraries (IDs 801, 802, 803, 805, 815, 816, 817, 810, 811, 820 = the Serpent `set gcu` universes: fuel compact, moderator pin, heat-pipe mix, shells, drum Be, drum B4C, monolith zones), tabulated on Tfuel x Tmod = {600,700,800,1000,1200} K; reactions Absorption, NGamma, Scattering(P?), Total, NeutronVelocity, nuFission, kappaFission, FissionSpectrum, DNFraction, DNSpectrum, DNPlambda. **No flux/SPH data of any kind in the file (grep -ci flux = 0)** **[CONFIRMED]**. Serpent input: full 360° core, 30 fuel universes in a 13-ring hex lattice of pitch 26.752 cm (`lat 50 2 0.0 0.0 13 13 26.752`), pin lattice pitch 2.3 cm (`lat 20 ... 19 19 2.3`), core height 160 cm active (pz 20–180), `set nfg 11` with boundaries 8.0e-8, 1.8e-7, 6.25e-7, 1.3e-6, 4.0e-6, 1.4873e-4, 9.118e-3, 0.183, 0.5, 1.353 MeV, `set bc 1` (black), `set pop 100000 500 30`, `set gcu 801 802 803 805 815 816 817 810 811 820`; **no `set sph`/`set adf`/detectors** **[CONFIRMED, lines 611–622 and card listing]**. Materials with densities present (fuel 10.744 g/cc, matrix 1.806, Be 1.848, B4C drum 2.51, YH 4.085, …) **[CONFIRMED]**.
- Conclusion: an open full specification + an open 11-group Serpent-generated library, but no reference fluxes, so SPH factors cannot be recomputed from the repo alone → listed under Tier 2 (spec) and Tier 3 (Griffin results) below.

---

## TIER 2 — complete open specifications to generate one's own transport reference

### T2-1. NEAMS VTB HP-MR (ANL "MRAD"/HP-MR; Stauff et al.) — hex, thermal, 3-D, microreactor (closest to the user's case)
- (a) Repo (above). Documentation: `doc/content/microreactors/mrad/reactor_description.md`, `mrad_model.md`, `mrad_results.md` (raw GitHub) **[CONFIRMED read]**. Paper: Stauff et al., "High-Fidelity Multiphysics Modeling of a Heat Pipe Microreactor Using BlueCrab", NSE 2024, doi:10.1080/00295639.2024.2375175; accepted manuscript open at https://www.osti.gov/servlets/purl/2569789 **[CONFIRMED, downloaded 28 pp]**.
- (b) 2 MWth; 30 hex fuel assemblies + 1 ring Be reflector; 12 control drums; central shutdown rod; TRISO UCO 19.95 at% LEU, 40 % PF, graphite matrix 1.806 g/cc; YH2 moderator pins (SS clad, He gaps); K heat pipes (SS); active length 160 cm, 20 cm Be axial reflectors; Na variant identical except working fluid **[CONFIRMED, reactor_description.md and NSE paper Table 1, lines 150–232]**.
- (c) Published reference results: NSE paper Table 3 — Serpent-2 k_eff 1.03143 (700 K), 1.02853 (800 K), 1.02286 (1000 K), ±7.5 pcm; Griffin DFEM-SN 1.03456 / 1.03204 / 1.02572 (Δ = −313, −351, −286 pcm); Doppler −2.73 (Serpent) vs −2.36 pcm/K (Griffin); Table 4 at 700 K: Serpent k_eff 0.96970 / 1.01243 / 1.03143 for YH1 / YH1.5 / YH2 **[CONFIRMED, lines 379–432]**. XS: one averaged fuel set for all fuel regions; multiple temperature grids and H-content grid **[CONFIRMED, lines 328–337]**. VTB results page: coupled Griffin-BISON-Sockeye k_eff 1.050024 (2.07 MW) and 1.048425 (4.32 MW); **no Serpent comparison, no power maps, no fluxes on that page** **[CONFIRMED]**. The HPMR-H2 (SiMBA, 1/12 core) page: Griffin eigenvalue error 259 pcm vs Serpent; power RMSE 1.66 %/1.41 %/1.75 % per assembly type, total 1.31 %; 11 groups (40 MeV–6.7e-8 MeV), tabulated in T_f, T_m, T_r, c_H; DFEM-SN, no SPH; **no Serpent input provided** **[CONFIRMED, hpmr_h2_model.md / hpmr_h2_results.md]**.
- (d) Open (VTB, LGPL-type VTB licence; TRISO particle file requires `git lfs`).
- (e) Mapping: 30 assemblies of pitch 26.752 cm → 30 hex nodes (or 1/6 core); 11-group library already homogenized per material zone (not per assembly); assembly-averaged fluxes would have to come from the user's own transport solve of this Serpent geometry (ten material zones, double heterogeneity via TRISO).
- (f) Effort: high for S_N (TRISO double heterogeneity must be pre-homogenized; the open ISOXML can serve as the material-zone XS for a *multigroup* S_N run of the 2-D/3-D core with vacuum BC — that gives a consistent multigroup transport reference in hours).

### T2-2. NEAMS VTB GC-MR (gas-cooled microreactor) — hex, thermal, 3-D
- (a) https://github.com/idaholab/virtual_test_bed/tree/main/microreactors/gcmr/core : `Serpent_Model/serpent_input.i` (19,578 B) **[CONFIRMED downloaded]**, 4 particle files (LFS pointers), `ISOXML/XS_Griffin.xml` (249 kB, 11 groups, 14 libraries, Tfuel = 725/825/925 K, **no flux data**) **[CONFIRMED]**, `GCMR_XS_2grid_detailed.xml` (3.4 MB), `GCMR_XS_Xe_TR.xml`; Griffin DFEM-SN steady-state input; doc `GCMR_Core_Neutronics.md`: three assembly types A/B/C, TRISO 19.75 wt% LEU, YH moderator, compact radius 0.85 cm, 40 % PF; 1/6 core; Serpent vs Griffin normalized axial power figure **[CONFIRMED]**; k_eff comparison numbers not in the doc text I grepped **[UNCONFIRMED]**.

### T2-3. OECD/NEA C5G7 MOX benchmark, 2-D/3-D (NEA/NSC/DOC(2003)16) — Cartesian, thermal, 7-group, heterogeneous
- (a) Open PDF https://www.oecd-nea.org/upload/docs/application/pdf/2019-12/nsc-doc2003-16.pdf **[CONFIRMED, downloaded 147 pp]** (cms.oecd-nea.org mirror does not resolve from here).
- (b) 2x2 assemblies (2 UO2, 2 MOX) + water reflector, 17x17 pins, pitch 1.26 cm, assembly 21.42 cm; 2-D: vacuum right/bottom, reflective top/left; 3-D: 21.42 cm axial water reflector, reflected below/vacuum above **[CONFIRMED, lines 267–273, 1447–1452]**.
- (c) 7-group transport-corrected P0 cross sections per material: Appendix A "Benchmark specification" (p. 67) **[CONFIRMED present in the PDF]**. Reference: multigroup MCNP: k_eff 2-D = 1.186550 (±0.008 % 1σ), 3-D unrodded = 1.183810 **[CONFIRMED, lines 627, 679, 892]**; pin powers (Tables 4, 11, 18, 27), assembly-power per-cent errors (Tables 12, 19, 28, Figs 9–11, 21–23) **[CONFIRMED table list]**. Full reference pin-power maps are in Appendices D–I, which are "available on CD-ROM" only **[CONFIRMED line 98]**. No region fluxes.
- (d) Open PDF; CD-ROM appendices not online **[CONFIRMED]**.
- (e) Mapping: 4 assembly nodes + reflector; the 7-group XS make a multigroup Monte Carlo (OpenMC MG mode) or fine S_N run cheap, with no XS processing; assembly-averaged 7-group fluxes are then obtained directly.
- (f) Effort: low (hours).

### T2-4. C5G7 3-D extension, rodded (NEA/NSC/DOC(2005)16)
- (a) Open PDF https://www.oecd-nea.org/upload/docs/application/pdf/2019-12/nsc-doc2005-16.pdf **[CONFIRMED, downloaded 156 pp]**.
- (c) Reference MCNP k_eff: Unrodded 1.14308, Rodded A 1.12806, Rodded B 1.07777 (Tables 3–5) **[CONFIRMED lines 730–757]**; specific pin powers (Tables 6–8), assembly powers (Tables 9–11) and per axial slice (Tables 19, 29–32) **[CONFIRMED table list]**; pin-power maps "see CD-ROM" **[CONFIRMED lines 5328–5337]**.

### T2-5. VVER-1000 MOX Core Computational Benchmark (NEA/NSC/DOC(2005)17, NEA No. 6088; Gomin, Kalugin, Oleynik) — HEXAGONAL, thermal, 2-D full core
- (a) Open PDF https://www.oecd-nea.org/upload/docs/application/pdf/2019-12/nsc-doc2005-17.pdf **[CONFIRMED, downloaded 88 pp; title page seen]**.
- (b) Full-size 2-D VVER-1000 core, 30 % MOX loading, infinite axial dimension, vacuum side BC, reflector included **[CONFIRMED lines 274–300]**; 6 states S1–S6 (Table 2.1).
- (c) Reference MCU (Kurchatov) + MCNP (GRS) + RADAR: k_eff for S1–S6 (Table 2), **assembly-average fission-rate distributions for all 6 states with statistical uncertainties (Tables 3–9)**, pin-by-pin fission rates for assemblies 3, 21, 27 (Tables 11–13), isotopic compositions (Tables A.3–A.9) **[CONFIRMED table list lines 200–228]**. No homogenized XS, no fluxes.
- (e) Mapping: 163 hex assemblies → direct hex-node validation of k_eff and assembly fission rates; one would still need own transport to get assembly MG fluxes for SPH.
- (f) Effort: moderate (2-D pin-resolved hex core; continuous-energy data needed → the user's S_N code would need MG XS from a lattice code).

### T2-6. Neutronics benchmark of a NuScale-like core (Fridman, Bilodid, Valtavirta, NET 55 (2023) 3639, doi:10.1016/j.net.2023.06.029; McSAFER) — Cartesian SMR, thermal, 3-D
- (a) Paper open access **[CONFIRMED via HZDR record Publ-37135]**; dataset CC-BY-4.0, doi:10.14278/rodare.2457, files `01-Material_Compositions_v3.xlsx`, `02-Reference_Solution_v3.xlsx` **[CONFIRMED, downloaded]**.
- (c) Reference Serpent 2.2.0 / ENDF/B-VII.1 full core: sheets `k-eff + CRW` (ARO, RE1 in, RE2 in, SH3 in, SH4 in, ARI, with STD), `rad_pow_ref`/`_std`, `rad_pow_CR`/`_std` (assembly radial powers), `pin_pow_ref`/`_std`, `pin_pow_CR`/`_std`, `pow_ax` **[CONFIRMED sheet names/strings]**. No fluxes, no homogenized XS.
- (e) 37-assembly 17x17 SMR core; most SMR-like of the LWR items.
- (f) Effort: high (3-D continuous-energy spec; needs lattice XS generation or MG library before an S_N run).

### T2-7. OECD/NEA MHTGR-350 benchmark (NEA/NSC/R(2017)4 Vol I–II; Phase I Ex. 1 results NEA/NSC/R(2019)5) — hex prismatic HTGR, thermal, 3-D
- (a) Spec Vol I/II open PDF https://www.oecd-nea.org/upload/docs/application/pdf/2020-01/dir1/nsc-r2017-4.pdf **[CONFIRMED downloaded 110 pp]**; results INL/LTD-17-43099 = NEA/NSC/R(2019)5 https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_3111.pdf **[CONFIRMED downloaded 79 pp]**; NEA page https://oecd-nea.org/jcms/pl_32146 (extranet "working area" for data files) **[CONFIRMED]**.
- (c) Exercise 1 uses a "provided simplified cross-section set with no state parameter dependence (Appendix II)", block-homogenized and condensed to **26 groups**, generated with DRAGON-4 from end-of-equilibrium-cycle number densities **[CONFIRMED lines 1417–1435, 1860–1866]**; electronic files (e.g., `OECD-MHGTR350-power.inp`) are in the NEA working area **[CONFIRMED mention; access RESTRICTED/extranet]**. Results: 7 participants, 9 codes, **mean and SD of participants (Tables 4–5), no Monte Carlo reference** **[CONFIRMED lines 631–634, 823, 1040]**.
- (e) Already block-homogenized 26-group XS → a diffusion/transport code-to-code test (no SPH reference). VTB has `htgr/mhtgr/mhtgr_griffin/{benchmark,data}` folders **[CONFIRMED names; contents not listed — UNCONFIRMED]**.

### T2-8. VERA Core Physics Benchmark Progression Problems (CASL) — Cartesian PWR (Watts Bar 1), thermal
- (a) Results report CASL-U-2017-4148-000 open https://www.osti.gov/servlets/purl/3002109 **[CONFIRMED downloaded 80 pp]**: P1–P5 vs KENO-VI (SCALE 6.2β) k_eff and pin powers (Tables P1-2, P2-1, P3-2, P4-2, P5-2), P6–P7 vs MC21/CTF **[CONFIRMED lines 130–131, table list]**. Spec CASL-U-2012-0131-004: link http://www.casl.gov/sites/default/files/docs/CASL-U-2012-0131-004.pdf now 404 **[CONFIRMED]**; possibly at https://www.osti.gov/servlets/purl/3015560 **[UNCONFIRMED]**. No homogenized XS.

### T2-9. BEAVRS v2.0.2 (MIT) — Cartesian PWR, thermal, 3-D, measured data
- (a) Spec PDF https://bpb-us-e1.wpmucdn.com/sites.mit.edu/dist/9/643/files/2024/07/BEAVRS_2.0.2_spec.pdf **[CONFIRMED downloaded 177 pp; title page/authors seen; body text did not extract cleanly, so contents below are UNCONFIRMED by me]**: measured HZP critical boron, rod worths, ITC, in-core detector flux maps for 58 instrumented assemblies, cycles 1–2 **[UNCONFIRMED — from search snippets/known literature]**. No assembly-homogenized XS **[UNCONFIRMED]**.

### T2-10. KAIST benchmark problems 1A/2A/3A (N. Z. Cho, "Benchmark Problems in Reactor and Particle Transport Physics", KAIST NurapT, 2000) — Cartesian, 7-group
- Host nurapt.kaist.ac.kr does not resolve from here **[CONFIRMED failure]**. KAIST 1A (ARO) reference k_eff 1.116271 from DeCART2D quoted in a KNS paper (18A-336) **[UNCONFIRMED — search snippet]**. Treat as "could not access".

### T2-11. Takeda benchmarks (Takeda & Ikeda, JNST 28(7) 656 (1991); NEACRP-L-330) — Model 1: 2-group small LWR core with rod/void; Model 4: hex-Z FBR, 4-group
- JNST paper (tandfonline) and NEACRP-L-330 could not be opened (403/404) **[CONFIRMED failure]**. Model 1 reference Monte Carlo k_eff = 0.9778 ± 5e-4 reproduced in arXiv:2306.11600 (iQMC) **[CONFIRMED in that secondary paper]**; KNS MC2017 paper P058S01-11 Table III gives Model 1 eigenvalues **[CONFIRMED]**. Region-averaged fluxes in the original **[UNCONFIRMED]**.

### T2-12. HTR-10 initial criticality (IAEA-TECDOC-1382 Part 2) — pebble bed
- https://www-pub.iaea.org/MTCD/Publications/PDF/te_1382_web/TE_1382_Part2.pdf **[CONFIRMED downloaded]**: design data (Tables 4-1, 4-2), homogenized zone atom densities (Table 4-3), VSOP and MCNP k_eff tables (4-4, 4-5) **[CONFIRMED table list]**. Pebble bed, 2-D R-Z; low relevance to hex assemblies.

### T2-13. VTB KRUSTY (fast, HEU-Mo, tiny) and S8ER (SNAP-8, hex UZrH lattice)
- KRUSTY: `microreactors/KRUSTY/Neutronics/Serpent/krusty_serpent_f300_s300_ANL_Simplified_g22.inp` (146 kB, `set gcu` many universes, `set nfg ANL22`, `set bc 1`) **[CONFIRMED]**; ISOXML and `_res.m` are LFS pointers; doc table: ICSBEP MCNP6 1.00042, simplified MCNP6 1.00076, Serpent 1.00081 (ENDF/B-VIII.0) **[CONFIRMED krusty_mc.md]**; Griffin vs Serpent Δk −222.6 / −229.1 pcm (300/800 K) **[CONFIRMED krusty_results.md]**.
- S8ER: 16 Serpent inputs + 16-group ISOXML `core_2D_nodrums_16G_WET_XS.xml` (880 kB) **[CONFIRMED listing]**; 211-element hex lattice **[CONFIRMED s8er_desc.md]**.

---

## TIER 3 — code-to-code SPH / equivalence results

### T3-1. Lee, Jung, Zhong, Ortensi, Labouré, Wang, DeHart, "Assessment of the Griffin Reactor Multiphysics Application Using the Empire Micro Reactor Design Concept", ANL/NSE-20/23 / INL/LTD-20-59263 (2020), OSTI 1833008 (full text redirects to https://publications.anl.gov/anlpubs/2020/08/161410.pdf, which refuses scripted access; text copy in scratchpad) **[CONFIRMED]**
- Empire: 18 hex assemblies (SS monolith, 217 holes: 61 Na heat pipes, 96 moderator pins, 60 fuel pins), Be reflector, 12 drums (EuB 1 cm), central air hole, core 70 cm (60 active + 5+5 reflectors), 625 K / 400 K **[CONFIRMED lines 352–380]**; geometry details referenced to an Empire specification "Ref. [9]" (not reproduced in full) **[CONFIRMED]**.
- XS: Serpent 2, 11/23/15-group structures (Tables 4-1–4-3), per composition (fuel pin per assembly ring A/B/C, moderator, heat pipe, air, monolith, reflector, absorber) **[CONFIRMED]**.
- Results: Table 5-1 assembly k: Serpent 1.23850 (2-D), 1.12543 (3-D); Table 5-2 2-D core: Serpent 1.20100 (CD out), 1.17212 (CD in), drum worth 2052 pcm; Table 5-3 3-D core 1.06757 / 1.03753, worth 2712 pcm **[CONFIRMED]**. SPH-diffusion (PJFNK-SPH, CFEM): Table 6-1 2-D assembly Serpent 1.21378 vs SPH-diff 1.21360 (−14.9 pcm; pin RMS 0.31 % vs direct tallies); Table 6-2 3-D assembly 1.14084 vs 1.14086; Table 6-3 2-D core CD-in Serpent 1.14519 reproduced exactly, pin RMS 0.56 %, max 1.13 %; Table 6-4 CD-out 1.17242 reproduced exactly **[CONFIRMED lines 981–1041]**. SPH applied to fuel pins only (pin-homogenized, 6 axial zones) **[CONFIRMED]**. **SPH factors, reference fluxes and XS are not tabulated** (only flux figures 6-1–6-3) **[CONFIRMED]**.

### T3-2. Kumar, Lee, Labouré, Jung, Terlizzi, Wang, Ortensi, "Transient multiphysics simulations with pin power reconstruction in the Griffin reactor physics code", Ann. Nucl. Energy 230 (2026) 112111 — 2-D SiMBA (Empire-like) hex microreactor, SPH-based equivalence, core RMS 0.6 %, max pin 2.3 % vs Serpent2 **[UNCONFIRMED — abstract only]**.

### T3-3. VTB `drum_rotation` (SiMBA 2-D, Empire-like): Griffin DFEM-SN inputs, 11-group Serpent XS as LFS pointer `empire_core_modified_11G_CD.xml`, parametrized in T_fuel, T_mod, drum angle **[CONFIRMED drum_index.md]**; no Serpent input in folder **[CONFIRMED]**.

### T3-4. Stauff et al. NSE 2024 / VTB HP-MR: Griffin DFEM-SN (no SPH) vs Serpent (Tables 3–4, −286 to −514 pcm) **[CONFIRMED]**; ANL/NEAMS-21/3 (HP-MR, Griffin "diffusion + SPH" on unit cell per search snippet) **[UNCONFIRMED — publications.anl.gov refuses access]**.

### T3-5. Labouré, Wang, Ortensi, Schunert, Gleicher, DeHart, Martineau, "Hybrid Super Homogenization and Discontinuity Factor Method for Continuous Finite Element Diffusion", INL/JOU-18-51060 (ANE 2019) — text obtained **[CONFIRMED exists]**; test problems not yet checked **[UNCONFIRMED]** (see pending).

### T3-6. Takasugi, Labouré, Ortensi, Ivanov, Avramova, Martin, "Preservation of kinetics parameters generated by Monte Carlo calculations in two-step deterministic calculations", EPJ N 2023 / INL/JOU-21-65377 — Serpent→Griffin with SPH on IPEN/MB-01, Godiva, Flattop; sensitivity to energy grid and regional SPH **[CONFIRMED preprint text]**.

### T3-7. Naupa et al., "Verification of the Serpent-Griffin Workflow using the SNAP8 Experimental Reactor", INL/CON-22-69994 (2023): hex 211-element core, 18-group Hansen-Roach structure, S_N (no SPH) **[CONFIRMED]**.

### T3-8. Terlizzi, Vermaak, Charlot, "Refining NEAMS MARVEL Reactor Model Accuracy", PRS/RPT-25-06218 (Sept 2025), OSTI 3019557 — Griffin DFEM-SN vs MCNP k_eff (Table 2: 164–557 pcm), drum-angle curve; MCNP reference from internal ECAR-6099 **[CONFIRMED]**; geometry not fully public **[UNCONFIRMED]**.

### T3-9. DYN3D SPH: Rachamin & Kliem (M&C 2017; ANE 114 (2018)) BFS-73-1 pin-by-pin SPH with Serpent **[CONFIRMED M&C text]**; Däubler (KIT slides) DYN3D 4-group SPH on MOX/UO2 benchmark vs DeCART **[CONFIRMED slides text]**; Zhang, Mercatali, Sanchez-Espinoza, "PARCS pin-wise simulation with a cross section correction system based on the SPH method", NED 445 (2025) 114515 **[CONFIRMED exists; cases UNCONFIRMED]**.

### T3-10. Hébert, "A Reformulation of the Transport-Transport SPH Equivalence Technique" (7ICMSNSE 2015): test TCM32 with XS Table 1 **[CONFIRMED]** — method paper (DRAGON5).

### T3-11. Bostelmann, Strydom, Reitsma, Ivanov — MHTGR-350 Ex. I-1 Serpent vs KENO-VI k_inf (Tables 1–2) **[CONFIRMED text]**; Pandya et al., Shift+Griffin Empire (ANE 2022) **[UNCONFIRMED abstract]**.

---

## RESTRICTED / could not access
- C5G7-TD specification (Boyarinov et al. 2016): NEA working area, participants only; only "Conditions for Release" is public **[CONFIRMED NEA page]**. Hou et al. M&C 2017 paper (open) contains no data tables **[CONFIRMED]**.
- NEA HP-MR benchmark (2026–2029): specifications/results in password-protected working area; conditions-of-release form required **[CONFIRMED NEA page pl_115087]**.
- NEA-1780 (MOX/UO2 XS library) and NEA-1746 (PBMR-400 XS files `OECD-PBMR400-Simplified.XS`): NEA tools pages return 403; snippets say participant/Data-Bank access **[UNCONFIRMED]**.
- MHTGR-350 electronic data files: NEA extranet working area **[CONFIRMED mention]**.
- publications.anl.gov (ANL/NSE-20/23 PDF, ANL/NEAMS-21/3) and tandfonline (Takeda 1991; NSE 2024 publisher version) refuse scripted fetch; OSTI intermittently rate-limits (429).
- nurapt.kaist.ac.kr (KAIST benchmarks): host does not resolve.
- Empire full specification "Ref. [9]" of ANL/NSE-20/23 — not located.
- OECD iLibrary page for VVER-1000 MOX benchmark: 403 (but NEA PDF obtained).

## Pending checks (to be filled)
- Labouré 2019 test problems; C5G7 reference assembly-power values; MHTGR-350 Appendix II (are the 26-group numbers printed in the PDF?); VTB mhtgr_griffin/benchmark folder contents; OpenMC MGXS microreactor data; AER/VVER hex 2-group benchmarks.
