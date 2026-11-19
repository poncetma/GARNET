"""Drum-angle tables and pin powers with the baseline SPH-DF method (HP-MR 2-D).

    python tools/assess_sph_df_tables.py generate ANGLES [NODES] [DEVICE] [CACHE] [TABLES]
    python tools/assess_sph_df_tables.py evaluate [NODES] [DEVICE] [CACHE] [TABLES] [SPACINGS]
    python tools/assess_sph_df_tables.py currents ANGLES [NODES] [DEVICE] [CACHE]
    python tools/assess_sph_df_tables.py attribute [NODES] [DEVICE] [CACHE] [TABLES] [SPACINGS]

The baseline equivalence method (audit report Sec. 10.6): per-hex zones
(``hpmr_assembly_zones(per_hex=True)``), every hex on the vacuum boundary split
into an inner and an outer zone (``split_boundary_layer``), SPH factors on all
zones plus boundary discontinuity factors on the outer zones, flux-volume
normalized (:func:`garnet.sph.sph_bcf_correct`, Laboure et al. 2019), and
per-group pin form functions (Kumar et al. 2026, Eqs. 4-5). Reference: the
pin-resolved refine-16 anisotropic S_N solve (``sn-p2:4:16``) of
tools/hpmr_pin_reference.py at the 25 drum angles 0, 7.5, ..., 180 (all 12
drums together).

generate : for each angle of ANGLES, solve SPH-DF on the node model and store
    the zone constants, SPH factors, boundary factors and form functions in
    TABLES/sphdf_<angle>.npz (also solving the reference's zone-surface
    currents if they are not cached). One GPU job per angle runs in parallel.
evaluate : for every table spacing in SPACINGS (default 15,30,45,90 degrees,
    tables on that sub-grid of the generated angles), interpolate zone
    constants, factors, boundary factors and form functions linearly in angle
    at each 7.5-degree angle off the table, solve the node model and compare
    with the reference solved at that angle: k (pcm), worst assembly power, pin
    power RMS and worst pin (%, powers normalized to mean 1 over the 1890 fuel
    compacts). Baselines at the same angles: no equivalence (the reference's own
    zone constants) and SPH-DF without form functions (f = 1).

attribute : at every off-table angle that also has a generated table,
    interpolate one class of quantity from the table on the spacing and keep
    the rest exact (that angle's own table): zone constants and SPH factors of
    the fuel, reflector or drum zones (drums also constants and factors
    separately), boundary factors, form functions. Locates the interpolation
    error; "exact" must reproduce the reference.
currents : only tally the reference outgoing current of every exposed node
    cell (boundary_cell_currents; needs SPH_DRUM_SECTORS), the input of every
    zoning made of whole node cells.

NODES defaults to raster:8 (1.9 cm cells, the absorber arc resolved).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hpmr_pin_reference import DEFAULT_CACHE, load_or_solve_currents  # noqa: E402
from sph_hpmr_nodes import (absorber_split_zones, boundary_cell_currents,  # noqa: E402
                            node_model, pin_reference, sector_zoning_hooks,
                            shielded_mix, split_boundary_layer, transfer_boundary_layer,
                            zone_boundary_currents)

from garnet import apply_sph_factors, sph_bcf_correct  # noqa: E402
from garnet.benchmarks.hpmr_core_pins import R_FUEL, hpmr_pin_layout  # noqa: E402
from garnet.materials import Material  # noqa: E402
from garnet.pin_power import (footprint_flux, group_form_functions,  # noqa: E402
                              reconstruct_group_form_functions)
from garnet.power import fission_energy_xs  # noqa: E402

REFINE, METHOD = 16, "sn-p2:4:16"
# SPH_TEST_STEPS: the reference angles, as the union of the multiples of each step
# in [0, 180] (default 7.5: the 25-angle set; "5,7.5" adds the 5-degree set)
STEPS = [float(s) for s in os.environ.get("SPH_TEST_STEPS", "7.5").split(",")]
# SPH_TEST_EXTRA: further reference angles, e.g. off the table midpoints
EXTRA = [float(a) for a in os.environ.get("SPH_TEST_EXTRA", "").split(",") if a.strip()]
GRID = np.unique(np.round(np.concatenate(
    [np.arange(0.0, 180.0 + 1e-9, s) for s in STEPS] + [np.array(EXTRA)]), 6))
TOL = 1e-7
# SPH_VERBOSE=1 prints every SPH-DF iteration; SPH_MAXITER caps the coarse solves
VERBOSE = os.environ.get("SPH_VERBOSE", "") not in ("", "0")
# SPH_DRUM_SECTORS=N: each drum one merged zone (Be + arc), its outer boundary
# layer, a core and N fixed ring sectors (sector_zoning_hooks) instead of a
# separate absorber-arc zone; 0 (default) keeps the per-arc zoning
SECTORS = int(os.environ.get("SPH_DRUM_SECTORS", "0"))
# SPH_OUTER_SECTORS=1 (with SPH_DRUM_SECTORS): each drum's outer boundary layer
# in the same sectors, one boundary factor each; reference currents tallied per
# exposed node cell (boundary_cell_currents)
OUTER = SECTORS > 0 and os.environ.get("SPH_OUTER_SECTORS", "") not in ("", "0")
# SPH_SECTOR_ALIGN=1: sector boundaries on each drum's outward line (drum_ring_sectors
# aligned=True), so the arc edges cross them at multiples of 360/N degrees
ALIGNED = SECTORS > 0 and os.environ.get("SPH_SECTOR_ALIGN", "") not in ("", "0")
# SPH_DRUM_OUTER=0 (with SPH_DRUM_SECTORS): no inner/outer split of the drum hexes;
# their ring sectors reach the vacuum boundary and carry the boundary factors.
# With the arc facing out (165-180 deg) a drum outer layer is mostly B4C, its
# zone D is tiny, and no boundary factor can make it leak what the reference
# does (tools/probe_sph_drum_zonings.py); an unsplit sector keeps the Be
# inside the arc and stays feasible.
DRUM_OUTER = not (SECTORS > 0 and os.environ.get("SPH_DRUM_OUTER", "1") in ("0",))
# SPH_OUTER_BE=1 (with SPH_DRUM_SECTORS): each drum's outer layer one zone of Be
# only, the absorber fraction of its cells moved to the ring sector at the same
# azimuth (drum_ring_sectors outer_arc_to_sector): no boundary zone holds B4C and
# SPH-DF is feasible at every drum angle
OUTER_BE = SECTORS > 0 and os.environ.get("SPH_OUTER_BE", "") not in ("", "0")
# per-exposed-cell reference currents whenever the boundary zones are not the
# per-hex outer zones
CELL_CURRENTS = OUTER or not DRUM_OUTER
MAXITER = int(os.environ.get("SPH_MAXITER", "400"))
# SPH_INTERP (evaluate): "factors" (default) interpolates the zone cross sections
# and the SPH factors separately and multiplies them; "corrected" interpolates
# the SPH-corrected cross sections mu * Sigma (and mu * D) themselves. Where a
# zone's Sigma rises as mu falls (the arc entering a drum sector) the product of
# the two interpolants overshoots the interpolated product by -dmu dSigma / 4.
# "shielded" (drum zones holding B4C): rebuild each drum zone's constants at the
# test angle as the flux-volume mix of its non-absorber and absorber parts,
# (S_rest + rho S_B4C) / (1 + rho), rho = V_B4C / V_rest * s, with the part
# volumes exact at the test angle and only S_rest, S_B4C and the flux ratio
# s = phi_B4C / phi_rest interpolated (absorber_split_zones, shielded_mix);
# everything else as "factors". Tested 2026-11-18 and NOT recommended: at 30-deg
# spacing it gave -1378..+481 pcm against -248..-652 for "factors", because the flux
# ratio s itself varies strongly while the arc only partly covers a sector.
INTERP = os.environ.get("SPH_INTERP", "factors")
if INTERP not in ("factors", "corrected", "shielded"):
    raise SystemExit(f"SPH_INTERP must be factors, corrected or shielded, got {INTERP}")

mode = sys.argv[1]
if mode == "currents":
    OUTER = CELL_CURRENTS = True
if mode in ("generate", "currents"):
    angles = [float(a) for a in sys.argv[2].split(",")]
    rest = sys.argv[3:]
elif mode in ("evaluate", "attribute"):
    rest = sys.argv[2:]
else:
    raise SystemExit(__doc__)
spec = rest[0] if len(rest) > 0 and rest[0] else "raster:8"
device = rest[1] if len(rest) > 1 and rest[1] else "auto"
cache = rest[2] if len(rest) > 2 and rest[2] else DEFAULT_CACHE
tables = (rest[3] if len(rest) > 3 and rest[3] else
          os.path.join(cache, "sphdf_tables" + (f"_sec{SECTORS}" if SECTORS else "")
                       + ("o" if OUTER else "") + ("" if DRUM_OUTER else "u")
                       + ("b" if OUTER_BE else "") + ("a" if ALIGNED else "")
                       + ("" if spec == "raster:8" else "_" + spec.replace(":", ""))))
spacings = ([float(s) for s in rest[4].split(",")] if len(rest) > 4 and rest[4]
            else [15.0, 30.0, 45.0, 90.0])
PINS = hpmr_pin_layout()


def setup(angle):
    """Node model and reference at one angle, both with the baseline zones."""
    split = {}

    def node_hook(z, q):
        zs, outer, outer_of = split_boundary_layer(z, q)
        split.update(q=q, outer=outer, outer_of=outer_of)
        return zs

    def ref_hook(z, p):
        return transfer_boundary_layer(z, p, split["q"], split["outer"], split["outer_of"])

    if SECTORS:
        node_hook, ref_hook, split = sector_zoning_hooks(
            SECTORS, outer_sectors=OUTER, aligned=ALIGNED, drum_outer=DRUM_OUTER,
            outer_arc_to_sector=OUTER_BE)
    node = node_model(spec, angle, device, PINS.xy[PINS.fuel], R_FUEL, per_hex=True,
                      zones_hook=node_hook)
    ref = pin_reference(REFINE, METHOD, angle, False, device, cache, per_hex=True,
                        zones_hook=ref_hook)
    z = ref["zones"]
    if tuple(z.names) != tuple(node["zones"].names):
        raise RuntimeError("node and reference zonings differ")
    if INTERP == "shielded" and mode == "evaluate":
        box = {}

        def split_hook(zz, pp):
            zs, box["of"] = absorber_split_zones(ref_hook(zz, pp))
            return zs

        sp = pin_reference(REFINE, METHOD, angle, False, device, cache, per_hex=True,
                           zones_hook=split_hook)
        ref["split"] = dict(hm=sp["hm"], rflux=sp["rflux"], rvol=sp["rvol"], of=box["of"])
    zone_of_pin = int(np.flatnonzero(z.fuel)[0]) + PINS.assembly[PINS.fuel]
    return node, ref, split, zone_of_pin


def reference_boundary(angle, node, split):
    """(boundary zones, reference J_out (B, G)) for the node zoning."""
    if CELL_CURRENTS:
        spec_tag = spec.replace(":", "")
        cells, J = boundary_cell_currents(REFINE, angle, METHOD, device, cache, split["q"],
                                          split["p"].raster, tag=f"bcell-{spec_tag}")
        return zone_boundary_currents(node["zones"], cells, J)
    cur = load_or_solve_currents(REFINE, angle, METHOD, device, cache, per_hex=True)
    parents = sorted(cur["boundaries"])
    return ([split["outer_of"][b] for b in parents],
            np.array([cur["boundaries"][b][:, 0] for b in parents]))


def pin_powers(materials, flux, node, zone_of_pin, form):
    table, _ = fission_energy_xs(materials)
    phi_p = footprint_flux(node["W"], flux)
    if form is None:
        return (table[zone_of_pin] * phi_p).sum(axis=1)
    return reconstruct_group_form_functions(form, zone_of_pin, table, phi_p)


def material_arrays(mats):
    G = mats[0].n_groups
    kf = np.array([m.kappa_fission if m.kappa_fission is not None else np.zeros(G)
                   for m in mats])
    return dict(D=np.array([m.diffusion for m in mats]), sa=np.array([m.sigma_a for m in mats]),
                nsf=np.array([m.nu_sigma_f for m in mats]),
                ss=np.array([m.sigma_s for m in mats]), st=np.array([m.sigma_t for m in mats]),
                chi=np.array([m.chi for m in mats]), kf=kf,
                has_kf=np.array([m.kappa_fission is not None for m in mats]))


def materials_from(a):
    return [Material(name=f"zone-{i}", diffusion=a["D"][i], sigma_a=a["sa"][i],
                     nu_sigma_f=a["nsf"][i], sigma_s=a["ss"][i], chi=a["chi"][i],
                     total=a["st"][i],
                     kappa_fission=a["kf"][i] if a["has_kf"][i] else None)
            for i in range(len(a["D"]))]


if mode == "currents":
    for angle in angles:
        t0 = time.perf_counter()
        node, ref, split, _ = setup(angle)
        bregs, Jref = reference_boundary(angle, node, split)
        print(f"CURRENTS angle {angle:g}: {len(bregs)} boundary zones, total leakage per group "
              + " ".join(f"{v:.4g}" for v in Jref.sum(axis=0))
              + f" [{time.perf_counter() - t0:.0f} s]", flush=True)
    raise SystemExit(0)

if mode == "generate":
    os.makedirs(tables, exist_ok=True)
    for angle in angles:
        t0 = time.perf_counter()
        node, ref, split, zone_of_pin = setup(angle)
        bregs, Jref = reference_boundary(angle, node, split)

        def bcf_solve(mats, gam):
            return node["solve_bcf"](mats, gam, bregs)

        out = sph_bcf_correct(ref["hm"], ref["rflux"], Jref, bcf_solve,
                              region_volume=ref["rvol"], tol=TOL, max_iter=MAXITER,
                              verbose=VERBOSE)
        flux, k, _ = node["solve_bcf_full"](out.corrected_materials, out.bcf, bregs)
        table, _ = fission_energy_xs(out.corrected_materials)
        phi_p = footprint_flux(node["W"], flux)
        form = group_form_functions(ref["rate"], zone_of_pin, table, phi_p)
        back = reconstruct_group_form_functions(form, zone_of_pin, table, phi_p)
        exact = float(np.abs(back / ref["rate"].sum(axis=1) - 1).max())
        names = list(ref["zones"].names)
        np.savez(os.path.join(tables, f"sphdf_{angle:g}.npz"), angle=angle, k_ref=ref["k"],
                 k=k, mu=out.factors, gamma=out.bcf, bregs=np.array(bregs), form=form,
                 converged=out.converged, residual=out.residual, iterations=out.iterations,
                 names=np.array(names), **material_arrays(ref["hm"]))
        fuel = ref["zones"].fuel
        drum = np.array([n.startswith("drum") for n in names])
        dmu = np.where(drum, out.factors[:, -1], np.nan)
        lo, hi = int(np.nanargmin(dmu)), int(np.nanargmax(dmu))
        print(f"GENERATED angle {angle:g}: converged {out.converged} ({out.iterations} solves, "
              f"residual {out.residual:.1e}), dk {1e5 * (k - ref['k']):+.2f} pcm, "
              f"fuel thermal mu {out.factors[fuel, -1].min():.3f}-{out.factors[fuel, -1].max():.3f}, "
              f"drum thermal mu {dmu[lo]:.3g} ({names[lo]})-{dmu[hi]:.3g} ({names[hi]}), "
              f"gamma {out.bcf.min():.3g}-{out.bcf.max():.3g}, form-function construction "
              f"{exact:.1e} [{time.perf_counter() - t0:.0f} s]", flush=True)
    raise SystemExit(0)

# ---------------------------------------------------------------- evaluate
entries = {}
for a in GRID:
    path = os.path.join(tables, f"sphdf_{a:g}.npz")
    if os.path.exists(path):
        d = np.load(path)
        entries[float(a)] = {k: d[k] for k in d.files}
print(f"SPH-DF tables in drum angle, HP-MR 2-D: {METHOD} reference at refine {REFINE}, "
      f"node model {spec}, per-hex zones with inner/outer boundary split"
      f"{f', drums as core + {SECTORS} ring sectors' if SECTORS else ', separate absorber-arc zones'}"
      f"{' (outer layer in sectors too)' if OUTER else ''}"
      f"{' (sector boundaries on the outward line)' if ALIGNED else ''}"
      f"{'' if DRUM_OUTER else ' (drum hexes not split, sectors reach the boundary)'}"
      f"{' (drum outer layers Be only, arc to the ring sectors)' if OUTER_BE else ''}"
      f"; interpolating {'SPH-corrected cross sections' if INTERP == 'corrected' else 'cross sections and factors separately'}, "
      f"device {device}; generated at {sorted(entries)}\n", flush=True)
bad = [a for a, e in entries.items() if not bool(e["converged"])]
if bad:
    print(f"WARNING: unconverged generation at {bad}\n", flush=True)


def corrected_entry(e):
    """Table entry with its zone cross sections SPH-corrected and mu = 1."""
    out = dict(e)
    out.update(material_arrays(apply_sph_factors(materials_from(e), e["mu"])))
    out["mu"] = np.ones_like(e["mu"])
    return out


def interpolated(a, grid):
    """(materials, mu, gamma, form) at angle a from the table on ``grid``."""
    i = int(np.searchsorted(grid, a))
    lo, hi = entries[float(grid[i - 1])], entries[float(grid[i])]
    if INTERP == "corrected":
        lo, hi = corrected_entry(lo), corrected_entry(hi)
    w = (a - grid[i - 1]) / (grid[i] - grid[i - 1])
    mix = {k: (1 - w) * lo[k] + w * hi[k] for k in ("D", "sa", "nsf", "ss", "st", "chi", "kf",
                                                    "mu", "gamma", "form")}
    mix["has_kf"] = lo["has_kf"]
    if INTERP == "shielded":
        shield_drums(mix, a, float(grid[i - 1]), float(grid[i]), w)
    return materials_from(mix), mix["mu"], mix["gamma"], mix["form"]


def _parts(split, z):
    """(rest arrays, absorber arrays or None, V_rest, V_B4C, s or None) of drum zone z."""
    def arrays(m):
        return dict(sa=np.asarray(m.sigma_a), st=np.asarray(m.sigma_t),
                    tr=1.0 / (3.0 * np.asarray(m.diffusion)), ss=np.asarray(m.sigma_s),
                    nsf=np.asarray(m.nu_sigma_f))
    rest = arrays(split["hm"][z])
    zb = split["of"].get(z)
    if zb is None or split["rvol"][zb] <= 0:
        return rest, None, split["rvol"][z], 0.0, None
    s = split["rflux"][zb] / np.maximum(split["rflux"][z], 1e-300)
    return rest, arrays(split["hm"][zb]), split["rvol"][z], split["rvol"][zb], s


def shield_drums(mix, a, a_lo, a_hi, w):
    """Overwrite the drum zones' constants in ``mix`` (see SPH_INTERP=shielded)."""
    s_lo, s_hi, s_at = (case(x)[1]["split"] for x in (a_lo, a_hi, a))
    for z in sorted(set(s_lo["of"]) | set(s_hi["of"]) | set(s_at["of"])):
        r_lo, b_lo, _, _, f_lo = _parts(s_lo, z)
        r_hi, b_hi, _, _, f_hi = _parts(s_hi, z)
        _, _, v_rest, v_abs, _ = _parts(s_at, z)
        rest = {k: (1 - w) * r_lo[k] + w * r_hi[k] for k in r_lo}
        if v_abs > 0 and (b_lo is not None or b_hi is not None):
            if b_lo is not None and b_hi is not None:
                absb = {k: (1 - w) * b_lo[k] + w * b_hi[k] for k in b_lo}
                s = (1 - w) * f_lo + w * f_hi
            else:
                absb, s = (b_lo, f_lo) if b_lo is not None else (b_hi, f_hi)
            out = {k: shielded_mix(rest[k], absb[k], v_rest, v_abs, s) for k in rest}
        elif v_abs > 0:
            continue                      # no absorber at either table point: keep linear
        else:
            out = rest
        mix["sa"][z], mix["st"][z], mix["ss"][z] = out["sa"], out["st"], out["ss"]
        mix["nsf"][z], mix["D"][z] = out["nsf"], 1.0 / (3.0 * out["tr"])


t0 = time.perf_counter()
_cases = {}


def case(a):
    """(node, reference, zone_of_pin) at angle a, built once."""
    a = float(a)
    if a not in _cases:
        node, ref, _, zone_of_pin = setup(a)
        _cases[a] = (node, ref, zone_of_pin)
    return _cases[a]


def score(node, ref, zone_of_pin, materials, gam, form, bregs):
    """(dk pcm, worst assembly %, pin RMS %, worst pin %) of one node solve."""
    flux, k, _ = node["solve_bcf_full"](materials, gam, bregs)
    pin = pin_powers(materials, flux, node, zone_of_pin, form)
    ref_pin = ref["rate"].sum(axis=1)
    err = (pin / pin.mean()) / (ref_pin / ref_pin.mean()) - 1.0
    asm = np.bincount(zone_of_pin, weights=pin)
    asm_ref = np.bincount(zone_of_pin, weights=ref_pin)
    keep = asm_ref > 0
    a_err = (asm[keep] / asm[keep].mean()) / (asm_ref[keep] / asm_ref[keep].mean()) - 1.0
    return (1e5 * (k - ref["k"]), 100 * np.abs(a_err).max(),
            100 * np.sqrt((err ** 2).mean()), 100 * np.abs(err).max())


def table_and_tests(spacing):
    """(table angles present, off-table test angles bracketed by them)."""
    full = GRID[np.isclose(np.mod(GRID + 1e-9, spacing), 0.0, atol=1e-6)]
    grid = np.array([g for g in full if float(g) in entries])
    if grid.size < 2:
        return grid, []
    tests = [float(a) for a in GRID if not np.any(np.isclose(full, a))
             and grid[0] < a < grid[-1]
             and np.searchsorted(full, a) == np.searchsorted(full, grid[np.searchsorted(grid, a)])
             and np.searchsorted(full, a) - 1 == np.searchsorted(full, grid[np.searchsorted(grid, a) - 1])]
    missing = [float(g) for g in full if float(g) not in entries]
    if missing:
        print(f"spacing {spacing:g}: tables missing at {missing}; angles next to them skipped",
              flush=True)
    return grid, tests


ZONE_KEYS = ("D", "sa", "nsf", "ss", "st", "chi", "kf", "mu")


def raw_interpolation(a, grid):
    """Table arrays linearly interpolated to angle a (no SPH applied)."""
    i = int(np.searchsorted(grid, a))
    lo, hi = entries[float(grid[i - 1])], entries[float(grid[i])]
    w = (a - grid[i - 1]) / (grid[i] - grid[i - 1])
    return {k: (1 - w) * lo[k] + w * hi[k] for k in ZONE_KEYS + ("gamma", "form")}


if mode == "attribute":
    names = None
    for e in entries.values():
        if "names" in e:
            names = [str(n) for n in e["names"]]
            break
    if names is None:
        raise SystemExit("attribute needs tables that store their zone names")
    cls = np.array(["fuel" if n.startswith("fuel") else "drum" if n.startswith("drum")
                    else "reflector" for n in names])
    # variant: (zone mask for constants, zone mask for mu, gamma?, form?)
    none = np.zeros(len(names), bool)
    allz = np.ones(len(names), bool)
    VARIANTS = {
        "exact": (none, none, False, False),
        "all": (allz, allz, True, True),
        "fuel zones": (cls == "fuel", cls == "fuel", False, False),
        "reflector zones": (cls == "reflector", cls == "reflector", False, False),
        "drum zones": (cls == "drum", cls == "drum", False, False),
        "drum constants": (cls == "drum", none, False, False),
        "drum mu": (none, cls == "drum", False, False),
        "boundary factors": (none, none, True, False),
        "form functions": (none, none, False, True),
    }
    print("Attribution: one class interpolated from the table on the spacing, the rest exact "
          "(the generated table at the test angle); with form functions.\n"
          "Columns: dk pcm | pin RMS % | worst pin %\n", flush=True)
    for spacing in spacings:
        grid, tests = table_and_tests(spacing)
        tests = [a for a in tests if a in entries]
        if not tests:
            print(f"spacing {spacing:g}: no test angle with its own table, skipped\n", flush=True)
            continue
        print(f"Table spacing {spacing:g} deg, {len(tests)} test angles with their own tables",
              flush=True)
        print(f"{'angle':>6} " + " ".join(f"{v:>21}" for v in VARIANTS), flush=True)
        res = {v: [] for v in VARIANTS}
        for a in tests:
            node, ref, zone_of_pin = case(a)
            ex, ip = entries[a], raw_interpolation(a, grid)
            bregs = list(ex["bregs"])
            row = []
            for v, (zc, zm, g_i, f_i) in VARIANTS.items():
                arr = {k: np.where(zc.reshape((-1,) + (1,) * (ex[k].ndim - 1)), ip[k], ex[k])
                       for k in ZONE_KEYS if k != "mu"}
                arr["has_kf"] = ex["has_kf"]
                mu = np.where(zm[:, None], ip["mu"], ex["mu"])
                mats = apply_sph_factors(materials_from(arr), mu)
                gam = ip["gamma"] if g_i else ex["gamma"]
                form = ip["form"] if f_i else ex["form"]
                r = score(node, ref, zone_of_pin, mats, gam, form, bregs)
                res[v].append(r)
                row.append(f"{r[0]:+7.1f} {r[2]:5.2f} {r[3]:6.2f}")
            print(f"{a:6.1f} " + " ".join(f"{c:>21}" for c in row), flush=True)
        print(f"{'mean':>6} " + " ".join(
            f"{np.mean([r[0] for r in res[v]]):+7.1f} {np.mean([r[2] for r in res[v]]):5.2f} "
            f"{np.mean([r[3] for r in res[v]]):6.2f}".rjust(21) for v in VARIANTS), flush=True)
        print(f"{'worst':>6} " + " ".join(
            f"{max((r[0] for r in res[v]), key=abs):+7.1f} {max(r[2] for r in res[v]):5.2f} "
            f"{max(r[3] for r in res[v]):6.2f}".rjust(21) for v in VARIANTS), flush=True)
        parts = sum(np.array([r[0] for r in res[v]])
                    for v in ("fuel zones", "reflector zones", "drum zones", "boundary factors"))
        print(f"  sum of the fuel, reflector, drum and boundary-factor dk against 'all': "
              f"mean {np.mean(parts):+.1f} vs {np.mean([r[0] for r in res['all']]):+.1f} pcm\n",
              flush=True)
    print(f"total {time.perf_counter() - t0:.0f} s")
    raise SystemExit(0)

bregs = None
summary = []
for spacing in spacings:
    grid, tests = table_and_tests(spacing)
    if grid.size < 2:
        print(f"spacing {spacing:g}: fewer than two table angles, skipped", flush=True)
        continue
    print(f"Table spacing {spacing:g} deg (table at {list(grid)}); "
          f"dk pcm | worst assembly % | pin RMS % | worst pin %", flush=True)
    print(f"{'angle':>6} {'no equivalence':>30} {'SPH-DF, f = 1':>30} {'SPH-DF + form functions':>30}",
          flush=True)
    worst = np.zeros((3, 4))
    for a in tests:
        node, ref, zone_of_pin = case(a)
        mats, mu, gam, form = interpolated(a, grid)
        bregs = list(entries[float(grid[0])]["bregs"])
        rows = [score(node, ref, zone_of_pin, materials, g, f, bregs)
                for materials, g, f in ((ref["hm"], np.ones_like(gam), None),
                                        (apply_sph_factors(mats, mu), gam, None),
                                        (apply_sph_factors(mats, mu), gam, form))]
        worst = np.maximum(worst, np.abs(np.array(rows)))
        print(f"{a:6.1f} " + " ".join(f"{r[0]:+8.1f} {r[1]:5.2f} {r[2]:5.2f} {r[3]:6.2f}".rjust(30)
                                      for r in rows), flush=True)
    summary.append((spacing, len(tests), worst))
    print(flush=True)

print("Worst over the off-table angles (|dk| pcm, worst assembly %, pin RMS %, worst pin %):")
print(f"{'spacing':>8} {'tests':>5} {'no equivalence':>30} {'SPH-DF, f = 1':>30} "
      f"{'SPH-DF + form functions':>30}")
for spacing, n, w in summary:
    print(f"{spacing:7.1f}d {n:5d} " + " ".join(
        f"{r[0]:8.1f} {r[1]:5.2f} {r[2]:5.2f} {r[3]:6.2f}".rjust(30) for r in w), flush=True)
print(f"\ntotal {time.perf_counter() - t0:.0f} s")
