"""HP-MR drum-rotation transients with the tabulated SPH-DF model (2-D, all 12
drums together), and the model's own quasi-static (adiabatic) solution.

    python tools/hpmr_drum_transient.py run CASE [DEVICE] [VARIANT] [DT]
    python tools/hpmr_drum_transient.py report [CASES]
    python tools/hpmr_drum_transient.py compare [CASES]
    python tools/hpmr_drum_transient.py compare_full [CASE]

Plan: docs/sph_drum_transient_plan.md (cases A-E, steps 5 and 6).

run : the full space-time diffusion transient of the SPH-DF tables (refine-32
    S_N reference, drum-band mesh local:4:1, ``SphDfTable``), monolithic steps,
    speeds v/mu(theta0). Every step records the solver power and the pin powers
    (form functions at theta(t)). With VARIANT ``base`` it also solves the static
    SPH-DF model at every angle the path visits (k, pin powers, the adjoint-
    weighted shape norm) and marches the adiabatic point kinetics with
    rho = 1 - k0/k(theta(t)) and beta_eff, Lambda projected from the model's own
    forward and adjoint at theta0 (``project_effective_kinetics``). That is the
    construction of the quasi-static S_N reference applied to the SPH model, so
    full minus adiabatic is the quasi-static error (step 5).
    Attribution variants (no static part): ``v`` the library speeds v instead of
    v/mu, ``gamma0`` the boundary factors frozen at theta0.
    DT overrides the case's step (time-step convergence of the difference).
compare : step 6, the SPH-DF transients against the quasi-static S_N
    reference (``SnReference``): static reactivity along the path, P(t) of the
    full and adiabatic SPH-DF solutions, pin and assembly shapes against the
    S_N pins interpolated at theta(t), and the attribution variants.
report : step-5 comparison of the stored runs: power (max relative difference,
    energy), shape (pin RMS, worst pin, worst assembly of the full transient
    against the static shape at theta(t), over time).

Results: <CACHE>/drum_transients/<case>_<variant>[_dt<dt>].npz (DRUM_OUT overrides
the directory). SPH_TABLE_STEP (default 5) picks the table spacing.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from garnet import (MeshDiffusionEigenSolver, MeshGrid, MeshGroupOperator,  # noqa: E402
                    SphDfTable, TransientSolver)
from garnet.backend import asnumpy  # noqa: E402
from garnet.benchmarks.hpmr import hpmr_sphdf_geometry  # noqa: E402
from garnet.benchmarks.hpmr_core_pins import R_FUEL, hpmr_pin_layout  # noqa: E402
from garnet.benchmarks.hpmr_thermal import hpmr_kinetics_11g  # noqa: E402
from garnet.pin_power import (footprint_flux, mesh_pin_footprints,  # noqa: E402
                              reconstruct_group_form_functions)
from garnet.power import fission_energy_xs  # noqa: E402
from garnet.quasistatic import (_shape_time_importance, adiabatic_point_kinetics,  # noqa: E402
                                integrate_point_kinetics, project_effective_kinetics)

CACHE = os.environ.get("GARNET_REF_CACHE", "/data/scratch/shared/poncet_m/garnet-sph-ref-net")
TABLES = os.environ.get("SPH_TABLES", "sphdf_tables_sec8ba_local41polar_r32")
# SPH_TABLE_STEP: use the table entries at the multiples of this spacing only
# (the directory holds the 5-degree set and, since 2026-11-20, the 2.5-degree one)
TABLE_STEP = float(os.environ.get("SPH_TABLE_STEP", "5"))
OUT = os.environ.get("DRUM_OUT", os.path.join(CACHE, "drum_transients"))
STEADY = dict(tol_k=1e-10, tol_source=1e-9)
TOL_STEP = 1e-7


def ramp(a0, a1, t1):
    return lambda t: a0 + (a1 - a0) * min(max(t, 0.0) / t1, 1.0)


# theta(t) in degrees, end time and step (s); docs/sph_drum_transient_plan.md
CASES = {
    "A": (ramp(90.0, 92.5, 2.0), 20.0, 0.02),            # slow ramp out, hold
    "B": (ramp(90.0, 92.5, 0.2), 2.0, 0.005),            # fast ramp out
    "C": (lambda t: 90.0 + 2.5 * np.sin(np.pi * t), 6.0, 0.01),   # +-2.5 deg, 2 s period
    "D": (ramp(180.0, 0.0, 2.0), 4.0, 0.005),            # insertion at 90 deg/s
    "E": (ramp(40.0, 50.0, 1.0), 5.0, 0.01),             # across the 45-deg sector edge
    # E inserts +624 pcm (S_N), about $0.9: near prompt critical, power 1e5 at 5 s.
    # E2 crosses the same edge with a $0.2 move, like A and B.
    "E2": (ramp(43.75, 46.25, 1.0), 5.0, 0.01),
    "H": (lambda t: 180.0, 1.0, 0.005),                  # hold: solver checks only
}


def key(a):
    return round(float(a), 9)


class Model:
    """The tabulated SPH-DF model, its mesh and the pin reconstruction."""

    def __init__(self, device):
        self.device = device
        self.geo = hpmr_sphdf_geometry(refine=4, levels=1)
        directory = os.path.join(CACHE, TABLES)
        found = [float(f[6:-4]) for f in os.listdir(directory)
                 if f.startswith("sphdf_") and f.endswith(".npz")]
        keep = [a for a in found if abs(a / TABLE_STEP - round(a / TABLE_STEP)) < 1e-9]
        self.tab = SphDfTable.load(directory, self.geo, angles=keep)
        q = self.geo.problem(90.0)
        self.grid = MeshGrid(q.mesh, q.alpha_boundary)
        pins = hpmr_pin_layout()
        self.W = mesh_pin_footprints(q.mesh, pins.xy[pins.fuel], R_FUEL)
        self.area = np.asarray(q.mesh.area, dtype=float)
        zones = self.geo(90.0)[0]
        self.zone_of_pin = int(np.flatnonzero(zones.fuel)[0]) + pins.assembly[pins.fuel]
        self.base_kinetics = hpmr_kinetics_11g()

    def pins(self, state, flux):
        table, _ = fission_energy_xs(state.materials)
        return reconstruct_group_form_functions(state.form, self.zone_of_pin, table,
                                                footprint_flux(self.W, flux))

    def production(self, state, flux):
        """sum_g nuSigma_f phi over the cells (area weighted): the solver's
        power integral up to the constant 1/k0."""
        nsf = np.array([m.nu_sigma_f for m in state.materials])[state.region_map]   # (cells, G)
        return float((nsf.T * flux * self.area).sum())

    def eigen_solver(self, state):
        return MeshDiffusionEigenSolver(self.grid, state.materials, state.region_map,
                                        mix_material=state.mix_region_map,
                                        mix_weight=state.mix_weight, bcf=state.bcf,
                                        device=self.device, linear_solver="gmres")


def run(case, device="gpu", variant="base", dt=None):
    theta, t_end, dt0 = CASES[case]
    t_end = float(os.environ.get("DRUM_T_END", t_end))      # smoke runs
    dt = dt0 if dt is None else float(dt)
    m = Model(device)
    a0 = theta(0.0)
    kin = m.base_kinetics if variant == "v" else m.tab.kinetics(m.base_kinetics, a0)
    problem_at = m.tab.problem_at(theta)
    if variant == "gamma0":
        bcf0 = m.tab.state(a0).bcf

        def problem_at(t, _p=problem_at):           # noqa: F811
            return tuple(_p(t)[:4]) + (bcf0,)
    elif variant not in ("base", "v"):
        raise SystemExit(f"unknown variant {variant!r}")
    print(f"case {case} ({variant}): theta {a0:g} -> {theta(t_end):g} deg, t_end {t_end:g} s, "
          f"dt {dt:g} s, tables {TABLES} every {TABLE_STEP:g} deg "
          f"({len(m.tab.entries)} entries), device {device}", flush=True)

    # Pin powers at each step; their level is carried by the solver's power
    # (the flux scale differs between the steady solve and the steps), so each
    # step stores pin sum / production and P_pin = power * that, relative to t = 0.
    rec = dict(t=[], ratio=[], pins=[], power=[])

    def on_step(t, flux, power):
        st = m.tab.state(theta(t))
        f = asnumpy(flux)
        p = m.pins(st, f)
        rec["t"].append(t)
        rec["ratio"].append(p.sum() / m.production(st, f))
        rec["power"].append(power)
        rec["pins"].append(p.astype(np.float32))

    t0 = time.perf_counter()
    s = TransientSolver(m.grid, problem_at, kin, group_operator=MeshGroupOperator,
                        eig_solver=MeshDiffusionEigenSolver, device=device,
                        linear_solver="gmres", on_step=on_step)
    r = s.solve(t_end, dt, tol_step=TOL_STEP, step_solver="monolithic", steady_kwargs=STEADY)
    wall = time.perf_counter() - t0
    st0 = m.tab.state(a0)
    pins0 = m.pins(st0, r.steady.flux_numpy)
    if not rec["t"] or rec["t"][0] > 0.0:
        rec["t"].insert(0, 0.0)
        rec["ratio"].insert(0, pins0.sum() / m.production(st0, r.steady.flux_numpy))
        rec["power"].insert(0, 1.0)
        rec["pins"].insert(0, pins0.astype(np.float32))
    times = np.asarray(r.times)
    if len(rec["t"]) != len(times) or not np.allclose(rec["t"], times):
        raise RuntimeError("on_step times differ from the result times")
    ratio = np.asarray(rec["ratio"])
    out = dict(case=case, variant=variant, dt=dt, t_end=t_end, times=times,
               table_step=TABLE_STEP,
               theta=np.array([theta(t) for t in times]), power=np.asarray(r.power),
               P_pin=np.asarray(r.power) * ratio / ratio[0], pins=np.asarray(rec["pins"]),
               rec_t=np.asarray(rec["t"]), k0=r.k0, wall=wall,
               step_iterations=np.asarray(r.step_iterations))
    print(f"full transient: {len(times) - 1} steps, {wall:.0f} s, P(end) {r.power[-1]:.6f}, "
          f"FGMRES applications per step {np.mean(r.step_iterations):.1f}", flush=True)

    if variant == "base":
        out.update(static(m, theta, times, kin, r.k0))
    os.makedirs(OUT, exist_ok=True)
    tag = f"{case}_{variant}" + ("" if dt == dt0 else f"_dt{dt:g}")
    np.savez(os.path.join(OUT, tag + ".npz"), **out)
    if variant == "base":
        summarize(out)


def static(m, theta, times, kin, k0_transient):
    """Static SPH-DF solves along the path and the adiabatic point kinetics.

    beta_eff and Lambda are projected at every angle with the FIXED weight
    phi0* (adjoint at theta0), the weight that also normalizes the amplitude
    and the shape norm; rho = 1 - k0/k(theta) is then exactly the projected
    reactivity. Freezing them at theta0 instead (``adiabatic_power_frozen``)
    costs 0.09-0.56 % in power on cases A-C, almost all through Lambda (-0.56 %
    at 87.5 deg), against 0.01-0.06 % with the moving parameters."""
    t0 = time.perf_counter()
    a0 = theta(0.0)
    st0 = m.tab.state(a0)
    s0 = m.eigen_solver(st0)
    fwd = s0.solve(**STEADY)
    adj = s0.solve(adjoint=True, **STEADY)
    par = project_effective_kinetics(s0, fwd, adj, kin, k0=fwd.k_eff)
    print(f"theta0 {a0:g}: k {fwd.k_eff:.7f} (transient k0 {k0_transient:.7f}), adjoint k "
          f"{adj.k_eff:.7f}, beta_eff {par.beta_total * 1e5:.1f} pcm, Lambda "
          f"{par.generation_time:.4e} s, projected rho {par.rho * 1e5:+.2e} pcm", flush=True)
    angles = sorted({key(theta(t)) for t in times})
    k, pins, norm, moving = {}, {}, {}, {}
    state, kg = [fwd.flux[g] for g in range(len(fwd.flux))], fwd.k_eff
    for a in angles:
        st = m.tab.state(a)
        s = m.eigen_solver(st)
        res = s.solve(state0=state, k_guess=kg, **STEADY)
        if not res.converged:
            raise RuntimeError(f"static solve at {a} did not converge")
        state, kg = [res.flux[g] for g in range(len(res.flux))], res.k_eff
        k[a] = res.k_eff
        p = m.pins(st, res.flux_numpy)
        pins[a] = p
        # adiabatic shape normalized like the amplitude: <phi0*, V^-1 psi> = 1
        norm[a] = p.sum() / _shape_time_importance(s, res, adj, kin)
        moving[a] = project_effective_kinetics(s, res, adj.flux, kin, k0=fwd.k_eff)
    print(f"static solves at {len(angles)} angles: {time.perf_counter() - t0:.0f} s", flush=True)
    dt = float(times[1] - times[0])
    k0 = k[key(a0)]
    frozen = adiabatic_point_kinetics(lambda a: k[key(a)], theta, par, float(times[-1]), dt,
                                      k0=k0)

    def parameters_at(t):
        q = moving[key(theta(t))]
        return replace(par, rho=1.0 - k0 / k[key(theta(t))], beta=q.beta,
                       generation_time=q.generation_time)

    pk = integrate_point_kinetics(parameters_at, float(times[-1]), dt)
    shape_factor = np.array([norm[key(theta(t))] for t in pk.times]) / norm[key(a0)]
    return dict(angles=np.array(angles), k_static=np.array([k[a] for a in angles]),
                pins_static=np.array([pins[a] for a in angles], dtype=np.float32),
                adiabatic_amplitude=pk.amplitude, adiabatic_power=pk.amplitude * shape_factor,
                adiabatic_power_frozen=frozen.amplitude * shape_factor,
                rho=pk.rho, beta_eff=par.beta, generation_time=par.generation_time,
                beta_theta=np.array([moving[a].beta for a in angles]),
                lambda_theta=np.array([moving[a].generation_time for a in angles]),
                rho_projected=np.array([moving[a].rho for a in angles]),
                k_adjoint=adj.k_eff)


def static_only(case, device="gpu"):
    """Redo the static part of a stored base run (e.g. after a change to it)."""
    f = os.path.join(OUT, f"{case}_base.npz")
    d = dict(np.load(f))
    m = Model(device)
    theta = CASES[case][0]
    kin = m.tab.kinetics(m.base_kinetics, theta(0.0))
    d.update(static(m, theta, d["times"], kin, float(d["k0"])))
    np.savez(f, **d)
    d["case"] = str(d["case"])
    summarize(d)


def shape_errors(d):
    """Per step (pin RMS %, worst pin %, worst assembly %): full vs static shape."""
    idx = {key(a): i for i, a in enumerate(d["angles"])}
    pins = hpmr_pin_layout()
    zop = pins.assembly[pins.fuel]
    rows = []
    for n, t in enumerate(d["rec_t"]):
        a = key(np.interp(t, d["times"], d["theta"]))
        if a not in idx:
            continue
        full, ref = d["pins"][n].astype(float), d["pins_static"][idx[a]].astype(float)
        e = (full / full.mean()) / (ref / ref.mean()) - 1.0
        fa, ra = np.bincount(zop, weights=full), np.bincount(zop, weights=ref)
        keep = ra > 0
        ea = (fa[keep] / fa[keep].mean()) / (ra[keep] / ra[keep].mean()) - 1.0
        rows.append((t, 100 * np.sqrt((e ** 2).mean()), 100 * np.abs(e).max(),
                     100 * np.abs(ea).max()))
    return np.array(rows)


def summarize(d):
    t, P, Pa = d["times"], d["P_pin"], d["adiabatic_power"]
    rel = Pa / P - 1.0
    energy = np.trapezoid(P, t)
    energy_a = np.trapezoid(Pa, t)
    rho = d["rho"] * 1e5
    print(f"case {d['case']}: rho(end) {rho[-1]:+.1f} pcm (${rho[-1] / (d['beta_eff'].sum() * 1e5):+.3f}), "
          f"P(end) full {P[-1]:.5f}, adiabatic {Pa[-1]:.5f} (amplitude only "
          f"{d['adiabatic_amplitude'][-1]:.5f}), solver power {d['power'][-1]:.5f}")
    i = np.argmax(np.abs(rel))
    print(f"  power: max |adiabatic/full - 1| {100 * abs(rel[i]):.3f} % at t = {t[i]:g} s; "
          f"energy {100 * (energy_a / energy - 1):+.3f} %; amplitude-only max "
          f"{100 * np.abs(d['adiabatic_amplitude'] / P - 1).max():.3f} %")
    if "adiabatic_power_frozen" in d:
        rf = d["adiabatic_power_frozen"] / P - 1.0
        dl = 100 * (d["lambda_theta"] / float(d["generation_time"]) - 1)
        drho = 1e5 * np.abs(d["rho_projected"] - (1 - float(d["k0"]) / d["k_static"])).max()
        print(f"  beta_eff, Lambda frozen at theta0: max {100 * np.abs(rf).max():.3f} %, energy "
              f"{100 * (np.trapezoid(d['adiabatic_power_frozen'], t) / energy - 1):+.3f} %; "
              f"Lambda(theta) moves {dl.min():+.3f} to {dl.max():+.3f} %, projected rho - "
              f"(1 - k0/k) max {drho:.2e} pcm")
    se = shape_errors(d)
    for j, name in ((1, "pin RMS"), (2, "worst pin"), (3, "worst assembly")):
        i = np.argmax(se[:, j])
        print(f"  shape lag, {name}: max {se[i, j]:.3f} % at t = {se[i, 0]:g} s, "
              f"end {se[-1, j]:.3f} %")


def report(cases):
    for c in cases:
        for f in sorted(os.listdir(OUT)):
            if f.startswith(c + "_base") and f.endswith(".npz"):
                d = dict(np.load(os.path.join(OUT, f)))
                d["case"] = str(d["case"])
                print(f"[{f}] dt {float(d['dt']):g} s, wall {float(d['wall']):.0f} s")
                summarize(d)
                print()


# --- step 6: the quasi-static S_N reference ---------------------------------

SN_METHOD, SN_REFINE = "sn-p2:4:16", 32
THETA0 = {"A": 90.0, "B": 90.0, "C": 90.0, "D": 180.0, "E": 40.0, "E2": 43.75}
# Where the quasi-static reference itself holds (step 5): D to 0.5 s (-$1.6;
# later the flux is precursor-driven, 6.6 % in power at 2 s); E ($0.88, near
# prompt critical) not at all (8 % in power): compared as a stress case only.
WINDOW = {"D": 0.5}
STRESS = {"E"}


class SnReference:
    """Adiabatic S_N reference along a drum path: k and pin powers at the exact
    refine-32 angles (linear in angle between them), the shape norm
    sum(pins) / <phi0*, V^-1 psi(theta)> with the adjoint scalar flux at theta0,
    and beta_eff, Lambda moving with the angle under the same fixed weight phi0*
    (as the SPH adiabatic solution). beta_eff(theta) is exact from the scalar
    fluxes; Lambda(theta) is the P_L-moment Lambda at theta0
    (tools/hpmr_sn_kinetics.py) times the scalar-flux ratio
    Lambda_P0(theta) / Lambda_P0(theta0). ``frozen=True`` keeps both at theta0."""

    def __init__(self, theta0):
        from hpmr_pin_reference import cache_path, method_tag
        from garnet.benchmarks.hpmr_assembly import FUEL as FUEL_COMPACT
        from garnet.benchmarks.hpmr_core_pins import PIN_OFFSET, build_hpmr2d_pins
        from garnet.pin_power import pin_footprints
        from garnet.quasistatic import EffectiveKinetics
        tag = f"{method_tag(SN_METHOD)}_r{SN_REFINE}_a"
        kin_file = os.path.join(CACHE, f"{tag}{theta0:g}_kinetics.npz")
        if not os.path.exists(kin_file):
            raise SystemExit(f"S_N kinetics at {theta0:g} deg missing: {kin_file}")
        kd = np.load(kin_file)
        lib = hpmr_kinetics_11g()
        self.params = EffectiveKinetics(rho=0.0, generation_time=float(kd["generation_time"]),
                                        beta=np.asarray(kd["beta_eff"]), decay=lib.decay.copy(),
                                        time_importance=1.0, fission_importance=1.0)
        adj = np.asarray(kd["adjoint_flux"])
        p, pins = build_hpmr2d_pins(refine=SN_REFINE, drum_angle_deg=theta0)
        vol = np.broadcast_to(np.asarray(p.grid.cell_volume, float), adj.shape[1:])
        table, _ = fission_energy_xs(p.materials)
        kf = table[PIN_OFFSET + FUEL_COMPACT]
        W = pin_footprints(p.raster, pins.xy[pins.fuel], R_FUEL)
        v = np.asarray(lib.velocities, float)[:, None, None, None]
        w = adj * (vol / v)
        # fission and delayed-emission weights per cell (fuel does not move)
        mm = np.asarray(p.material_map)
        G = adj.shape[0]
        nsf = np.array([np.asarray(m.nu_sigma_f, float) for m in p.materials])
        chi = np.array([np.zeros(G) if m.chi is None else np.asarray(m.chi, float)
                        for m in p.materials])
        live = np.ones(mm.shape, bool) if p.active is None else np.asarray(p.active, bool)
        nsf_c = np.moveaxis(nsf[mm], -1, 0) * live
        wchi = (adj * vol * np.moveaxis(chi[mm], -1, 0) * live).sum(axis=0)     # sum_g w chi_g
        cd = lib.chi_delayed
        I = len(lib.beta)
        if cd is None:
            wd = np.repeat(wchi[None], I, axis=0)
        else:
            cd = np.asarray(cd, float)
            cd = np.repeat(cd[None], I, axis=0) if cd.ndim == 1 else cd        # (I, G)
            wd = np.einsum("ig,g...->i...", cd, adj * vol * live)
        betas = np.asarray(lib.beta, float)
        k0_kin = float(kd["k"])
        files = [f for f in os.listdir(CACHE) if f.startswith(tag) and f.endswith(".npz")
                 and f[len(tag):-4].replace(".", "", 1).isdigit()]
        angles, k, pin, norm, beta_t, lam_t = [], [], [], [], [], []
        for f in sorted(files, key=lambda f: float(f[len(tag):-4])):
            d = np.load(os.path.join(CACHE, f))
            if not bool(d["converged"]):
                raise RuntimeError(f"{f} did not converge")
            flux = np.asarray(d["flux"])
            rate = (kf * footprint_flux(W, flux)).sum(axis=1)
            angles.append(float(f[len(tag):-4]))
            k.append(float(d["k"]))
            pin.append(rate / rate.mean())
            time_imp = float((w * flux).sum())
            norm.append(rate.sum() / time_imp)
            src = (nsf_c * flux).sum(axis=0)
            fis = float((wchi * src).sum())
            beta_t.append(betas * np.array([float((wd[i] * src).sum()) for i in range(I)]) / fis)
            lam_t.append(k0_kin * time_imp / fis)
        self.angles, self.k = np.array(angles), np.array(k)
        self.pins, self.norm = np.array(pin), np.array(norm)
        self._theta0 = theta0
        self.k0 = self.k_at(theta0)
        self.norm0 = float(np.interp(theta0, self.angles, self.norm))
        if abs(k0_kin - self.k0) > 1e-6:
            raise RuntimeError("kinetics forward k differs from the cached reference")
        self.beta_t = np.array(beta_t)
        lam_t = np.array(lam_t)
        self.lam_p0_ratio = lam_t / float(np.interp(theta0, self.angles, lam_t))
        b0 = np.array([np.interp(theta0, self.angles, self.beta_t[:, i]) for i in range(I)])
        self.beta_check = float(np.abs(b0 / self.params.beta - 1).max())
        if self.beta_check > 1e-6:
            raise RuntimeError(f"scalar-flux beta_eff at theta0 differs from the kinetics job "
                               f"by {self.beta_check:.2e}")

    def k_at(self, a):
        if getattr(self, "_kfit", None) is not None:
            return float(self._kfit(a))
        return float(np.interp(a, self.angles, self.k))

    def smooth_k(self, lo, hi, degree=3):
        """Replace the linear interpolation of the exact k(theta) by a
        least-squares polynomial of ``degree`` over [lo, hi]: the refine-32
        mesh puts a ripple of up to 27 pcm on k(theta) between the exact
        angles (88.75 deg), which the reference reactivity should not carry.
        Returns (residual RMS, max |residual| in pcm, points)."""
        sel = (self.angles >= lo - 1e-9) & (self.angles <= hi + 1e-9)
        a, k = self.angles[sel], self.k[sel]
        if a.size < degree + 3:
            raise ValueError(f"too few exact angles in [{lo}, {hi}] for degree {degree}")
        self._kfit = np.polynomial.Polynomial.fit(a, k, degree)
        r = 1e5 * (k - self._kfit(a))
        self.k0 = self.k_at(self._theta0)
        return float(np.sqrt((r ** 2).mean())), float(np.abs(r).max()), int(a.size)

    def pins_at(self, a):
        j = int(np.clip(np.searchsorted(self.angles, a), 1, len(self.angles) - 1))
        a0, a1 = self.angles[j - 1], self.angles[j]
        w = (a - a0) / (a1 - a0)
        return (1 - w) * self.pins[j - 1] + w * self.pins[j]

    def power(self, theta, t_end, dt, frozen=False):
        if frozen:
            pk = adiabatic_point_kinetics(self.k_at, theta, self.params, t_end, dt, k0=self.k0)
        else:
            def parameters_at(t):
                a = theta(t)
                beta = np.array([np.interp(a, self.angles, self.beta_t[:, i])
                                 for i in range(self.beta_t.shape[1])])
                lam = self.params.generation_time * float(np.interp(a, self.angles,
                                                                    self.lam_p0_ratio))
                return replace(self.params, rho=1.0 - self.k0 / self.k_at(a), beta=beta,
                               generation_time=lam)
            pk = integrate_point_kinetics(parameters_at, t_end, dt)
        shape = np.array([np.interp(theta(t), self.angles, self.norm) for t in pk.times])
        return pk, pk.amplitude * shape / self.norm0


def compare(cases):
    """SPH-DF transients (full and adiabatic) against the quasi-static S_N."""
    pins = hpmr_pin_layout()
    asm = pins.assembly[pins.fuel]
    refs = {}
    for c in cases:
        f = os.path.join(OUT, f"{c}_base.npz")
        if not os.path.exists(f):
            print(f"case {c}: no SPH-DF run yet\n")
            continue
        d = dict(np.load(f))
        theta, _, _ = CASES[c]
        if THETA0[c] not in refs:                         # A, B, C share theta0 = 90
            refs[THETA0[c]] = SnReference(THETA0[c])
        ref = refs[THETA0[c]]
        t, dt = d["times"], float(d["dt"])
        ref._kfit = None
        ref.k0 = ref.k_at(ref._theta0)
        _, P_lin = ref.power(theta, float(t[-1]), dt)
        k0_sph = float(d["k0"])
        win_ = t <= WINDOW.get(c, np.inf) + 1e-12
        path = np.array([theta(x) for x in t[win_]])
        lo, hi = max(0.0, path.min() - 5.0), min(180.0, path.max() + 5.0)
        if hi - lo < 10.0:
            lo, hi = max(0.0, 0.5 * (lo + hi) - 5.0), min(180.0, 0.5 * (lo + hi) + 5.0)
        holds = sorted({float(theta(0.0)), float(theta(t[win_][-1]))})
        alt = {}
        for deg in (4, 3):
            rms, mx, npts = ref.smooth_k(lo, hi, deg)
            alt[deg] = [ref.k_at(h) for h in holds]
        exact = [float(np.interp(h, ref.angles, ref.k)) for h in holds]
        print(f"  S_N k(theta) smoothed: cubic over [{lo:g}, {hi:g}] deg ({npts} exact angles), "
              f"residual RMS {rms:.1f} / max {mx:.1f} pcm; at "
              + ", ".join(f"{h:g} deg fit - exact {1e5 * (f - e):+.1f} (quartic {1e5 * (q - f):+.1f})"
                          for h, f, e, q in zip(holds, alt[3], exact, alt[4])) + " pcm")
        pk, P_sn = ref.power(theta, float(t[-1]), dt)
        win = t <= WINDOW.get(c, np.inf) + 1e-12
        if c in WINDOW:
            print(f"case {c}: compared to t = {WINDOW[c]:g} s, where the quasi-static "
                  f"reference holds (step 5)")
        if c in STRESS:
            print(f"case {c}: STRESS CASE, the quasi-static reference itself is off by "
                  f"several % here (step 5)")
        _, P_sn_frozen = ref.power(theta, float(t[-1]), dt, frozen=True)
        lam_path = ref.lam_p0_ratio[(ref.angles >= min(d["theta"]) - 1e-9)
                                    & (ref.angles <= max(d["theta"]) + 1e-9)]
        beta = ref.params.beta.sum()
        rho_sph = 1.0 - k0_sph / d["k_static"]
        rho_sn = np.array([1.0 - ref.k0 / ref.k_at(a) for a in d["angles"]])
        on = (d["angles"] >= theta(t[win][-1]) - 1e-9) if theta(t[-1]) < theta(0) else \
             (d["angles"] <= theta(t[win][-1]) + 1e-9)
        drho = 1e5 * (rho_sph - rho_sn)[on] if c in WINDOW else 1e5 * (rho_sph - rho_sn)
        print(f"case {c}: theta {theta(0):g} -> {theta(t[-1]):g} deg; S_N beta_eff "
              f"{beta * 1e5:.1f} pcm, Lambda {ref.params.generation_time:.4e} s; SPH-DF "
              f"beta_eff {d['beta_eff'].sum() * 1e5:.1f} pcm, Lambda {float(d['generation_time']):.4e} s")
        print(f"  S_N Lambda(theta)/Lambda(theta0) on the path's exact angles: "
              f"{100 * (lam_path.min() - 1):+.3f} to {100 * (lam_path.max() - 1):+.3f} %; "
              f"frozen-kinetics S_N reference vs moving: max "
              f"{100 * np.abs(P_sn_frozen / P_sn - 1).max():.3f} %")
        print(f"  static reactivity along the path, SPH-DF - S_N: max |d rho| "
              f"{np.abs(drho).max():.1f} pcm (${np.abs(drho).max() / (beta * 1e5):.3f}); "
              f"rho_SN(end) {1e5 * pk.rho[-1]:+.1f} pcm")
        P = d["P_pin"]
        curves = [("SPH-DF full", P), ("SPH-DF adiabatic", d["adiabatic_power"]),
                  ("S_N linear-interp k", P_lin)]
        if "beta_theta" in d:
            # attribution: the SPH-DF adiabatic solution with the S_N k(theta), i.e.
            # SPH-DF kinetics data and shape norm, S_N static reactivity
            idx = {key(a): i for i, a in enumerate(d["angles"])}
            p0 = replace(ref.params, beta=d["beta_eff"],
                         generation_time=float(d["generation_time"]))

            def at(tt):
                a = theta(tt)
                i = idx[key(a)]
                return replace(p0, rho=1.0 - ref.k0 / ref.k_at(a), beta=d["beta_theta"][i],
                               generation_time=float(d["lambda_theta"][i]))
            amp = integrate_point_kinetics(at, float(t[-1]), dt).amplitude
            curves.append(("SPH kinetics, S_N k", amp * d["adiabatic_power"]
                           / d["adiabatic_amplitude"]))
        for name, x in curves:
            rel = (x / P_sn - 1.0)[win]
            i = np.argmax(np.abs(rel))
            tw = t[win]
            print(f"  P(t) {name:>19} vs S_N: max {100 * rel[i]:+.3f} % at t = {tw[i]:g} s, "
                  f"end {100 * rel[-1]:+.3f} %, energy "
                  f"{100 * (np.trapezoid(x[win], tw) / np.trapezoid(P_sn[win], tw) - 1):+.3f} %")
        rows = []
        for n, tn in enumerate(d["rec_t"]):
            if tn > WINDOW.get(c, np.inf) + 1e-12:
                break
            full = d["pins"][n].astype(float)
            r = ref.pins_at(float(np.interp(tn, t, d["theta"])))
            e = (full / full.mean()) / r - 1.0
            fa, ra = np.bincount(asm, weights=full), np.bincount(asm, weights=r)
            keep = ra > 0
            ea = (fa[keep] / fa[keep].mean()) / (ra[keep] / ra[keep].mean()) - 1.0
            rows.append((tn, 100 * np.sqrt((e ** 2).mean()), 100 * np.abs(e).max(),
                         100 * np.abs(ea).max()))
        rows = np.array(rows)
        for j, name in ((1, "pin RMS"), (2, "worst pin"), (3, "worst assembly")):
            i = np.argmax(rows[:, j])
            print(f"  shape, {name}: max {rows[i, j]:.3f} % at t = {rows[i, 0]:g} s, "
                  f"t = 0 {rows[0, j]:.3f} %, end {rows[-1, j]:.3f} %")
        for v in ("v", "gamma0"):
            fv = os.path.join(OUT, f"{c}_{v}.npz")
            if os.path.exists(fv):
                dv = np.load(fv)
                rel = (dv["P_pin"] / P_sn - 1.0)[win]
                dif = (dv["P_pin"] / P - 1.0)[win]
                print(f"  variant {v:>6}: P vs S_N max {100 * np.abs(rel).max():.3f} %, "
                      f"vs base max {100 * np.abs(dif).max():.3f} %")
        print()


def compare_full(case="D", refine=32, dt=0.005, mode="anderson"):
    """SPH-DF transients and the quasi-static S_N against the FULL space-time
    S_N transient (tools/hpmr_sn_transient.py): power, energy, and pin and
    assembly shapes at every common time."""
    full_file = os.path.join(CACHE, "drum_transients_sn", f"{case}_r{refine}_dt{dt:g}_{mode}.npz")
    F = dict(np.load(full_file))
    tF, PF = F["times"], F["P_pin"]
    pins = hpmr_pin_layout()
    asm = pins.assembly[pins.fuel]
    theta, _, _ = CASES[case]
    print(f"case {case}: full S_N transient refine {refine}, dt {dt:g} s, to t = {tF[-1]:g} s "
          f"(theta {theta(0):g} -> {theta(tF[-1]):g} deg), P(end) {PF[-1]:.5f}")
    ref = SnReference(THETA0[case])

    def shape_err(a, b):
        e = (a / a.mean()) / (b / b.mean()) - 1.0
        fa, fb = np.bincount(asm, weights=a), np.bincount(asm, weights=b)
        keep = fb > 0
        ea = (fa[keep] / fa[keep].mean()) / (fb[keep] / fb[keep].mean()) - 1.0
        return 100 * np.sqrt((e ** 2).mean()), 100 * np.abs(e).max(), 100 * np.abs(ea).max()

    def report(name, t, P, pin_at=None):
        common = [i for i, x in enumerate(t) if x <= tF[-1] + 1e-12]
        t, P = t[common], P[common]
        PFi = np.interp(t, tF, PF)
        rel = P / PFi - 1.0
        i = np.argmax(np.abs(rel))
        print(f"  {name:>34}: P max {100 * rel[i]:+.3f} % at t = {t[i]:g} s, end {100 * rel[-1]:+.3f} %, "
              f"energy {100 * (np.trapezoid(P, t) / np.trapezoid(PFi, t) - 1):+.3f} %")
        if pin_at is not None:
            rows = []
            for n, tn in enumerate(F["rec_t"]):
                q = pin_at(tn)
                if q is not None:
                    rows.append((tn,) + shape_err(q, F["pins"][n].astype(float)))
            rows = np.array(rows)
            for j, nm in ((1, "pin RMS"), (2, "worst pin"), (3, "worst assembly")):
                k = np.argmax(rows[:, j])
                print(f"  {'':>34}  shape {nm}: max {rows[k, j]:.3f} % at t = {rows[k, 0]:g} s, "
                      f"end {rows[-1, j]:.3f} %")

    for label, out in (("SPH-DF full, 5 deg tables", OUT),
                       ("SPH-DF full, 2.5 deg tables", OUT + "_t2.5")):
        f = os.path.join(out, f"{case}_base.npz")
        if not os.path.exists(f):
            continue
        d = dict(np.load(f))
        idx = {round(float(x), 9): n for n, x in enumerate(d["rec_t"])}
        report(label, d["times"], d["P_pin"],
               lambda tn, d=d, idx=idx: (d["pins"][idx[round(float(tn), 9)]].astype(float)
                                         if round(float(tn), 9) in idx else None))
    t_end = float(tF[-1])
    for label, smooth in (("quasi-static S_N, linear k", False), ("quasi-static S_N, cubic k", True)):
        ref._kfit = None
        if smooth:
            path = np.array([theta(x) for x in tF])
            lo, hi = max(0.0, path.min() - 5.0), min(180.0, path.max() + 5.0)
            if hi - lo < 10.0:
                lo, hi = max(0.0, hi - 10.0), min(180.0, lo + 10.0)
            ref.smooth_k(lo, hi, 3)
        ref.k0 = ref.k_at(ref._theta0)
        pk, P_qs = ref.power(theta, t_end, dt)
        report(label, pk.times, P_qs,
               (lambda tn: ref.pins_at(theta(tn))) if smooth else None)


def rescale_forms(out_dir, table_step):
    """Apply the form-function renormalization (form_scale of each table
    entry, written by the renormalization of 2026-11-21) to stored SPH-DF
    runs made with the raw form functions: pin powers, the pin-sum power and
    the adiabatic shape norm all scale with the interpolated form scale at
    theta(t). The raw values are kept as <key>_raw; runs already corrected
    are skipped."""
    sc = np.loadtxt(os.path.join(CACHE, TABLES, "form_scale.txt"))
    keep = np.abs(sc[:, 0] / table_step - np.round(sc[:, 0] / table_step)) < 1e-9
    A, S = sc[keep, 0], sc[keep, 1]

    def scale(theta):
        return np.interp(theta, A, S)

    for f in sorted(os.listdir(out_dir)):
        if not (f.endswith(".npz") and "_" in f):
            continue
        path = os.path.join(out_dir, f)
        d = dict(np.load(path))
        if bool(d.get("form_scale_corrected", False)) or "P_pin" not in d:
            continue
        th, th_rec = d["theta"], np.interp(d["rec_t"], d["times"], d["theta"])
        s0 = scale(th[0])
        for k in ("P_pin", "pins", "adiabatic_power", "adiabatic_power_frozen", "pins_static"):
            if k in d:
                d[k + "_raw"] = d[k].copy()
        d["P_pin"] = d["P_pin"] * scale(th) / s0
        d["pins"] = d["pins"] * scale(th_rec)[:, None]
        for k in ("adiabatic_power", "adiabatic_power_frozen"):
            if k in d:
                d[k] = d[k] * scale(th) / s0
        if "pins_static" in d:
            d["pins_static"] = d["pins_static"] * scale(d["angles"])[:, None]
        d["form_scale_corrected"] = True
        np.savez(path, **d)
        print(f"{f}: P_pin(end) {d['P_pin_raw'][-1]:.5f} -> {d['P_pin'][-1]:.5f} "
              f"(scale {scale(th[-1]) / s0:.5f})")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        run(sys.argv[2], *(sys.argv[3:5] if len(sys.argv) > 3 else ()),
            dt=sys.argv[5] if len(sys.argv) > 5 else None)
    elif mode == "static":
        static_only(sys.argv[2], *sys.argv[3:4])
    elif mode == "compare":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        compare(sys.argv[2].split(",") if len(sys.argv) > 2 else sorted(CASES))
    elif mode == "compare_full":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        compare_full(*(sys.argv[2:3] or ["D"]))
    elif mode == "rescale_forms":
        rescale_forms(sys.argv[2], float(sys.argv[3]))
    elif mode == "report":
        report(sys.argv[2].split(",") if len(sys.argv) > 2 else sorted(CASES))
    else:
        raise SystemExit(__doc__)
