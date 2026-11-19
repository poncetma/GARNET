"""Point-kinetics parameters of the HP-MR S_N reference at given drum angles.

For each angle: forward and adjoint (``adjoint=True``) solves of the
pin-resolved 2-D core with the reference recipe (tools/hpmr_pin_reference.py),
then :func:`garnet.tri_sn.sn_effective_kinetics` with the library kinetics
(:func:`garnet.benchmarks.hpmr_thermal.hpmr_kinetics_11g`): beta_eff per
family and Lambda from the P_L flux moments (and the scalar-flux Lambda for
comparison). With ``insertion`` it also solves the core with Sigma_t + eps/v
(one-sided +eps, +2 eps, Richardson) for the exact first-order Lambda.

Usage: python tools/hpmr_sn_kinetics.py refine angles [method] [device] [insertion]
       e.g. 32 90,180,40 sn-p2:4:16 gpu insertion
Results are cached as <method>_r<refine>_a<angle>_kinetics.npz beside the references.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hpmr_pin_reference import DEFAULT_CACHE, TOL_SN, method_tag, parse, problem_for  # noqa: E402

from garnet.backend import asnumpy  # noqa: E402
from garnet.benchmarks.hpmr_thermal import hpmr_kinetics_11g  # noqa: E402
from garnet.materials import Material  # noqa: E402
from garnet.tri_sn import TriSNTransportSolver, sn_effective_kinetics  # noqa: E402

refine = int(sys.argv[1])
angles = [float(a) for a in sys.argv[2].split(",")]
method = sys.argv[3] if len(sys.argv) > 3 else "sn-p2:4:16"
device = sys.argv[4] if len(sys.argv) > 4 else "auto"
insertion = len(sys.argv) > 5 and sys.argv[5] == "insertion"
cache = os.environ.get("GARNET_REF_CACHE", DEFAULT_CACHE)
kind, tc, order, quad = parse(method)
if kind != "sn":
    raise SystemExit("an S_N method is needed")
kin = hpmr_kinetics_11g()
v = np.asarray(kin.velocities, dtype=float)


def solver(p, adjoint=False, materials=None):
    return TriSNTransportSolver(p.grid, p.materials if materials is None else materials,
                                p.material_map, active=p.active, mix_material=p.mix_material,
                                mix_weight=p.mix_weight, n_polar=quad[0], n_azi=quad[1],
                                bc="vacuum", scheme="scb", device=device,
                                scattering_order=order, adjoint=adjoint)


def with_inverse_speed(m, eps):
    return Material(name=m.name, diffusion=m.diffusion,
                    sigma_a=np.asarray(m.sigma_a) + eps / v, nu_sigma_f=m.nu_sigma_f,
                    sigma_s=m.sigma_s, chi=m.chi, total=np.asarray(m.sigma_t) + eps / v,
                    sigma_s_moments=m.sigma_s_moments, kappa_fission=m.kappa_fission)


print(f"{method} refine {refine}: S_N point-kinetics parameters, library kinetics "
      f"(beta {kin.beta.sum() * 1e5:.0f} pcm, {len(kin.beta)} families)\n", flush=True)
for a in angles:
    t0 = time.perf_counter()
    p = problem_for(method, refine, a)
    sf = solver(p)
    rf = sf.solve(**TOL_SN)
    sa = solver(p, adjoint=True)
    ra = sa.solve(**TOL_SN)
    par = sn_effective_kinetics(sf, rf, sa, ra, kin)
    p0 = sn_effective_kinetics(sf, rf, sa, ra, kin, angular=False)
    out = dict(angle=a, k=rf.k_eff, k_adjoint=ra.k_eff, beta_eff=par.beta,
               generation_time=par.generation_time, generation_time_p0=p0.generation_time,
               converged=bool(rf.converged and ra.converged),
               adjoint_flux=asnumpy(ra.flux))
    line = (f"angle {a:g}: k {rf.k_eff:.7f}, adjoint k {ra.k_eff:.7f} "
            f"({1e5 * (ra.k_eff - rf.k_eff):+.2f} pcm), beta_eff {par.beta_total * 1e5:.1f} pcm "
            f"({' '.join(f'{b * 1e5:.1f}' for b in par.beta)}), Lambda {par.generation_time:.4e} s "
            f"(scalar-flux {p0.generation_time:.4e})")
    print(line + f" [{time.perf_counter() - t0:.0f} s]", flush=True)
    if insertion:
        # One-sided (+eps, +2 eps) and Richardson: subtracting eps/v would turn
        # the net (n,2n) absorption of Be into a negative removal.
        eps = 1e-3 * float(np.min(np.asarray(p.materials[0].sigma_t)) * v.min())
        lam0 = 1.0 / rf.k_eff
        d = []
        for s in (1.0, 2.0):
            mats = [with_inverse_speed(m, s * eps) for m in p.materials]
            d.append((1.0 / solver(p, materials=mats).solve(**TOL_SN).k_eff - lam0) / (s * eps))
        exact = rf.k_eff * (2.0 * d[0] - d[1])
        out["generation_time_insertion"] = exact
        print(f"angle {a:g}: 1/v insertion Lambda {exact:.4e} s (one-sided {rf.k_eff * d[0]:.4e}); "
              f"moments {par.generation_time / exact - 1:+.2e}, scalar-flux "
              f"{p0.generation_time / exact - 1:+.2e} [{time.perf_counter() - t0:.0f} s]", flush=True)
    np.savez(os.path.join(cache, f"{method_tag(method)}_r{refine}_a{a:g}_kinetics.npz"), **out)
