"""Superhomogenization (SPH) against a transport reference.

Coarse diffusion with flux-weighted homogenized cross sections does not, on its
own, reproduce a transport reference: collapsing a heterogeneous region to one
set of constants preserves that region's *reaction rates at the reference flux*,
but the coarse diffusion flux differs from the reference, so the rates (and the
eigenvalue) drift. SPH restores the equivalence by multiplying each region-group
cross section by a factor mu chosen so the coarse flux matches the reference
region flux again.

The pipeline:

* :func:`flux_weighted_homogenize` -- collapse a fine reference solution into one
  Material per coarse region, flux-and-volume weighted (this step preserves the
  reference reaction rates by construction).
* :func:`region_average` -- the coarse solve's region fluxes, averaged with the
  coarse mesh's own region map and cell volumes. The coarse mesh may be a node
  mesh much coarser than the reference mesh.
* :func:`sph_correct` -- solve for mu so that mu * Phi = phi_ref, with either
  the per-group flux-volume normalization (reflective generation geometries) or
  the fixed-k, no-normalization formulation (vacuum-bounded geometries, the
  Griffin assembly-level workflow), optionally restricted to the fuel.
* :func:`sph_kinetics` -- per-material speeds v / mu for transients with the
  corrected materials.

Conventions (docs/qa/sph_audit_2026-11-13.md): every cross section AND the
diffusion coefficient are multiplied by mu; chi is not corrected; 1/v becomes
mu/v.

The transport reference is any garnet eigensolver's scalar-flux Result -- any of
the simplified-transport families GARNET offers above diffusion: SP3/TriSP3, the
double-PN SDP1 and SDP2/TriSDP1, TriSDP2, or the standard SPN orders. For every
one of these the Result carries the *physical* scalar flux phi0 (the
reconstructed 0th angular moment: SP3 forms phi0 = Phi1 - 2 phi2, SDPN dots the
even-moment vector with its phi0 closure weights), which is exactly the field
that drives the isotropic reaction rates SPH preserves -- so swapping the
reference between families feeds SPH genuinely different angular closures of the
same problem rather than a re-extraction of one field. See
examples/hpmr_sph_reference_families.py for an SP3-vs-SDP1-vs-SDP2 comparison on
the HP-MR drum arc.

References: G. Marleau, A. Hebert, R. Roy, "A User Guide for DRAGON Version5",
IGE-335, Polytechnique Montreal, Sec. 3.11 (Eqs. 3.42-3.46); A. Hebert, "A
Reformulation of the Transport-Transport SPH Equivalence Technique", 7ICMSNSE,
Ottawa (2015); Labouré et al., "Hybrid super homogenization and discontinuity
factor method for continuous finite element diffusion", Ann. Nucl. Energy 128
(2019) 443-454, doi:10.1016/j.anucene.2019.01.003; Kumar et al., "Transient
multiphysics simulations with pin power reconstruction in the Griffin reactor
physics code", Ann. Nucl. Energy 230 (2026) 112111,
doi:10.1016/j.anucene.2025.112111.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np

from .backend import asnumpy
from .linalg import anderson_step as _anderson_step
from .materials import Material
from .power import fission_energy_xs


def _cell_tables(materials, material_map):
    """Per-cell cross-section tables from a material map, flattened to (N, ...).

    Returns a dict of arrays indexed by flat cell: sigma_a, nu_sigma_f, sigma_t,
    diffusion, chi (each (N, G)) and sigma_s ((N, G, G)).
    """
    mats = list(materials)
    G = mats[0].n_groups
    flat = np.asarray(material_map).reshape(-1)
    sa = np.array([m.sigma_a for m in mats])          # (M, G)
    nf = np.array([m.nu_sigma_f for m in mats])
    st = np.array([m.sigma_t for m in mats])
    df = np.array([m.diffusion for m in mats])
    ch = np.array([m.chi for m in mats])
    ss = np.array([m.sigma_s for m in mats])          # (M, G, G)
    return dict(sigma_a=sa[flat], nu_sigma_f=nf[flat], sigma_t=st[flat],
                diffusion=df[flat], chi=ch[flat], sigma_s=ss[flat], G=G)


def _mixed_entries(material_map, region_map, cell_volume, shape_n,
                   mix_material=None, mix_weight=None, mix_region_map=None):
    """Expand per-cell volume mixing into fractional sub-cell entries.

    A cell blended between two materials (the solvers' ``mix_material`` /
    ``mix_weight``, e.g. the volume-mixed control-drum arc) is not one material,
    so homogenizing it by ``material_map`` alone attributes the whole cell to the
    base material and loses the absorber's reaction rate. Splitting it into two
    entries -- (1-w) of the volume with the base material, w with the mix partner,
    both carrying the same cell-average flux -- is what keeps
    ``Sigma_hom <phi> V == the fine reaction rate`` exact.

    Returns (cell_index, region, volume, material_index) arrays over entries.
    """
    idx = np.arange(shape_n)
    base_mat = np.asarray(material_map).reshape(-1)
    base_reg = np.asarray(region_map).reshape(-1)
    # Broadcast against the UNflattened map, so a documented (*shape) per-cell
    # volume array lines up cell for cell, then flatten like the maps above.
    V = np.broadcast_to(np.asarray(cell_volume, dtype=float),
                        np.shape(material_map)).reshape(-1).astype(float)
    if mix_material is None or mix_weight is None:
        return idx, base_reg, V, base_mat
    mm = np.asarray(mix_material).reshape(-1)
    mw = np.asarray(mix_weight, dtype=float).reshape(-1)
    live = (mw > 0.0) & (mm >= 0)
    mreg = (mm if mix_region_map is None
            else np.asarray(mix_region_map).reshape(-1))
    return (np.concatenate([idx, idx[live]]),
            np.concatenate([base_reg, mreg[live]]),
            np.concatenate([V * (1.0 - np.where(live, mw, 0.0)), V[live] * mw[live]]),
            np.concatenate([base_mat, mm[live]]))


def flux_weighted_homogenize(flux, materials, material_map, region_map,
                             cell_volume=1.0, mix_material=None,
                             mix_weight=None, mix_region_map=None):
    """Collapse a fine reference solution into one Material per coarse region.

    flux         : (G, *shape) reference scalar flux (e.g. an SP3 Result.flux,
                   or a TriSNTransportSolver SNResult.flux -- any reference that
                   carries a physical scalar flux).
    materials    : fine material list. material_map : (*shape) index into it.
    region_map   : (*shape) coarse-region index in [0, R); the homogenization
                   regions (e.g. one per assembly).
    cell_volume  : scalar cell volume, or a (*shape) array (uniform grids: pass
                   the constant; it cancels in the ratios but sets the scale of
                   the returned region volumes/fluxes). Non-uniform grids need
                   the array -- on an r-z grid ``grid.cell_volumes`` (annuli),
                   not the rectangular ``grid.cell_volume`` -- and the same
                   value must go to :func:`region_average`.
    mix_material, mix_weight : the solvers' per-cell two-material blend. Supply
                   them and each blended cell is split into two fractional
                   entries, so a volume-mixed absorber arc is homogenized at its
                   true volume fraction instead of being attributed wholly to
                   the base material. Without this, SPH is restricted to a
                   rasterized (discrete per-cell) geometry, which needs a very
                   fine mesh before a thin arc is resolved at all.
    mix_region_map : region index for the mixed component; defaults to
                   ``mix_material`` (correct for the common "one region per
                   material" mapping).

    Returns (homogenized_materials, region_flux, region_volume):
      homogenized_materials : list of R Materials, each cross section
        flux-and-volume weighted over its region and group so that
        Sigma_hom * <phi>_region * V_region == the region's fine reaction rate.
        kappa_fission is collapsed the same way as nu_sigma_f whenever the fine
        materials carry it (non-fissile ones without it count as zero; a mix of
        fissile materials with and without it is rejected, see
        :func:`garnet.power.fission_energy_xs`); non-fissile regions get None.
      region_flux           : (R, G) volume-average scalar flux per region.
      region_volume         : (R,) total volume per region.

    Reaction-rate preservation is exact by construction (see
    tests/verification/test_sph.py); it is what lets the SPH factor solve, added
    next, recover the reference eigenvalue rather than just its flux shape.
    """
    flux = asnumpy(flux)          # device Result.flux -> host (see module note)
    G = int(flux.shape[0])
    ncell = int(np.asarray(material_map).size)
    entry_cells, entry_regions, entry_volumes, entry_materials = _mixed_entries(
        material_map, region_map, cell_volume, ncell, mix_material, mix_weight,
        mix_region_map)
    material_tables = _cell_tables(materials, entry_materials)
    # kappa*Sigma_f is a reaction-rate coefficient like nu*Sigma_f: dropping it
    # here sent every SPH-homogenized model's power edit back to nu-weighting.
    energy_table, energy_key = fission_energy_xs(materials)
    kappa_table = (energy_table[entry_materials]
                   if energy_key == "kappa_fission" else None)
    R = int(max(entry_regions.max(), np.asarray(region_map).max())) + 1
    region_counts = np.bincount(entry_regions, weights=entry_volumes, minlength=R)
    if np.any(region_counts <= 0.0):
        # Returned lists are indexed by region, so an empty index cannot simply
        # be skipped; it would otherwise surface as 0/0 and an IndexError.
        raise ValueError("region_map must use every index in [0, R) on cells of "
                         "positive volume; empty regions: "
                         f"{np.flatnonzero(region_counts <= 0.0).tolist()}")
    phi = flux.reshape(G, -1)[:, entry_cells]               # (G, n_entries)

    region_flux = np.zeros((R, G))
    region_volume = np.zeros(R)
    mats_out = []
    for i in range(R):
        region_entries = np.where(entry_regions == i)[0]
        region_entry_volumes = entry_volumes[region_entries]
        region_flux_entries = phi[:, region_entries]
        flux_volume_weights = region_flux_entries * region_entry_volumes
        group_weight_sums = flux_volume_weights.sum(axis=1)
        vol = region_entry_volumes.sum()
        region_volume[i] = vol
        region_flux[i] = (region_flux_entries * region_entry_volumes).sum(axis=1) / vol

        # Fall back to volume weighting in any group with no flux (a void or a
        # decoupled region), so the region still yields a finite Material rather
        # than a division-by-zero; such regions carry no reaction rate and are
        # frozen (mu = 1) by the SPH solve anyway.
        safe_weight_sums = np.where(group_weight_sums > 0, group_weight_sums, 1.0)

        def flux_weighted_average(table):                                # flux-volume weighted, (G,)
            flux_weighted = (table[region_entries].T * flux_volume_weights).sum(axis=1) / safe_weight_sums
            volume_weighted = (table[region_entries].T * region_entry_volumes).sum(axis=1) / vol
            return np.where(group_weight_sums > 0, flux_weighted, volume_weighted)

        sigma_a = flux_weighted_average(material_tables["sigma_a"])
        nu_sigma_f = flux_weighted_average(material_tables["nu_sigma_f"])
        kappa_fission = (flux_weighted_average(kappa_table)
                         if kappa_table is not None and np.any(nu_sigma_f > 0)
                         else None)
        sigma_t = flux_weighted_average(material_tables["sigma_t"])
        # diffusion: flux-weight the transport cross section 1/(3D), then invert
        sigma_tr = flux_weighted_average(1.0 / (3.0 * material_tables["diffusion"]))
        diffusion = 1.0 / (3.0 * sigma_tr)
        # scattering g->g' weighted by the source-group (g) flux-volume weight
        scattering_tables = material_tables["sigma_s"][region_entries]                    # (ni, G, G)
        sigma_s = np.zeros((G, G))
        for g in range(G):
            source_group_weights = flux_volume_weights[g]
            denom = source_group_weights.sum()
            if denom > 0:
                sigma_s[g] = (scattering_tables[:, g, :].T * source_group_weights).sum(axis=1) / denom
        # chi: fission-production weighted over the region
        fission_production = (
            material_tables["nu_sigma_f"][region_entries] * region_flux_entries.T
        ).sum(axis=1)
        production_weights = fission_production * region_entry_volumes
        region_spectra = material_tables["chi"][region_entries]
        chi = ((region_spectra.T * production_weights).sum(axis=1)
               / production_weights.sum()
               if production_weights.sum() > 0 else region_spectra[0])

        mats_out.append(Material(
            name=f"sph-region-{i}", diffusion=diffusion, sigma_a=sigma_a,
            nu_sigma_f=nu_sigma_f, sigma_s=sigma_s,
            chi=(chi if np.isclose(chi.sum(), 1.0) else np.array([1.0] + [0.0] * (G - 1))),
            total=sigma_t, kappa_fission=kappa_fission))
    return mats_out, region_flux, region_volume


def region_average(flux, region_map, mix_material=None, mix_weight=None,
                   mix_region_map=None, cell_volume=1.0):
    """Volume-average scalar flux per coarse region, (R, G).

    flux : (G, *shape) scalar flux; region_map : (*shape) region index.
    cell_volume : scalar or (*shape) cell volumes, as for
    :func:`flux_weighted_homogenize`; pass the SAME value used there. The
    default 1.0 is only right on a uniform grid: on an r-z grid use
    ``grid.cell_volumes`` (unit volumes overweight the axis annuli, e.g. a
    ratio of 1.38 against the reference region flux for one r-z region).
    Pass the same ``mix_material`` / ``mix_weight`` too, so the coarse solve's
    region fluxes are averaged over the *same* fractional volumes -- SPH
    compares these two directly, so an inconsistent weighting biases every
    factor.
    """
    f = asnumpy(flux)
    G = f.shape[0]
    cells, n, V, _ = _mixed_entries(region_map, region_map, cell_volume, f[0].size,
                                    mix_material, mix_weight, mix_region_map)
    ff = f.reshape(G, -1)[:, cells]
    R = int(max(n.max(), np.asarray(region_map).max())) + 1
    out = np.zeros((R, G))
    for i in range(R):
        m = n == i
        vol = V[m].sum()
        if vol > 0:
            out[i] = (ff[:, m] * V[m]).sum(axis=1) / vol
    return out


def _scale_material_get(mat, mu, nu):
    """SPH factor mu on the reaction cross sections, GET factor nu on D alone.

    Plain SPH ties leakage to reaction rates: it scales every cross section and
    D by mu, so one factor per region-group must serve both. That is
    exactly why it preserves absorption but not leakage against an S_N reference
    (measured: absorption 0.27%, leakage 37% at HP-MR drums withdrawn). Letting D
    carry its own factor nu adds the missing degree of freedom in the SPH idiom
    -- no discontinuity factors in the stencil, no operator change -- and the two
    conditions (region reaction rate, region leakage) then determine both.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    nu = np.atleast_1d(np.asarray(nu, dtype=float))
    return Material(
        name=f"{mat.name}-get", diffusion=np.asarray(mat.diffusion) * nu,
        sigma_a=np.asarray(mat.sigma_a) * mu,
        nu_sigma_f=np.asarray(mat.nu_sigma_f) * mu,
        sigma_s=np.asarray(mat.sigma_s) * mu[:, None],
        chi=np.asarray(mat.chi), total=np.asarray(mat.sigma_t) * mu,
        # kappa*Sigma_f is a macroscopic reaction-rate coefficient too. Losing
        # it made an SPH-corrected model silently fall back to nu*Sigma_f for
        # thermal coupling, changing the power shape by energy group.
        kappa_fission=(None if mat.kappa_fission is None else
                       np.asarray(mat.kappa_fission) * mu))


def sph_get_correct(homogenized_materials, region_map, reference_region_flux,
                    reference_region_leakage, solve, tol=1e-8,
                    max_iter=200, method="hybr") -> SphResult:
    """SPH + a generalized-equivalence leakage factor, solved together by JFNK.

    Unknowns per live region-group: log(mu) and log(nu). Conditions:

        mu * Phi   = phi_ref        (region reaction rate, plain SPH)
        L(mu, nu)  = L_ref          (region net leakage, the GET addition)

    solve : callable(materials, region_df) -> (region_flux (R,G), k_eff,
        region_leakage (R,G)). region_df is the (R,G) discontinuity factor the
        caller expands onto cells and passes to the solver as ``df=``. Leakage
        is the region's net current out, i.e. production/k - absorption.

    The second factor is a *discontinuity factor on the region surface*, not a
    scaling of the region's D. That distinction is the whole point: a region-D
    scaling moves leakage only through the global balance (measured: drum-only
    nu x1.10 gave +12.9 pcm against -129 pcm for a global scaling, i.e. it has
    almost no local authority) and leaves the Newton system rank-deficient,
    because per-region leakage is not independent of per-region reaction rate
    once global balance holds. A DF acts per *face*, asymmetrically, so it can
    redistribute leakage between the faces of one region -- which is exactly
    what classical GET is for, and why it must live in the stencil.

    Newton is used rather than a fixed point for the reason recorded in
    sph_correct: the mu iteration is unstable, not merely slow.
    """
    from scipy.optimize import newton_krylov, root
    try:
        from scipy.optimize import NoConvergence
    except ImportError:                      # pragma: no cover
        from scipy.optimize.nonlin import NoConvergence

    R = len(homogenized_materials)
    ref = np.asarray(reference_region_flux, dtype=float)
    lref = np.asarray(reference_region_leakage, dtype=float)
    shape = ref.shape
    live = ((ref > ref.max() * 1e-12)
            & (np.abs(lref) > np.abs(lref).max() * 1e-12)).reshape(-1)
    n = int(live.sum())
    state = {"k": float("nan"), "n": 0}

    def build(z):
        # Clip the exponents: Newton probes wide steps, and exp() underflowing
        # to 0 makes a zero total cross section (a hard Material error) rather
        # than a large residual it could back off from.
        # Separate bounds: mu is a cross-section scaling and may legitimately
        # be far from 1, but a discontinuity factor is a ratio of surface to
        # region-average flux and is physically O(1). Left unbounded Newton
        # probes f ~ 1e5, which wrecks the conditioning of the (now
        # non-symmetric) operator and GMRES fails outright.
        z = np.asarray(z, dtype=float)
        x = np.zeros(ref.size)
        y = np.zeros(ref.size)
        x[live] = np.clip(z[:n], -12.0, 12.0)
        y[live] = np.clip(z[n:], -1.2, 1.2)          # f in [0.30, 3.3]
        mu, nu = np.exp(x).reshape(shape), np.exp(y).reshape(shape)
        return mu, nu, [_scale_material(homogenized_materials[i], mu[i])
                        for i in range(R)]

    def residual(z):
        state["n"] += 1
        mu, nu, mats = build(z)
        phi, k, leak = solve(mats, nu)
        state["k"] = k
        phi = np.asarray(phi, float).reshape(-1)
        leak = np.asarray(leak, float).reshape(-1)
        r1 = np.full(ref.size, 0.0)
        r2 = np.full(ref.size, 0.0)
        ok = live & (phi > 0)
        r1[ok] = np.log(mu.reshape(-1)[ok] * phi[ok] / ref.reshape(-1)[ok])
        r1[live & ~ok] = 1e3
        sgn = live & (leak * lref.reshape(-1) > 0)
        r2[sgn] = np.log(np.abs(leak[sgn]) / np.abs(lref.reshape(-1)[sgn]))
        r2[live & ~sgn] = 1e3            # leakage flipped sign: reject
        return np.concatenate([r1[live], r2[live]])

    z0 = np.zeros(2 * n)
    converged = True
    if method == "hybr":
        # Dense-Jacobian trust region, as for sph_correct. The joint mu+DF
        # system is 2n ~ 24 unknowns -- far too small for Jacobian-free Krylov,
        # which failed here two different ways (rank-deficient Jacobian on one
        # drum state, a near-miss GMRES tolerance on the other). Forming the
        # 2n x 2n Jacobian costs 2n+1 coarse solves and lets MINPACK take
        # properly bounded steps.
        sol = root(residual, z0, method="hybr",
                   options=dict(xtol=max(tol * 1e-4, 1e-14)))
        z, converged = sol.x, bool(sol.success)
    else:
        try:
            z = newton_krylov(residual, z0, f_tol=tol, maxiter=max_iter,
                              method="lgmres")
        except NoConvergence as e:
            z = np.asarray(e.args[0])
            converged = False
    mu, nu, mats = build(z)
    _, k, _ = solve(mats, nu)
    return SphResult(corrected_materials=mats, factors=np.stack([mu, nu]),
                     k_eff=k, iterations=state["n"], converged=converged)


# How the diffusion coefficient is corrected alongside the cross sections.
#   "direct"   D~ = mu D   -- the exact transform phi = mu Phi of the diffusion
#                            equation: IGE-335 Eq. (3.43), Hebert 2015,
#                            Laboure et al. 2019 Eq. (7), Kumar et al. 2026 Eq. (2).
#   "inverse"  D~ = D / mu -- NON-STANDARD, kept only to reproduce results made
#                            before the 2026-11-13 SPH audit. It breaks the
#                            per-group scaling symmetry of the corrected operator
#                            (k then depends on the normalization: +830 to +980 pcm
#                            for a x1.3 rescaling of one group's factors on a
#                            reflective colorset), and on vacuum-bounded problems
#                            it acts as a hidden diffusion-length knob.
# See docs/qa/sph_audit_2026-11-13.md, findings F1 and F11.
D_SCALING = "direct"


def _scale_material(mat, mu, d_scaling=None):
    """SPH-corrected material: every cross section and D multiplied by mu.

    With the macroscopic flux Phi = phi / mu, constant in each region, every
    term of the diffusion equation that multiplies phi_g scales by mu_g:

        Sigma~_g = mu_g Sigma_g      (Sigma_a, nu*Sigma_f, kappa*Sigma_f, Sigma_t,
                                      and the scattering row of source group g)
        D~_g     = mu_g D_g          (IGE-335 Eq. 3.43; Laboure et al. 2019 Eq. 7)

    chi is not corrected. A common rescaling of one group's factors is then an
    exact symmetry of the corrected operator on a reflective problem, which is
    what makes the per-group normalization of :func:`sph_correct` a pure gauge.

    ``Sigma_t * mu`` is the even-parity total cross section. It is what a
    diffusion coarse solver needs; an SPN odd-parity equation would need
    ``Sigma_t / mu`` (IGE-335 Eq. 3.46), so corrected materials are for
    diffusion coarse solvers only (audit finding F6).

    ``d_scaling="inverse"`` (D~ = D / mu) is accepted only to reproduce
    pre-audit results; it is not an SPH correction in the sense of any
    reference and makes the converged eigenvalue depend on the normalization.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    d_scaling = D_SCALING if d_scaling is None else d_scaling
    if d_scaling not in ("inverse", "direct"):
        raise ValueError(f"d_scaling must be 'inverse' or 'direct', got {d_scaling!r}")
    dfac = mu if d_scaling == "direct" else 1.0 / mu
    return Material(
        name=f"{mat.name}-sph", diffusion=np.asarray(mat.diffusion) * dfac,
        sigma_a=np.asarray(mat.sigma_a) * mu,
        nu_sigma_f=np.asarray(mat.nu_sigma_f) * mu,
        sigma_s=np.asarray(mat.sigma_s) * mu[:, None],
        chi=np.asarray(mat.chi), total=np.asarray(mat.sigma_t) * mu,
        # Preserve the energy-release reaction rate for coupled power edits.
        kappa_fission=(None if mat.kappa_fission is None else
                       np.asarray(mat.kappa_fission) * mu),
        # higher scattering moments scale with their source group like P0
        sigma_s_moments=(None if mat.sigma_s_moments is None else
                         np.asarray(mat.sigma_s_moments) * mu[None, :, None]))


def apply_sph_factors(materials, factors):
    """SPH-corrected copies of ``materials``: material i gets the factors
    ``factors[i]`` (one per group) on every cross section and on D, as in
    :func:`sph_correct`. This is how tabulated factors are applied to
    tabulated cross sections of another state (e.g. interpolated in drum
    angle); ``SphResult.corrected_materials`` is the same operation at the
    generation state."""
    mu = np.asarray(factors, dtype=float)
    if mu.ndim != 2 or mu.shape[0] != len(materials):
        raise ValueError(f"factors must be ({len(materials)}, G), got {mu.shape}")
    return [_scale_material(m, mu[i]) for i, m in enumerate(materials)]


@dataclass
class SphResult:
    corrected_materials: list        # R Materials with SPH-corrected cross sections
    factors: np.ndarray              # (R, G) SPH factors mu
    k_eff: float                     # eigenvalue of the corrected coarse solve
    iterations: int                  # coarse solves (residual evaluations)
    converged: bool                  # residual <= tol (sph_correct); see each solver
    # Set only by sph_df_correct (GET): per-region discontinuity factors, per
    # vacuum-boundary-region boundary coefficients, and the least-squares cost
    # at the solution -- the diagnostic that says whether a nonzero residual is
    # a formulation failure or the per-cell (rather than per-face) DF limit.
    df: np.ndarray = None
    bcf: np.ndarray = None
    cost: float = float("nan")
    # Set by sph_correct. residual is the largest |log| mismatch of the SPH
    # conditions at the returned factors (and |log(k / k_ref)| for
    # normalization="none"): a measured quantity, not a solver flag. For
    # normalization="none" flux_scale is the constant s with mu * s * Phi =
    # phi_ref, i.e. the factor that turns the coarse eigenvector into the
    # reference-equivalent flux level.
    residual: float = float("nan")
    normalization: str = ""
    flux_scale: float = float("nan")


# Strength of the linear pull back to the start from an infeasible trial point.
# Only its sign and scale matter: the Newton step it induces is -(z - z0).
_INFEASIBLE_PULL = 10.0
# Bound on |log mu| inside the residual; exp(+-8) spans ~3e-4 .. 3e3.
_MAX_ABS_LOG_FACTOR = 8.0
# Lower bound on log gamma (sph_bcf_correct). The upper bound stays at
# _MAX_ABS_LOG_FACTOR: reaching it means the zero-flux leakage limit is below
# the reference and no solution exists. Small gamma is always reachable
# (gamma -> 0 is a reflective face) and is the right answer for a boundary zone
# that an absorber shadows: an HP-MR drum sector behind the B4C arc leaks
# ~1e-6 of a Be zone's thermal current.
_MIN_LOG_BCF = -30.0
# Largest fixed-k system solved with a dense Jacobian by default.
_DENSE_MAX_UNKNOWNS = 64
# Relative finite-difference step of the Jacobian-free Newton products. Each
# residual is a converged eigen-solve, so it carries the solver's noise; a
# step well above that noise (scipy's default is ~1.5e-8) keeps the products
# meaningful. Solve the coarse problem at least ~1e3 times tighter than this.
_JFNK_RDIFF = 1e-6


def _sph_live_set(reference_region_flux, sph_regions):
    """Flat mask of the region-group entries whose factor is solved for.

    Entries outside ``sph_regions`` and entries without reference flux (void,
    a decoupled region) keep mu = 1 and are excluded from the residual and the
    normalization. A non-void entry with a negative reference flux (an SP3
    phi0 = Phi1 - 2 phi2 artifact, typically in a reflector) is frozen too, but
    not silently.
    """
    ref = np.asarray(reference_region_flux, dtype=float)
    R, G = ref.shape
    if sph_regions is None:
        mask = np.ones((R, G), dtype=bool)
    else:
        m = np.asarray(sph_regions, dtype=bool)
        if m.shape == (R,):
            mask = np.repeat(m[:, None], G, axis=1)
        elif m.shape == (R, G):
            mask = m.copy()
        else:
            raise ValueError(f"sph_regions must have shape ({R},) or ({R}, {G}), "
                             f"got {m.shape}")
    floor = np.abs(ref).max() * 1e-12
    negative = mask & (ref < -floor)
    if negative.any():
        bad = sorted({int(i) for i in np.argwhere(negative)[:, 0]})
        warnings.warn(f"SPH regions {bad} have a negative reference flux in some "
                      "group; those factors are frozen at mu = 1. Exclude the "
                      "regions with sph_regions (e.g. correct the fuel only) or use "
                      "a reference without negative scalar flux.", RuntimeWarning,
                      stacklevel=3)
    return (mask & (ref > floor)).reshape(-1)


class _SphProblem:
    """Bookkeeping shared by the SPH solution methods.

    Every residual evaluation is one coarse solve. The best evaluated point
    (smallest max-norm residual) is kept with its eigenvalue, so a method that
    stops early or wanders still returns the best factors it actually solved,
    paired with their own k.
    """

    def __init__(self, homogenized_materials, ref, live, solve, verbose):
        self.hm = list(homogenized_materials)
        self.ref = ref
        self.shape = ref.shape
        self.live = live
        self.solve = solve
        self.verbose = verbose
        self.nfev = 0
        self.t0 = time.perf_counter()
        self.best = {"res": np.inf, "x": np.zeros(int(live.sum())), "k": float("nan"),
                     "s": float("nan")}

    def factors(self, x_live):
        x = np.zeros(self.live.size)
        x[self.live] = x_live
        return np.exp(x).reshape(self.shape)

    def materials(self, mu):
        return [_scale_material(self.hm[i], mu[i]) for i in range(len(self.hm))]

    def evaluate(self, x_live):
        """Coarse solve at mu = exp(x); returns (live region flux, k) or None
        if the trial point made the coarse flux non-positive or non-finite."""
        self.nfev += 1
        flux, k = self.solve(self.materials(self.factors(x_live)))
        phi = np.asarray(flux, dtype=float).reshape(-1)[self.live]
        if not (np.isfinite(k) and np.all(np.isfinite(phi)) and np.all(phi > 0.0)):
            return None
        return phi, float(k)

    def record(self, x_live, k, rows, s=float("nan")):
        res = float(np.abs(rows).max()) if rows.size else 0.0
        if self.verbose:
            print(f"    [sph] eval {self.nfev:4d}  max|r| = {res:.3e}  k = {k:.8f}"
                  f"  t = {time.perf_counter() - self.t0:.1f} s", flush=True)
        if res < self.best["res"]:
            self.best = {"res": res, "x": np.array(x_live, copy=True), "k": k, "s": s}


def _group_log_normalizer(live, R, G, region_volume):
    """log(values / N_g) on the live entries, N_g = sum_i V_i values_ig over the
    live entries of group g: the per-group flux-volume normalization."""
    group = np.tile(np.arange(G), R)
    weight = np.repeat(np.asarray(region_volume, dtype=float), G)
    sets = [live & (group == g) for g in range(G)]

    def log_normalized(values_live):
        values = np.zeros(live.size)
        values[live] = values_live
        out = np.zeros(live.size)
        for sel in sets:
            if sel.any():
                out[sel] = np.log(values[sel] / (weight[sel] * values[sel]).sum())
        return out[live]
    return log_normalized


def sph_correct(homogenized_materials, region_map, reference_region_flux, solve, *,
                region_volume=None, normalization="flux-volume", k_ref=None,
                sph_regions=None, method=None, tol=1e-8, max_iter=200,
                relax=1.0, depth=5, mu0=None, verbose=False) -> SphResult:
    """Solve for the SPH factors that make a coarse solve reproduce the reference.

    The factor mu_{i,g} multiplies every cross section and the diffusion
    coefficient of region i, group g (:func:`_scale_material`). The defining
    condition is ``mu_{i,g} Phi_{i,g} = phi_ref_{i,g}``: the corrected coarse
    reaction rates Sigma~ Phi V equal the reference rates Sigma phi_ref V.
    A coarse eigenvector is defined only up to a constant, so the condition
    needs one more statement about the flux level. Two are offered:

    normalization="flux-volume" (default; reflective generation geometries)
        One constant per group: the macroscopic flux is scaled so that
        ``sum_i V_i Phi_{i,g} = sum_i V_i phi_ref_{i,g}`` over the corrected
        entries of group g, then ``mu = phi_ref / Phi``. This is the standard
        normalization (IGE-335 Sec. 3.11, keyword STD; IGE-344 Eq. 7.7). On a
        reflective problem the factors are fixed only up to one constant per
        group, and under D~ = mu D that constant is an exact gauge: k does not
        depend on it.
        With a vacuum boundary the normalization is no longer a gauge and k
        is not reproduced exactly (Laboure et al. 2019, Sec. 3.1); use "none".
        Requires ``region_volume``.

    normalization="none" (vacuum-bounded generation geometries; Griffin)
        No normalization; the eigenvalue is fixed at the reference instead
        (Laboure et al. 2019, Sec. 3.2; Kumar et al. 2026). Unknowns are
        log mu on the corrected entries and log s, the level of the coarse
        eigenvector; conditions are ``mu s Phi = phi_ref`` entry by entry and
        ``k(mu) = k_ref``. The system is square, needs ``k_ref``, and is
        singular on a purely reflective problem (use "flux-volume" there).

        A solution need not exist. With D~ = mu D a region-uniform mu leaves
        the flux shape inside a region unchanged; at a vacuum face it only
        moves the effective Marshak coefficient between alpha/mu -> 0
        (reflective) and infinity (zero flux). If a corrected region at the
        vacuum boundary mixes materials whose flux shape the coarse model
        cannot represent, the required leakage may lie outside that range.
        The same happens when coarse nodes are too large to represent a
        reflector's thermal flux (measured: 7.5 cm finite-volume nodes against
        a 2.85 cm water diffusion length). The solver then returns its
        least-squares floor with ``converged=False`` (measured: k still 760 to
        1060 pcm off on the unaligned three-region colorset of
        tests/verification/test_sph.py). Griffin avoids this by correcting the
        fuel only and meshing reflector and drums explicitly: pass
        ``sph_regions``, and leave uncorrected at least the regions that carry
        the vacuum boundary (on HP-MR, correcting the outer reflector moves
        every fast factor to ~0.26 to absorb a 10 pcm mismatch).

    homogenized_materials : R Materials (e.g. from
        :func:`flux_weighted_homogenize`), indexed by region.
    region_map            : coarse region map (kept for API symmetry; unused).
    reference_region_flux : (R, G) reference region fluxes (second return of
        :func:`flux_weighted_homogenize`).
    solve                 : callable(materials) -> (region_flux (R, G), k_eff),
        running the coarse diffusion eigen-solve with the given per-region
        materials. The caller wires up grid, geometry and boundary conditions;
        the coarse mesh may differ from the reference mesh, provided
        ``region_flux`` is averaged with the coarse mesh's own region map and
        cell volumes (:func:`region_average`).
    region_volume         : (R,) region volumes (third return of
        :func:`flux_weighted_homogenize`). Required for "flux-volume".
    k_ref                 : reference eigenvalue. Required for "none".
    sph_regions           : optional (R,) or (R, G) boolean mask of the
        entries to correct; the rest keep mu = 1 and are excluded from the
        conditions and the normalization. Griffin corrects the fuel only.
    method                : "anderson" (default for "flux-volume"): the fixed
        point log mu <- log(phi_ref / Phi_normalized), Anderson-accelerated over
        ``depth`` iterates (depth=1 with ``relax`` gives the damped Picard
        iteration). "hybr" and "lm": dense-Jacobian root finders (MINPACK),
        one coarse solve per unknown and Jacobian, for small systems.
        "jfnk": Jacobian-free Newton-Krylov, for many unknowns. For "none",
        "anderson" is an Anderson-accelerated fixed point of the classic SPH
        update plus a common shift that drives k to k_ref, one coarse solve
        per iteration, falling back to "jfnk" if it does not converge. The
        default for "none" is "hybr" up to 64 unknowns and "anderson" above
        (e.g. 30 assembly zones x 11 groups). For "none", "hybr" falls back to a
        Levenberg-Marquardt least-squares solve followed by a second "hybr"
        from its endpoint when the first attempt does not reach ``tol``.

        The plain fixed point is unstable whenever the coarse flux in a
        strongly absorbing region falls below the reference: the condition
        demands mu > 1, which raises that region's absorption, depresses Phi
        further and demands a larger mu again (measured on the HP-MR drum arc:
        mu past 10^3). Newton sees that coupling through the Jacobian; this is
        the PJFNK-SPH approach of Ortensi et al. used in Griffin.
    tol                   : tolerance on the residual, max |log| mismatch of
        the conditions (a relative error). It must be above the accuracy of
        the coarse solve. ``converged`` is ``residual <= tol``, measured at the
        returned factors whatever the method reported.
    mu0                   : optional (R, G) initial factors.

    Returns an :class:`SphResult` whose ``corrected_materials`` and ``k_eff``
    belong to the same coarse solve.
    """
    if normalization not in ("flux-volume", "none"):
        raise ValueError("normalization must be 'flux-volume' or 'none', got "
                         f"{normalization!r}")
    ref = np.asarray(reference_region_flux, dtype=float)
    R, G = ref.shape
    if len(homogenized_materials) != R:
        raise ValueError(f"{len(homogenized_materials)} materials for {R} regions")
    live = _sph_live_set(ref, sph_regions)
    prob = _SphProblem(homogenized_materials, ref, live, solve, verbose)
    ln_ref = np.log(ref.reshape(-1)[live])
    x0 = np.zeros(int(live.sum()))
    if mu0 is not None:
        m0 = np.asarray(mu0, dtype=float).reshape(-1)[live]
        if np.any(m0 <= 0.0) or not np.all(np.isfinite(m0)):
            raise ValueError("mu0 must be finite and positive on the corrected entries")
        x0 = np.log(m0)

    def finish(x_live, k, residual, s=float("nan")):
        mu = prob.factors(x_live)
        return SphResult(corrected_materials=prob.materials(mu), factors=mu,
                         k_eff=k, iterations=prob.nfev,
                         converged=bool(residual <= tol), residual=residual,
                         normalization=normalization, flux_scale=s)

    if not live.any():
        flux_k = prob.evaluate(x0)
        k = flux_k[1] if flux_k is not None else float("nan")
        return finish(x0, k, 0.0)

    if normalization == "flux-volume":
        if region_volume is None:
            raise ValueError(
                "normalization='flux-volume' needs region_volume (the third return "
                "of flux_weighted_homogenize): the normalization is one flux-VOLUME "
                "integral per group")
        volume = np.asarray(region_volume, dtype=float).reshape(-1)
        if volume.shape != (R,) or np.any(volume <= 0.0):
            raise ValueError(f"region_volume must be ({R},) and positive")
        log_normalized = _group_log_normalizer(live, R, G, volume)
        ln_ref_n = log_normalized(ref.reshape(-1)[live])
        method = "anderson" if method is None else method

        def residual(x_live):
            x_live = np.clip(np.asarray(x_live, dtype=float),
                             -_MAX_ABS_LOG_FACTOR, _MAX_ABS_LOG_FACTOR)
            out = prob.evaluate(x_live)
            if out is None:
                return _INFEASIBLE_PULL * (x_live - x0)
            phi, k = out
            r = ln_ref_n - log_normalized(phi) - x_live
            prob.record(x_live, k, r)
            return r

        if method == "anderson":
            X, F = [], []
            x = x0
            for _ in range(max_iter):
                x = np.clip(x, -_MAX_ABS_LOG_FACTOR, _MAX_ABS_LOG_FACTOR)
                out = prob.evaluate(x)
                if out is None:
                    break                   # no way to continue a fixed point
                phi, k = out
                f = ln_ref_n - log_normalized(phi) - x
                prob.record(x, k, f)
                if np.abs(f).max() <= tol:
                    break
                X.append(x)
                F.append(f)
                if len(F) > depth:
                    X.pop(0)
                    F.pop(0)
                x = _anderson_step(X, F, relax)
        elif method in ("hybr", "lm"):
            from scipy.optimize import root
            # MINPACK's xtol is a tolerance on the step, not the residual; keep it
            # well below tol so it does not stop early (measured: 41 pcm where
            # JFNK reached 14 pcm when tol was passed straight through).
            root(residual, x0, method=method,
                 options=dict(xtol=max(tol * 1e-4, 1e-14)))
        elif method == "jfnk":
            _guarded_krylov(residual, x0, tol, max_iter, verbose)
        else:
            raise ValueError("method must be 'anderson', 'hybr', 'lm' or 'jfnk', "
                             f"got {method!r}")
        b = prob.best
        return finish(b["x"], b["k"], b["res"])

    # normalization == "none": fixed k, unknowns (log mu on live entries, log s)
    if k_ref is None or not np.isfinite(k_ref) or k_ref <= 0.0:
        raise ValueError("normalization='none' needs the reference eigenvalue k_ref")
    # Dense Jacobians cost one coarse solve per unknown: fine for a few dozen
    # (one zone per material), prohibitive for 30 assemblies x 11 groups.
    if method is None:
        method = "hybr" if int(live.sum()) + 1 <= _DENSE_MAX_UNKNOWNS else "anderson"
    if method not in ("anderson", "hybr", "lm", "jfnk"):
        raise ValueError("method for normalization='none' must be 'anderson', "
                         f"'hybr', 'lm' or 'jfnk', got {method!r}")
    start = prob.evaluate(x0)
    if start is None:
        raise RuntimeError("the coarse solve at the initial factors has a "
                           "non-positive region flux")
    phi0, k0 = start
    z0 = np.append(x0, np.log(np.exp(ln_ref).sum() / (np.exp(x0) * phi0).sum()))
    prob.record(x0, k0, np.append(x0 + z0[-1] + np.log(phi0) - ln_ref,
                                  np.log(k0 / k_ref)), s=float(np.exp(z0[-1])))

    def residual_k(z):
        z = np.asarray(z, dtype=float)
        x_live = np.clip(z[:-1], -_MAX_ABS_LOG_FACTOR, _MAX_ABS_LOG_FACTOR)
        # Where no solution exists the factors run to zero and the level s to
        # infinity; bound log s so exp() stays finite.
        ln_s = float(np.clip(z[-1], -700.0, 700.0))
        out = prob.evaluate(x_live)
        if out is None:
            return _INFEASIBLE_PULL * (z - z0)
        phi, k = out
        r = np.append(x_live + ln_s + np.log(phi) - ln_ref, np.log(k / k_ref))
        prob.record(x_live, k, r, s=float(np.exp(ln_s)))
        return r

    def best_z():
        b = prob.best
        return np.append(b["x"], np.log(b["s"])) if np.isfinite(b["s"]) else z0

    from scipy.optimize import least_squares, root
    xtol = max(tol * 1e-4, 1e-14)
    if method == "anderson":
        _fixed_k_anderson(prob, x0, phi0, k0, ln_ref, k_ref, tol, max_iter, depth,
                          relax)
        if prob.best["res"] > tol:
            # Diverged or stalled (e.g. a strong absorber among the corrected
            # zones, where the Picard map is unstable): Newton from the best
            # point, which sees the coupling through its Jacobian.
            method = "jfnk"
            z0 = best_z()
    if method == "jfnk":
        _guarded_krylov(residual_k, z0, tol, max_iter, verbose)
    else:
        if method == "hybr":
            root(residual_k, z0, method="hybr", options=dict(xtol=xtol))
        if prob.best["res"] > tol:
            # Least squares from the best point so far: robust where Powell's
            # hybrid wanders into infeasible factors, and it returns the floor
            # when no exact solution exists. LM alone can stall short of an
            # existing root (measured: 9e-6 after its evaluation budget, where
            # a hybr restart from its endpoint reached 6e-15), hence the polish.
            n = z0.size
            least_squares(residual_k, best_z(), method="lm", xtol=xtol,
                          ftol=xtol, gtol=xtol, max_nfev=max(50, 10 * (n + 1)))
            if prob.best["res"] > tol:
                root(residual_k, best_z(), method="hybr", options=dict(xtol=xtol))
    b = prob.best
    return finish(b["x"], b["k"], b["res"], s=b["s"])


def _guarded_krylov(residual, z0, tol, max_iter, verbose):
    """Jacobian-free Newton-Krylov whose failure is not fatal.

    Every residual evaluation is recorded by _SphProblem, which keeps the best
    point; the caller returns that point with ``converged`` measured, so a
    Krylov breakdown (scipy raises ValueError "Jacobian inversion yielded zero
    vector" when the finite-difference products vanish, measured on HP-MR
    fixed-k SPH with the reflector corrected) or a linear-algebra failure ends
    the solve instead of the program.
    """
    from scipy.optimize import root
    try:
        root(residual, z0, method="krylov",
             options=dict(fatol=tol, maxiter=max_iter, disp=bool(verbose),
                          jac_options=dict(rdiff=_JFNK_RDIFF)))
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as err:
        if verbose:
            print(f"    [sph] JFNK stopped: {err}", flush=True)


def _fixed_k_anderson(prob, x0, phi0, k0, ln_ref, k_ref, tol, max_iter, depth,
                      relax):
    """Anderson-accelerated fixed point for the fixed-k SPH problem.

    Each iteration is one coarse eigen-solve. The map combines

    * the classic SPH update of the shape, log mu <- log(phi_ref / (s Phi)),
      with the level s of the coarse eigenvector set by least squares
      (the mean log mismatch is zero), and
    * a common shift of every corrected log mu that drives k to k_ref:
      -log(k / k_ref) / S, with S = d log k / d(common shift) measured by one
      extra solve at the start and refreshed by secant along the iteration.

    At a fixed point both corrections vanish, which is exactly mu s Phi =
    phi_ref and k = k_ref. The shape update is the Picard iteration that
    diverges for a strongly absorbing zone (see :func:`sph_correct`);
    the caller falls back to JFNK when this does not reach ``tol``. On fuel
    zones, whose factors stay near 1, it converges in a few dozen solves
    instead of the hundreds Newton-Krylov needs.
    """
    def residual_rows(x, phi, k):
        mismatch = ln_ref - x - np.log(phi)
        ln_s = float(mismatch.mean())
        return np.append(-mismatch + ln_s, np.log(k / k_ref)), ln_s

    x, phi, k = np.asarray(x0, dtype=float), phi0, k0
    # sensitivity of log k to a common shift of the corrected log factors
    h = 1e-3
    probe = prob.evaluate(np.clip(x + h, -_MAX_ABS_LOG_FACTOR, _MAX_ABS_LOG_FACTOR))
    if probe is None:
        return
    sens = (np.log(probe[1]) - np.log(k)) / h
    if not np.isfinite(sens) or abs(sens) < 1e-8:
        return                                 # k does not respond: no k lever
    X, F = [], []
    last = (float(x.mean()), np.log(k))
    for _ in range(max_iter):
        rows, ln_s = residual_rows(x, phi, k)
        prob.record(x, k, rows, s=float(np.exp(ln_s)))
        if np.abs(rows).max() <= tol:
            return
        target = x - rows[:-1] - rows[-1] / sens       # shape + level updates
        f = target - x
        X.append(x)
        F.append(f)
        if len(F) > depth:
            X.pop(0)
            F.pop(0)
        x_new = np.clip(_anderson_step(X, F, relax), -_MAX_ABS_LOG_FACTOR,
                        _MAX_ABS_LOG_FACTOR)
        out = prob.evaluate(x_new)
        if out is None:
            return
        phi_new, k_new = out
        # secant refresh of the level sensitivity, when the mean moved enough
        dmean = float(x_new.mean()) - last[0]
        if abs(dmean) > 1e-6:
            s_new = (np.log(k_new) - last[1]) / dmean
            if np.isfinite(s_new) and s_new * sens > 0.0 and 0.2 < s_new / sens < 5.0:
                sens = s_new
        last = (float(x_new.mean()), np.log(k_new))
        x, phi, k = x_new, phi_new, k_new
    rows, ln_s = residual_rows(x, phi, k)
    prob.record(x, k, rows, s=float(np.exp(ln_s)))


def sph_bcf_correct(homogenized_materials, reference_region_flux,
                    reference_boundary_current, solve, *, region_volume,
                    sph_regions=None, tol=1e-7, max_iter=300, depth=5, relax=1.0,
                    verbose=False) -> SphResult:
    """SPH factors plus boundary discontinuity factors, flux-volume normalized.

    The hybrid SPH-DF method of Laboure et al. (Ann. Nucl. Energy 128 (2019)
    443, Sec. 4.2): SPH factors mu on the regions and a discontinuity factor
    gamma (the "boundary coefficient", Eq. 36) on the vacuum boundary of each
    region that has one, multiplying the Marshak albedo there. Conditions per
    group g, with c_g the flux-volume normalization constant over the corrected
    entries:

        mu  c_g Phi        = phi_ref       (region reaction rates)
        c_g L(mu, gamma)   = J_out,ref     (outgoing current through each
                                            region's vacuum boundary)

    Why the normalization is now legitimate: if every region is corrected and
    every vacuum face carries gamma, the corrected problem is invariant under
    Phi_g -> c Phi_g, mu_g -> mu_g / c, gamma_g -> gamma_g / c (the Robin term
    8 D a / (h (h a + 2 sqrt3 D)) is homogeneous of degree one in (D, a)), so
    the per-group normalization only fixes that gauge and k is reproduced.
    Plain flux-volume SPH with a vacuum boundary has no such gauge, which is
    the k error Laboure et al. trace to the normalization (Sec. 3.2). Leaving
    regions uncorrected (``sph_regions``) breaks the gauge again.

    reference_boundary_current : (B, G) outgoing current through the vacuum
        boundary of each of the B boundary regions, in the normalization of
        ``reference_region_flux`` (TriSNTransportSolver.region_partial_currents).
    solve : callable(materials, gamma (B, G)) -> (region_flux (R, G), k_eff,
        boundary_leakage (B, G)); the caller expands gamma onto the boundary
        cells of each region as the solver's ``bcf`` and returns the coarse
        leakage of the same regions (garnet.tri.tri_boundary_leakage).

    Solved as an Anderson-accelerated fixed point: mu <- phi_ref / (c Phi),
    gamma <- gamma J_ref / (c L), one coarse solve per iteration. Returns an
    SphResult with ``factors`` = mu, ``bcf`` = gamma, ``residual`` = the
    largest |log| mismatch of the conditions. mu stays within exp(+-8);
    gamma within exp(-30) .. exp(8): a gamma at its upper bound with the
    leakage still short marks an infeasible condition, while gamma of 1e-6
    and below is legitimate for a boundary zone shadowed by an absorber.
    """
    ref = np.asarray(reference_region_flux, dtype=float)
    R, G = ref.shape
    if len(homogenized_materials) != R:
        raise ValueError(f"{len(homogenized_materials)} materials for {R} regions")
    V = np.asarray(region_volume, dtype=float).reshape(R)
    Jref = np.asarray(reference_boundary_current, dtype=float)
    if Jref.ndim != 2 or Jref.shape[1] != G:
        raise ValueError(f"reference_boundary_current must be (B, {G}), got {Jref.shape}")
    B = Jref.shape[0]
    live = _sph_live_set(ref, sph_regions)
    live2 = live.reshape(R, G)
    blive = (Jref > np.abs(Jref).max() * 1e-12).reshape(-1)
    nl, nb = int(live.sum()), int(blive.sum())
    ln_ref = np.log(ref.reshape(-1)[live])
    ln_J = np.log(Jref.reshape(-1)[blive])
    ref_norm = (V[:, None] * ref * live2).sum(axis=0)              # (G,)
    state = {"n": 0, "best": None}
    lo = np.concatenate([np.full(nl, -_MAX_ABS_LOG_FACTOR), np.full(nb, _MIN_LOG_BCF)])
    hi = np.full(nl + nb, _MAX_ABS_LOG_FACTOR)

    def evaluate(x):
        x = np.clip(x, lo, hi)
        mu = np.ones(R * G)
        mu[live] = np.exp(x[:nl])
        mu = mu.reshape(R, G)
        gam = np.ones(B * G)
        gam[blive] = np.exp(x[nl:])
        gam = gam.reshape(B, G)
        mats = [_scale_material(homogenized_materials[i], mu[i]) for i in range(R)]
        phi, k, leak = solve(mats, gam)
        phi = np.asarray(phi, dtype=float).reshape(R, G)
        leak = np.asarray(leak, dtype=float).reshape(B, G)
        state["n"] += 1
        c = ref_norm / np.maximum((V[:, None] * phi * live2).sum(axis=0), 1e-300)
        cphi = (c[None, :] * phi).reshape(-1)[live]
        cl = (c[None, :] * leak).reshape(-1)[blive]
        if np.any(cphi <= 0) or np.any(cl <= 0):
            return None
        f = np.concatenate([ln_ref - np.log(cphi) - x[:nl], ln_J - np.log(cl)])
        res = float(np.abs(f).max()) if f.size else 0.0
        if state["best"] is None or res < state["best"][0]:
            state["best"] = (res, x.copy(), mats, mu, gam, k)
        if verbose:
            i = int(np.argmax(np.abs(f))) if f.size else 0
            if i < nl:
                r, g = divmod(int(np.flatnonzero(live)[i]), G)
                worst = f"mu[{r},{g}]={mu[r, g]:.3g}"
            else:
                b, g = divmod(int(np.flatnonzero(blive)[i - nl]), G)
                worst = (f"gamma[{b},{g}]={gam[b, g]:.3g} (c L / J_ref = "
                         f"{float(np.exp(-f[i])):.3f})")
            print(f"    [sph-bcf] eval {state['n']:4d}  max|r| = {res:.3e}  k = {k:.8f}"
                  f"  worst {worst}  gamma {gam.min():.3g}..{gam.max():.3g}", flush=True)
        return f, res

    x = np.zeros(nl + nb)
    X, F = [], []
    for _ in range(max_iter):
        out = evaluate(x)
        if out is None:
            break
        f, res = out
        if res <= tol:
            break
        X.append(x.copy())
        F.append(f)
        if len(F) > depth:
            X.pop(0)
            F.pop(0)
        x = np.clip(_anderson_step(X, F, relax), lo, hi)
    res, x, mats, mu, gam, k = state["best"]
    return SphResult(corrected_materials=mats, factors=mu, k_eff=k,
                     iterations=state["n"], converged=bool(res <= tol), bcf=gam,
                     residual=res, normalization="flux-volume")


def sph_kinetics(kinetics, factors):
    """Kinetics data for a transient with SPH-corrected materials.

    Under Phi = phi / mu every term of the time-dependent diffusion equation
    that multiplies phi_g scales by mu_g, including the time derivative: the
    corrected model needs ``1 / v~_g = mu_g / v_g``, i.e. per-material speeds
    ``v_g / mu_{i,g}``. Delayed-neutron data need no change (beta multiplies
    the corrected nu*Sigma_f, which already carries mu).

    Without this the SPH model's generation time is biased (measured on a
    reflective colorset: +20.7 % with v, +0.2 % with v/mu; audit finding F7).
    Griffin uses the uncorrected 1/v (Takasugi et al. 2023, Eq. 8).

    kinetics : :class:`~garnet.materials.Kinetics` with (G,) or (R, G) speeds.
    factors  : (R, G) SPH factors, one row per entry of the coarse model's
               materials list (``SphResult.factors``).

    Returns a new Kinetics with (R, G) velocities. Per-material velocities are
    supported by :class:`~garnet.transient.TransientSolver` (diffusion).
    """
    from .materials import Kinetics

    mu = np.asarray(factors, dtype=float)
    if mu.ndim != 2 or np.any(mu <= 0.0) or not np.all(np.isfinite(mu)):
        raise ValueError("factors must be a finite, positive (R, G) array")
    v = np.asarray(kinetics.velocities, dtype=float)
    if v.ndim == 1:
        if v.shape[0] != mu.shape[1]:
            raise ValueError(f"{v.shape[0]} velocities for {mu.shape[1]} groups")
        v = np.broadcast_to(v, mu.shape)
    elif v.shape != mu.shape:
        raise ValueError(f"velocities {v.shape} do not match factors {mu.shape}")
    return Kinetics(velocities=v / mu, beta=kinetics.beta, decay=kinetics.decay,
                    chi_delayed=kinetics.chi_delayed)


def sph_correct_monolithic(homogenized_materials, region_map,
                           reference_region_flux, k_ref, balance_residual,
                           region_avg, flux0, tol=1e-8, max_iter=100,
                           precondition=None, verbose=False) -> SphResult:
    """Monolithic PJFNK-SPH: solve for flux and SPH factors in one Newton system.

    :func:`sph_correct` treats the coarse solve as a black box, so every residual
    evaluation costs a full eigenvalue solve. Ortensi et al. instead recast the
    problem as a *source-free steady state* in which flux and SPH factors are one
    nonlinear vector, updating the factors inside each linear iteration; that is
    what buys their reported 5x (diffusion) to 10-15x (transport) speedup over
    the Picard approach.

    The recast: fix the eigenvalue at the *reference* k and let the factors make
    the coarse operator critical there. With mu unknown, the unknowns and
    equations balance exactly:

        R1(phi, mu) = [A(mu) - F(mu)/k_ref] phi          (per cell, per group)
        R2(phi, mu) = log(mu_i,g * Phi_i,g / phi_ref_i,g) (per region, per group)

    R1 alone is homogeneous in phi and would admit phi = 0; R2 pins both the
    scale and the factors, which is exactly the non-linearity the paper notes
    "allows convergence to a non-zero solution". No inner eigensolve appears.

    balance_residual : callable(materials, flux) -> array, R1 for the given
        SPH-corrected materials and flux (the caller owns geometry/BCs).
    region_avg       : callable(flux) -> (R, G) region-average flux, consistent
        with the volume split used to build reference_region_flux.
    flux0            : initial flux guess, shape (G, *cells).
    precondition     : callable(flux_block) -> approximately J^-1 applied to the
        flux block (the natural choice is the diffusion inverse at the current
        factors). This is the "P" of PJFNK and it is not optional at scale: with
        ~10^4 flux unknowns the inner Krylov stalls without it and the factors
        never leave mu = 1, whereas the black-box solve (a dozen unknowns) needs
        no preconditioner at all.
    k_ref            : the reference eigenvalue the corrected coarse operator
        must reproduce.
    """
    from scipy.optimize import newton_krylov
    try:                      # moved to the public API in newer scipy
        from scipy.optimize import NoConvergence
    except ImportError:       # pragma: no cover
        from scipy.optimize.nonlin import NoConvergence

    R = len(homogenized_materials)
    ref = np.asarray(reference_region_flux, dtype=float)
    live = (ref > ref.max() * 1e-12).reshape(-1)
    flux0 = np.asarray(asnumpy(flux0), dtype=float)
    fshape = flux0.shape
    nflux = flux0.size
    scale = float(np.abs(flux0).max()) or 1.0
    nfev = [0]

    def unpack(v):
        phi = v[:nflux].reshape(fshape) * scale
        x = np.zeros(ref.size)
        x[live] = v[nflux:]
        return phi, np.exp(x).reshape(ref.shape)

    def residual(v):
        nfev[0] += 1
        phi, mu = unpack(v)
        mats = [_scale_material(homogenized_materials[i], mu[i])
                for i in range(R)]
        r1 = np.asarray(balance_residual(mats, phi), dtype=float).ravel() / scale
        Phi = np.asarray(region_avg(phi), dtype=float).reshape(-1)
        r2 = np.zeros(ref.size)
        good = live & (Phi > 0)
        r2[good] = np.log(mu.reshape(-1)[good] * Phi[good] / ref.reshape(-1)[good])
        r2[live & ~good] = 1e3          # reject factors that kill a region
        return np.concatenate([r1, r2[live]])

    v0 = np.concatenate([flux0.ravel() / scale, np.zeros(int(live.sum()))])
    inner_M = None
    if precondition is not None:
        from scipy.sparse.linalg import LinearOperator

        def _apply(v):
            out = np.empty_like(v)
            out[:nflux] = np.asarray(
                precondition(v[:nflux].reshape(fshape)), dtype=float).ravel()
            out[nflux:] = v[nflux:]          # factors: identity block
            return out

        inner_M = LinearOperator((v0.size, v0.size), matvec=_apply, dtype=float)
    # Capture the running iterate: on NoConvergence scipy may hand back the
    # starting point, which would silently report mu = 1 as "the answer".
    best = {"x": v0.copy(), "f": np.inf}

    def _cb(x, fx):
        n = float(np.linalg.norm(np.atleast_1d(fx)))
        if n < best["f"]:
            best["f"], best["x"] = n, np.array(x, copy=True)

    converged = True
    try:
        v = newton_krylov(residual, v0, f_tol=tol, maxiter=max_iter,
                          method="lgmres", inner_M=inner_M, callback=_cb,
                          verbose=verbose)
    except NoConvergence:
        v = best["x"]
        converged = False
    phi, mu = unpack(v)
    corrected = [_scale_material(homogenized_materials[i], mu[i])
                 for i in range(R)]
    return SphResult(corrected_materials=corrected, factors=mu, k_eff=k_ref,
                     iterations=nfev[0], converged=converged)


def sph_df_correct(homogenized_materials, region_map, reference_region_flux,
                   reference_interfaces, reference_boundaries, solve,
                   tol=1e-8, max_iter=200, method="lm", bounds=1.2,
                   df_pairs=None, verbose=False):
    """SPH + discontinuity factors + boundary coefficients, solved jointly.

    SHELVED (July 2026) -- correct and verified, but not the supported path.
    On HP-MR it does not earn its cost: plain diffusion already reproduces the
    control-drum WORTH to -20 pcm (its ~800 pcm absolute error at each drum state
    cancels in the difference), better than SPH (-52) and better than every DF
    variant here (bcf-only +63, near-boundary -261, full -259). SPH's value is
    absolute k (994 -> 41 pcm), so quote that, not worth. Use plain
    :func:`sph_correct`; this is kept for future work, not deleted.

    If picked up again, the state is:
      * per-SURFACE DF works (see garnet.tri.face_df_from_pairs); per-region does
        not -- it is over-determined and pegs at the bounds.
      * df_pairs=[] (bcf-only) is the best and cheapest variant AND keeps the
        operator symmetric, so CG stays valid. Start there.
      * the interface variants stall at cost ~5e-2 while bcf-only reaches ~5e-5.
        That is NOT a DoF limit. The coarse partial current is reconstructed from
        diffusion's P1 closure (J_out = phi_s/4 + J/2); asking it to match an S_N
        half-range integral beside a near-black absorber demands something the
        diffusion model structurally cannot produce. Progress needs a better
        coarse partial-current reconstruction (or an SP3/SN coarse operator),
        not more factors.
      * diagnostic that found every convention bug: at unit factors the
        coarse/reference ratio must be ONE constant across all rows. Distinct
        clusters = convention bug; spread within a cluster = the physics DF is
        meant to fix.

    Equivalence conditions, following Laboure et al. (PHYSOR 2018 / Ann. Nucl.
    Energy 2019) rather than the region net-leakage condition tried earlier:

        mu_m,g  : region reaction rate      mu Phi = phi_ref
        DF      : per region-pair interface, reference PARTIAL OUTGOING current
        BCf     : per vacuum-boundary region, reference partial outgoing current

    A region's NET leakage is a sum over its faces and so does not determine the
    per-face partials -- that is why the earlier net-leakage condition left the
    factors underdetermined and they pegged at their bounds.

    solve : callable(materials, df_by_pair, bcf_by_region)
            -> (region_flux (R,G), k_eff, interfaces, boundaries)
        df_by_pair maps an ORDERED region pair to that surface's per-group
        factors and bcf_by_region maps a boundary region to its per-group
        coefficients -- the same keys as the reference dicts (see
        TriSNTransportSolver.aggregate_partial_currents,
        garnet.tri.face_df_from_pairs and garnet.tri.tri_partial_currents).

    One unknown per SURFACE, not per region. Per-region factors are
    over-determined -- a region touching n neighbours gets n reference
    conditions but one unknown -- and least squares then drives them to their
    bounds (measured on HP-MR: cost stalled at 0.678 with every drum factor
    pegged at exp(+-1.2)). Per-surface factors match GET's actual prescription
    and the count of reference conditions.

    GAUGE. The coarse solve is an EIGENproblem, so its solution is fixed only up
    to a constant -- garnet normalizes it to sum(fission source) = n_cells, which
    has nothing to do with the reference's normalization. Comparing partial
    currents in absolute terms therefore compares two arbitrarily-scaled
    solutions, and no factor can absorb the difference: scaling mu, DF or BCf
    changes the PHYSICS, and the solver then re-normalizes to the same gauge
    anyway. (Observed: the factors ran to their bounds and k moved thousands of
    pcm.) Every residual row here is a log ratio and the gauge multiplies the
    coarse flux and every current by one constant, so it enters each row as the
    same additive shift -- removed exactly, and least-squares-optimally, by
    subtracting the mean of the rows. What is left is the physical content: the
    coarse and reference solutions must agree up to one overall constant, which
    is all an eigenproblem can require. This is a gauge fix applied identically
    to all rows, NOT a Laboure-style per-region normalization weight (which
    varies with the factors and does break equivalence).
    """
    from scipy.optimize import least_squares

    R = len(homogenized_materials)
    ref = np.asarray(reference_region_flux, dtype=float)
    G = ref.shape[1]
    live = (ref > ref.max() * 1e-12).reshape(-1)
    bkeys = sorted(reference_boundaries)
    # Which interior surfaces get a DF unknown. df_pairs=[] is the least
    # invasive variant -- boundary coefficients only, no interior DF at all, so
    # the operator stays SYMMETRIC and CG still applies. A surface with no
    # unknown also gets no condition: imposing its reference current with
    # nothing free to meet it is what over-determines the fit.
    ikeys = (sorted(reference_interfaces) if df_pairs is None
             else [tuple(k) for k in df_pairs])
    unknown = set(ikeys) - set(reference_interfaces)
    if unknown:
        raise ValueError(f"df_pairs not in the reference interfaces: {sorted(unknown)}")
    nmu, ndf, nbc = int(live.sum()), len(ikeys) * G, len(bkeys) * G
    state = {"k": float("nan"), "n": 0, "cost": float("nan")}

    def unpack(z):
        z = np.clip(np.asarray(z, dtype=float), -bounds, bounds)
        x = np.zeros(ref.size)
        x[live] = z[:nmu]
        mu = np.exp(x).reshape(ref.shape)
        df = np.exp(z[nmu:nmu + ndf]).reshape(len(ikeys), G)
        bc = np.exp(z[nmu + ndf:]).reshape(len(bkeys), G)
        return mu, df, bc

    def as_dicts(df, bc):
        return ({k: df[i] for i, k in enumerate(ikeys)},
                {k: bc[i] for i, k in enumerate(bkeys)})

    def residual(z):
        state["n"] += 1
        mu, df, bc = unpack(z)
        mats = [_scale_material(homogenized_materials[i], mu[i]) for i in range(R)]
        df_d, bc_d = as_dicts(df, bc)
        phi, k, itf, bnd = solve(mats, df_d, bc_d)
        state["k"] = k
        phi = np.asarray(phi, dtype=float).reshape(-1)
        rows, bad = [], []
        ok = live & (phi > 0)
        rr = np.zeros(ref.size)
        rr[ok] = np.log(mu.reshape(-1)[ok] * phi[ok] / ref.reshape(-1)[ok])
        rows.append(rr[live])
        bad.append(~ok[live])
        for g in range(G):
            for keys, dst, src in ((ikeys, itf, reference_interfaces),
                                   (bkeys, bnd, reference_boundaries)):
                for key in keys:
                    a = dst.get(key, [0.0, 0.0])[0]
                    b = src[key][0]
                    good = a > 0 and b > 0
                    rows.append([np.log(a / b) if good else 0.0])
                    bad.append([not good])
        q = np.concatenate([np.atleast_1d(v) for v in rows])
        bad = np.concatenate([np.atleast_1d(v) for v in bad]).astype(bool)
        # Gauge fix (see docstring). Taken over the FLUX rows only, not all rows.
        # Letting the current rows help set it lets them shift every mu by a
        # common factor, and with a vacuum boundary mu is not a gauge (and under
        # the pre-audit D_SCALING = "inverse", D ~ 1/mu, it never was), so a
        # global mu shift changes leakage and therefore k. Reading it off the flux rows pins mu at the level
        # SPH itself would choose and leaves DF/BCf to carry the currents.
        # (All-row gauge measured: every variant converged but k sat 2900-4100
        # pcm low, systematically, with cost as small as 2e-7.)
        gauge_rows = (~bad[:nmu])
        if gauge_rows.any():
            q = q - q[:nmu][gauge_rows].mean()
        elif (~bad).any():
            q = q - q[~bad].mean()
        q[bad] = 1e2
        # DF normalization, appended AFTER the gauge shift -- it is not a
        # log-ratio of solution quantities, so including it in the mean would
        # corrupt both it and the gauge. Scaling EVERY interface factor by a
        # common c leaves the operator bit-identical (the face pair is
        # (f_L C, f_R C) with C = kf / (f_L/2D_L + f_R/2D_R), so c cancels),
        # an exact null direction distinct from the eigenvector gauge. Pin it by
        # driving the geometric mean of the DF to 1; being null, this cannot
        # compete with the physical conditions, it only restores uniqueness.
        if ndf:
            q = np.append(q, np.log(df).mean())
        state["cost"] = 0.5 * float(q @ q)
        if verbose:
            print(f"    [sph_df] nfev={state['n']:4d} cost={state['cost']:.4e} "
                  f"k={state['k']:.6f}", flush=True)
        return q

    # Start from mu = DF = BCf = 1. No scale-ratio warm start is needed (or
    # meaningful) now that the gauge is removed inside the residual.
    z0 = np.zeros(nmu + ndf + nbc)

    sol = least_squares(residual, z0, method="lm" if method == "lm" else "trf",
                        xtol=tol, ftol=tol, max_nfev=max_iter * (len(z0) + 1))
    mu, df, bc = unpack(sol.x)
    mats = [_scale_material(homogenized_materials[i], mu[i]) for i in range(R)]
    df_d, bc_d = as_dicts(df, bc)          # solve() takes dicts, not raw arrays
    _, k, _, _ = solve(mats, df_d, bc_d)
    # factors is mu alone: df is now per SURFACE, so it no longer shares mu's
    # (R, G) shape and cannot be stacked with it -- it travels in its own field.
    return SphResult(corrected_materials=mats,
                     factors=mu, k_eff=k,
                     iterations=state["n"], df=df, bcf=bc, cost=float(sol.cost),
                     converged=bool(sol.success and sol.cost < 1e-6))
