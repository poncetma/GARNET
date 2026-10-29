"""Temperature feedback: how a hot core differs, neutronically, from a cold one.

This is what makes a neutronics/thermal coupling a coupling rather than a
one-way post-processing step. Without it, power heats the core and the story
ends; with it, the temperature comes back and changes the cross sections, and
the two physics have to agree with each other.

Two effects, both linearized about a reference temperature at which the cross
sections were evaluated:

**Doppler broadening.** As fuel heats, thermal motion of the nuclei broadens
the resonance absorption peaks. The resonances are self-shielded -- the flux is
depressed exactly where the cross section spikes -- so broadening exposes more
of each resonance to the flux and the *effective* absorption rises. Empirically
it goes as sqrt(T) over the range of interest, which is the classical result
for a Doppler-broadened resonance integral:

    dSigma_a,g(T) = c_D * Sigma_a,g(T_ref) * (sqrt(T) - sqrt(T_ref))

It is applied **additively to absorption**, never as a scale on the removal
cross section: removal is `sigma_a + out-scatter`, so scaling it would drag the
scattering along and quietly change the spectrum as well as the absorption.
This is why :class:`garnet.solver.Fields` keeps ``sigma_a`` separately.

**Density / expansion.** Heating lowers the density, so every macroscopic cross
section scales with the material's atom density while the diffusion
coefficient, being an inverse transport cross section, scales the other way:

    f(T) = 1 - beta_exp * (T - T_ref);   Sigma *= f,   D /= f

Unlike Doppler this one *is* legitimately multiplicative on removal, because it
scales absorption and the whole scattering matrix by the same factor. The
emission spectrum chi is untouched: it is a normalized probability, not a
cross section.

**An honest limitation.** The mesh does not move. So `beta_exp` here is a
density effect on a frozen geometry -- the migration area grows as 1/f^2 while
the core dimensions stay put. That is a standard and defensible reduction of a
fuel-density coefficient, but it is not thermal expansion of the core, and it
should not be reported as one.

Both coefficients are per material, so a reflector can be given a different (or
zero) response than the fuel. In a volume-mixed cell each material's change is
evaluated under ITS OWN law -- its own coefficient, its own reference
temperature, its own cross sections -- and the changes are then blended through
exactly the rules the cross sections themselves use (linear for reaction cross
sections, harmonic -- i.e. linear in 1/D -- for the diffusion coefficient). So
a cell that is half fuel and half absorber Doppler-broadens half the fuel's
absorption, not half the cell's, and a material sitting at its own reference
temperature contributes no change whatever its partner's reference is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThermalFeedback:
    """Per-material temperature coefficients.

    t_ref      : (M,) reference temperature in K -- the temperature at which
                 each material's tabulated cross sections were evaluated. For
                 the HP-MR's Griffin library that is the ``grid_index`` node,
                 ~800 K.
    doppler    : (M,) or (M, G) coefficient c_D in K^-1/2 acting on absorption.
                 A per-group form lets the resonance range carry the effect
                 while thermal groups do not; a scalar per material applies it
                 to every group. Zero for non-fuel.
    expansion  : (M,) density coefficient beta_exp in 1/K, or None.
    doppler_updates_diffusion : also feed the added absorption into the
                 transport cross section, 1/D += 3 dSigma_a. Exact when
                 D = 1/(3 Sigma_tr), which is what ``volume_homogenize``
                 produces. Turn off for libraries carrying an independent
                 ``total``.
    """

    t_ref: np.ndarray
    doppler: np.ndarray
    expansion: np.ndarray | None = None
    doppler_updates_diffusion: bool = True

    def __post_init__(self):
        self.t_ref = np.atleast_1d(np.asarray(self.t_ref, dtype=float))
        self.doppler = np.asarray(self.doppler, dtype=float)
        if self.doppler.ndim == 1:
            self.doppler = self.doppler.reshape(-1, 1)      # broadcast over groups
        if len(self.doppler) != len(self.t_ref):
            raise ValueError("doppler must have one row per material")
        if np.any(self.t_ref <= 0):
            raise ValueError("reference temperatures must be positive (kelvin)")
        if self.expansion is not None:
            self.expansion = np.atleast_1d(np.asarray(self.expansion, dtype=float))
            if len(self.expansion) != len(self.t_ref):
                raise ValueError("expansion must have one value per material")

    @property
    def n_materials(self) -> int:
        return len(self.t_ref)

    def hook(self, temperature, materials=None):
        """Build the ``xs_update`` callable for a given temperature field.

        Pass the result as ``xs_update=`` to any eigen solver. The closure
        captures the temperature, so a coupling driver makes a new one each
        iteration -- which is also when the solver (and hence its operators)
        must be rebuilt.

        materials : the material list the solver is built from. Only read in a
            volume-mixed cell whose two materials respond DIFFERENTLY (e.g.
            fuel blended with a non-Doppler absorber), where the per-material
            change needs each component's own cross sections. A ``Fields``
            that carries its ``materials`` takes precedence. Without either,
            such a cell is refused rather than mis-blended.
        """
        # Deliberately NOT np.asarray: on GPU the temperature is a CuPy array
        # and numpy refuses the implicit conversion, so forcing it here would
        # make the whole coupled path CPU-only. The array module is resolved
        # from the fields' own backend in _apply instead.
        T = temperature

        def update(fields):
            self._apply(fields, T, materials)

        return update

    def _apply(self, fields, temperature, materials=None):
        blend = fields.blend
        xp = blend.xp
        G = fields.n_groups
        dtype = blend.dtype
        # xp.asarray keeps a device array on the device and lifts a host array
        # onto it; either way the arithmetic below stays on one backend.
        temperature_field = xp.asarray(temperature, dtype=dtype)
        if temperature_field.shape != blend.shape:
            raise ValueError(f"temperature shape {temperature_field.shape} != grid shape "
                             f"{blend.shape}")
        coefficient_groups = self.doppler.shape[1]
        if coefficient_groups not in (1, G):
            # Otherwise the group lookup below would clamp to the last column
            # and silently give every remaining group the wrong coefficient.
            raise ValueError(
                f"doppler must have 1 column (applied to every group) or one "
                f"per group ({G}), got {coefficient_groups}")

        # The law is evaluated per MATERIAL and only then blended. Blending the
        # coefficients first (c_D, t_ref, beta each volume-averaged, then
        # multiplied by the cell's blended Sigma_a) puts the fuel's Doppler on
        # its partner's absorption too -- 50x too much in a fuel/B4C cell --
        # and averages reference temperatures, so a cell whose fuel sits
        # exactly at ITS reference still moves.
        components = blend.components()
        indices = [index for index, _ in components]
        fractions = [fraction for _, fraction in components]
        t_ref = xp.asarray(self.t_ref, dtype=dtype)
        # Clamp: sqrt of a negative temperature is not a physics question, and
        # a coupling iterate can transiently undershoot before it converges.
        sqrt_temperature = xp.sqrt(xp.maximum(temperature_field, 0.0))
        sqrt_change = [sqrt_temperature - xp.sqrt(t_ref[i]) for i in indices]
        doppler = xp.asarray(self.doppler, dtype=dtype)

        def doppler_multipliers(g):
            column = doppler[:, 0 if coefficient_groups == 1 else g]
            return [column[i] * s for i, s in zip(indices, sqrt_change)]

        if self.expansion is not None:
            beta = xp.asarray(self.expansion, dtype=dtype)
            density_factors = [1.0 - beta[i] * (temperature_field - t_ref[i])
                               for i in indices]
            if any(bool(xp.any((fraction > 0.0) & (factor <= 0.0)))
                   for fraction, factor in zip(fractions, density_factors)):
                raise ValueError(
                    "the expansion law drove a density factor to zero or "
                    "below; the linearization is only valid for "
                    "beta_exp * (T - T_ref) << 1")
        else:
            density_factors = None

        # Cells genuinely shared by two materials. Elsewhere one component has
        # fraction 1 and its own law applies unchanged.
        shared = None
        if len(components) == 2:
            shared = (fractions[0] > 0.0) & (fractions[1] > 0.0)
            if not bool(xp.any(shared)):
                shared = None

        # Where both materials respond identically, the per-material blend
        # collapses to that common law and no cross-section table is needed.
        # Only a cell whose two materials respond DIFFERENTLY needs them.
        weighted = False
        if shared is not None:
            differ = xp.zeros(blend.shape, dtype=bool)
            for g in range(G if coefficient_groups > 1 else 1):
                kb, ko = doppler_multipliers(g)
                differ |= kb != ko
            if density_factors is not None:
                differ |= density_factors[0] != density_factors[1]
            weighted = bool(xp.any(shared & differ))
        if weighted:
            materials = getattr(fields, "materials", None) or materials
            if materials is None:
                raise ValueError(
                    "a volume-mixed cell blends two materials with different "
                    "feedback laws, and the per-material change needs each "
                    "component's own cross sections: pass materials= to "
                    "ThermalFeedback.hook")
            materials = list(materials)
            if len(materials) != self.n_materials:
                raise ValueError(f"feedback has {self.n_materials} materials, "
                                 f"the solver {len(materials)}")

            def table(attr):
                return xp.asarray(np.array([getattr(m, attr) for m in materials],
                                           dtype=float), dtype=dtype)

            tables = {attr: table(attr) for attr in
                      ("sigma_a", "removal", "sigma_t", "nu_sigma_f",
                       "diffusion", "sigma_s")}

        def single_law(multipliers):
            # Fraction-weighted, so a fully-converted cell (w = 1) takes its
            # partner's law, and an identical law comes back unchanged.
            out = fractions[0] * multipliers[0]
            for fraction, multiplier in zip(fractions[1:], multipliers[1:]):
                out = out + fraction * multiplier
            return out if len(components) > 1 else multipliers[0]

        def blended_multiplier(values, multipliers):
            """Per-cell factor m with  m * sum_i f_i v_i = sum_i f_i m_i v_i:
            each material's change weighted by its own share of the quantity.
            Applied as a RATIO to the current field, so it composes with any
            per-cell change an earlier hook made."""
            law = single_law(multipliers)
            if not weighted:
                return law
            num = sum(f * m * v for f, m, v in zip(fractions, multipliers, values))
            den = sum(f * v for f, v in zip(fractions, values))
            usable = shared & (den != 0.0)
            return xp.where(usable, num / xp.where(usable, den, 1.0), law)

        def per_component(values):
            return [values[i] for i in indices]

        for g in range(G):
            doppler_g = doppler_multipliers(g)
            absorption = (per_component(tables["sigma_a"][:, g]) if weighted
                          else None)
            d_sa = fields.sigma_a[g] * blended_multiplier(absorption, doppler_g)

            fields.sigma_a[g] = fields.sigma_a[g] + d_sa
            fields.removal[g] = fields.removal[g] + d_sa
            fields.sigma_t[g] = fields.sigma_t[g] + d_sa
            if self.doppler_updates_diffusion:
                # 1/(1/D + 3 dSa) written as D/(1 + 3 dSa D): algebraically the
                # same, but exact when dSa = 0 (the reciprocal-of-a-reciprocal
                # form loses a bit or two there, which would make a switched-off
                # feedback measurably different from no feedback at all). The
                # harmonic blend is linear in 1/D, so adding the BLENDED dSa is
                # the same as blending each material's 1/D + 3 dSa_i.
                fields.diffusion[g] = fields.diffusion[g] / (
                    1.0 + 3.0 * d_sa * fields.diffusion[g])

            if density_factors is None:
                continue
            if weighted:
                d_absorption = [k * a for k, a in zip(doppler_g, absorption)]

                def after_doppler(attr):
                    return [v + d for v, d in zip(
                        per_component(tables[attr][:, g]), d_absorption)]

                inverse_d = [1.0 / v for v in per_component(tables["diffusion"][:, g])]
                if self.doppler_updates_diffusion:
                    inverse_d = [v + 3.0 * d for v, d in zip(inverse_d, d_absorption)]
                factor = {
                    "sigma_a": blended_multiplier(after_doppler("sigma_a"), density_factors),
                    "removal": blended_multiplier(after_doppler("removal"), density_factors),
                    "sigma_t": blended_multiplier(after_doppler("sigma_t"), density_factors),
                    "nu_sigma_f": blended_multiplier(
                        per_component(tables["nu_sigma_f"][:, g]), density_factors),
                    "diffusion": blended_multiplier(inverse_d, density_factors),
                }
            else:
                common = single_law(density_factors)
                factor = dict.fromkeys(
                    ("sigma_a", "removal", "sigma_t", "nu_sigma_f", "diffusion"), common)
            fields.sigma_a[g] = fields.sigma_a[g] * factor["sigma_a"]
            fields.removal[g] = fields.removal[g] * factor["removal"]
            fields.sigma_t[g] = fields.sigma_t[g] * factor["sigma_t"]
            fields.nu_sigma_f[g] = fields.nu_sigma_f[g] * factor["nu_sigma_f"]
            fields.diffusion[g] = fields.diffusion[g] / factor["diffusion"]
            for gt in range(G):
                if fields.sigma_s[g][gt] is not None:
                    scatter_factor = (blended_multiplier(
                        per_component(tables["sigma_s"][:, g, gt]), density_factors)
                        if weighted else factor["sigma_a"])
                    fields.sigma_s[g][gt] = fields.sigma_s[g][gt] * scatter_factor
        # chi is deliberately untouched: it is a normalized emission spectrum,
        # not a cross section, and scaling it would lose fission neutrons.


@dataclass
class TabulatedFeedback:
    """Temperature feedback by interpolating a library's own branches.

    The honest version of :class:`ThermalFeedback`. Rather than scaling one
    tabulation node by an analytic sqrt(T) law with a fitted coefficient, this
    reads the cross sections the evaluation actually produced at each
    temperature node and interpolates between them. Everything moves the way
    the data says: absorption, production, the full scattering matrix, the
    diffusion coefficient and the emission spectrum, in every group at once.

    material : index into the solver's materials list whose data varies.
    temperatures : (N,) the tabulation nodes, ascending, K.
    tables : mapping of Fields attribute -> array, ``(N, G)`` for the per-group
        quantities and ``(N, G, G)`` for ``sigma_s``.

    Linear in T between nodes and clamped outside them -- extrapolating a
    resonance-broadening curve past the evaluated range is not something the
    data supports, and a coupling iterate can transiently overshoot.
    """

    material: int
    temperatures: np.ndarray
    tables: dict
    #: Index of the node the solver's OWN material data was tabulated at.
    reference: int = 2

    def __post_init__(self):
        self.temperatures = np.asarray(self.temperatures, dtype=float)
        if self.temperatures.ndim != 1 or len(self.temperatures) < 2:
            raise ValueError("need at least two temperature nodes")
        if np.any(np.diff(self.temperatures) <= 0):
            raise ValueError("temperature nodes must be strictly ascending")
        self.tables = {k: np.asarray(v, dtype=float)
                       for k, v in self.tables.items()}
        n = len(self.temperatures)
        for k, v in self.tables.items():
            if v.shape[0] != n:
                raise ValueError(f"table {k!r} has {v.shape[0]} nodes, "
                                 f"expected {n}")

    @property
    def t_ref(self):
        """Present for interface parity with ThermalFeedback (drivers use it
        to name a reference state): the node the solver's own data was
        tabulated at, ``temperatures[reference]``."""
        return self.temperatures[[int(self.reference)]]

    def hook(self, temperature, materials=None):
        """``xs_update`` for a temperature field. ``materials`` is accepted
        for interface parity with :meth:`ThermalFeedback.hook` and unused: the
        branch data already is this material's own."""
        T = temperature

        def update(fields):
            self._apply(fields, T)

        return update

    def _apply(self, fields, temperature):
        blend = fields.blend
        xp = blend.xp
        G = fields.n_groups
        T = xp.asarray(temperature, dtype=blend.dtype)
        if T.shape != blend.shape:
            raise ValueError(f"temperature shape {T.shape} != grid shape "
                             f"{blend.shape}")

        nodes = xp.asarray(self.temperatures, dtype=blend.dtype)
        n = len(self.temperatures)
        # Bracketing interval and linear weight, per cell.
        idx = xp.clip(xp.searchsorted(nodes, T) - 1, 0, n - 2)
        lo = nodes[idx]
        hi = nodes[idx + 1]
        w = xp.clip((T - lo) / (hi - lo), 0.0, 1.0)

        # Only this material's cells take the branch data. In the HP-MR the
        # fuel is never volume-mixed (mixing is a drum-arc device), so a plain
        # index test is exact; a mixed fuel cell would need the blend applied
        # to the interpolated values instead. Refused on BOTH sides of the
        # blend: as the base it would scale the partner's share too, and as the
        # partner its cells would silently get no feedback at all.
        mmap = blend.mmap
        if mmap is None:
            here = xp.ones(blend.shape, dtype=bool)
        else:
            here = mmap == self.material
            if blend.mix:
                as_base = here & blend.active_mix
                as_partner = (blend.active_mix & (blend.mm2c == self.material)
                              & (blend.w > 0.0))
                if bool(xp.any(as_base | as_partner)):
                    raise ValueError(
                        "TabulatedFeedback material is volume-mixed in some "
                        "cells; interpolating there would ignore the blend "
                        "partner")

        # Applied as a RATIO to the reference node, not as an absolute value.
        # Setting the cross sections outright would silently erase any other
        # perturbation to the same material -- a uniform absorption insertion
        # for a transient, an SPH factor, a density scaling -- because the fuel
        # is most of the core and the feedback would overwrite all of it. As a
        # ratio the feedback expresses what it actually knows ("this is how much
        # the data moves with temperature") and composes with everything else.
        ref = int(self.reference)

        def interp(table_2d):
            col = xp.asarray(table_2d, dtype=blend.dtype)
            v = (1.0 - w) * col[idx] + w * col[idx + 1]
            base = col[ref]
            return v / base if float(abs(base)) > 0.0 else v * 0.0 + 1.0

        for attr in ("sigma_a", "nu_sigma_f", "diffusion", "sigma_t"):
            tab = self.tables.get(attr)
            if tab is None:
                continue
            for g in range(G):
                fl = getattr(fields, attr)
                fl[g] = xp.where(here, fl[g] * interp(tab[:, g]), fl[g])
        # chi is a normalized spectrum, so a ratio would not preserve its sum;
        # its temperature dependence is negligible and it is left alone.

        s_tab = self.tables.get("sigma_s")
        if s_tab is not None:
            for gf in range(G):
                for gt in range(G):
                    if gf == gt:
                        continue
                    col = s_tab[:, gf, gt]
                    if not np.any(col):
                        continue
                    cur = fields.sigma_s[gf][gt]
                    if cur is None:
                        continue
                    fields.sigma_s[gf][gt] = xp.where(
                        here, cur * interp(col), cur)

        # removal is a derived quantity: sigma_a plus everything that scatters
        # OUT of the group. Recompute it wherever the data moved, or the
        # operator keeps the old group coupling.
        for g in range(G):
            out_scatter = None
            for gt in range(G):
                if gt == g:
                    continue
                s = fields.sigma_s[g][gt]
                if s is None:
                    continue
                out_scatter = s if out_scatter is None else out_scatter + s
            rem = fields.sigma_a[g] if out_scatter is None else fields.sigma_a[g] + out_scatter
            fields.removal[g] = xp.where(here, rem, fields.removal[g])


def uniform(feedback: ThermalFeedback, shape, temperature):
    """``xs_update`` for a uniform temperature -- the probe used to measure a
    coefficient without involving the thermal solver at all."""
    return feedback.hook(np.full(shape, float(temperature)))


def measure_pcm_per_K(solver_factory, shape, feedback, t_cal=800.0, dT=50.0,
                      **solve_kwargs):
    """Reactivity coefficient in pcm/K by central difference about ``t_cal``.

    solver_factory : callable ``xs_update -> solver`` building the eigen solver
                     for a given feedback hook (so this works for any geometry
                     and any angular approximation).

    Returns alpha = 1e5 * (1/k(T-dT) - 1/k(T+dT)) / (2 dT), i.e. drho/dT. A
    physical fuel coefficient is NEGATIVE: hotter fuel absorbs more, k falls.
    """
    ks = []
    for t in (t_cal - dT, t_cal + dT):
        res = solver_factory(uniform(feedback, shape, t)).solve(**solve_kwargs)
        if not res.converged:
            raise RuntimeError(f"eigen solve did not converge at T = {t} K")
        ks.append(res.k_eff)
    return 1e5 * (1.0 / ks[0] - 1.0 / ks[1]) / (2.0 * dT)


def scale_to(feedback: ThermalFeedback, factor: float) -> ThermalFeedback:
    """The same feedback with both coefficient vectors scaled.

    To first order reactivity is linear in the coefficient amplitude
    (perturbation theory), so one measurement fixes the scale needed to hit a
    target pcm/K -- which is how :func:`calibrate` works.
    """
    return ThermalFeedback(
        t_ref=feedback.t_ref.copy(),
        doppler=feedback.doppler * factor,
        expansion=None if feedback.expansion is None else feedback.expansion * factor,
        doppler_updates_diffusion=feedback.doppler_updates_diffusion)


def calibrate(solver_factory, shape, feedback, target_pcm_per_K,
              t_cal=800.0, dT=50.0, **solve_kwargs):
    """Scale a feedback's coefficients to reproduce a requested pcm/K.

    The shape of the response (which materials and groups respond, and in what
    ratio) is the caller's physics; only its amplitude is fitted. Returns
    (calibrated_feedback, measured_alpha).
    """
    if target_pcm_per_K >= 0.0:
        raise ValueError(
            "a fuel temperature coefficient must be negative; a positive one "
            "is a reactor that heats itself into a runaway, not a tuning choice")
    alpha = measure_pcm_per_K(solver_factory, shape, feedback, t_cal, dT,
                              **solve_kwargs)
    if alpha == 0.0:
        raise ValueError("the trial feedback produced no reactivity change; "
                         "check that the responding materials are the fuel")
    scaled = scale_to(feedback, target_pcm_per_K / alpha)
    return scaled, measure_pcm_per_K(solver_factory, shape, scaled, t_cal, dT,
                                     **solve_kwargs)
