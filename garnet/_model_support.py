"""Shared validation for the public model builders."""

from collections.abc import Mapping

import numpy as np

from .materials import Kinetics, Material


def checked_materials(materials, *, count=None, groups=None, context="materials"):
    values = list(materials)
    if not values:
        raise ValueError(f"{context} must contain at least one Material")
    if any(not isinstance(m, Material) for m in values):
        raise TypeError(f"{context} must contain Material objects")
    if count is not None and len(values) != count:
        raise ValueError(f"{context} must preserve material count/order ({count} entries)")
    groups = values[0].n_groups if groups is None else groups
    if any(m.n_groups != groups for m in values):
        raise ValueError(f"{context}: all materials must have the same number of groups ({groups})")
    return values


def replace_materials(base, replacements, *, context="replacement materials"):
    """Resolve a complete sequence or partial name/index mapping in stable order."""
    base = list(base)
    if not base:
        raise ValueError("empty model: call fill() with a material first")
    if isinstance(replacements, Mapping):
        values = list(base)
        seen = set()
        for key, material in replacements.items():
            if isinstance(key, str):
                matches = [i for i, m in enumerate(base) if m.name == key]
                if not matches:
                    raise ValueError(f"{context}: unknown material name {key!r}")
                if len(matches) != 1:
                    raise ValueError(f"{context}: ambiguous material name {key!r}; use an index")
                index = matches[0]
            elif isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
                index = int(key)
                if not 0 <= index < len(base):
                    raise ValueError(f"{context}: material index {index} is out of range")
            else:
                raise TypeError(f"{context}: keys must be material names or integer indices")
            if index in seen:
                raise ValueError(f"{context}: material index {index} was specified twice")
            seen.add(index)
            values[index] = material
    else:
        values = replacements
    return checked_materials(values, count=len(base), groups=base[0].n_groups,
                             context=context)


def solver_options(defaults, options):
    """Keep construction options separate from the model's geometry and device."""
    options = dict(options or {})
    owned = {"grid", "mesh", "materials", "material_map", "cell_material", "bc",
             "active", "mask_bc", "mix_material", "mix_weight", "device",
             "alpha_boundary", "eig_solver", "group_operator", "problem_at",
             "kinetics"}
    conflicts = owned.intersection(options)
    if conflicts:
        raise ValueError("solver_kwargs cannot override model settings: "
                         + ", ".join(sorted(conflicts)))
    return dict(defaults, **options)


def checked_kinetics(kinetics, groups, *, material_count=None):
    if not isinstance(kinetics, Kinetics):
        raise TypeError("kinetics must be a Kinetics object")
    if kinetics.velocities.shape[-1] != groups:
        raise ValueError(f"kinetics needs {groups} velocities per row (one per group)")
    if material_count is not None:
        for name in ("velocities", "beta"):
            value = getattr(kinetics, name)
            if value.ndim == 2 and value.shape[0] != material_count:
                raise ValueError(f"kinetics.{name} needs {material_count} material rows")
    if kinetics.chi_delayed is not None and kinetics.chi_delayed.shape[-1] != groups:
        raise ValueError(f"kinetics.chi_delayed needs {groups} groups per spectrum")
    return kinetics


def box_range(value, axis):
    bounds = np.asarray(value, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
        raise ValueError(f"{axis} range must be a finite (lower, upper) pair with lower <= upper")
    return bounds


def material_map(values, shape, count):
    """Copy a host material map, rejecting fractional and invalid indices."""
    if values is None:
        if count != 1:
            raise ValueError("material_map is required with multiple materials")
        return np.zeros(shape, dtype=np.int64)
    values = np.asarray(values)
    if values.shape != tuple(shape):
        raise ValueError(f"material_map shape {values.shape} != grid shape {tuple(shape)}")
    if (not np.all(np.isfinite(values)) or np.any(values != np.floor(values)) or
            np.any(values < 0) or np.any(values >= count)):
        raise ValueError("material_map must contain integer indices in the material list")
    return values.astype(np.int64, copy=True)


# Fraction of the local step width within which a fixed-schedule endpoint is
# moved onto a nearby event rather than kept beside it.
_EVENT_SNAP = 1e-3


def transient_schedule(t_end, dt, event_times, options):
    """Split fixed steps at physical events, or hand them to a controller.

    Each event becomes a step end and a BDF restart point. As for
    ``bdf_restart_times``, the event endpoint belongs to the old segment:
    ``problem_at(t_e)`` must return the pre-event (left-limit) state, and
    history is discarded before the step leaving the event. A grid endpoint
    within ``1e-3`` of the local width of an event is moved onto it, so no
    sliver step is created; elsewhere the containing step is split in two.
    """
    options = dict(options)
    if options.get("adaptive_bdf") is not None:
        options.setdefault("step_solver", "monolithic")
        options.setdefault("time_scheme", "bdf1")
    if options.get("adaptive_ftm") is not None:
        options.setdefault("step_solver", "monolithic")
        options.setdefault("time_scheme", "ftm-bdf2")
    if event_times is None:
        return dt, options
    events = np.asarray(event_times, dtype=float)
    if (events.ndim != 1 or not np.all(np.isfinite(events)) or
            not np.isfinite(t_end) or np.any(events <= 0) or np.any(events >= t_end)):
        raise ValueError("event_times must be finite times strictly inside (0, t_end)")
    if options.get("bdf_restart_times") is not None:
        raise ValueError("use event_times or bdf_restart_times, not both")
    events = np.unique(events)
    if not events.size:
        return dt, options
    from .timescheme import make_time_scheme

    scheme = make_time_scheme(options.get("time_scheme", "backward-euler"),
                              options.get("time_weight"))
    if not scheme.is_bdf:
        raise NotImplementedError("event_times requires a BDF time scheme")
    options["bdf_restart_times"] = events
    if any(options.get(key) is not None for key in ("adaptive_bdf", "adaptive_ftm")):
        return dt, options
    widths = np.asarray(dt, dtype=float)
    if widths.ndim == 0:
        if not np.isfinite(widths) or widths <= 0:
            raise ValueError("dt must be finite and positive")
        endpoints = np.arange(1, int(np.floor(t_end / float(widths))) + 1) * float(widths)
        endpoints = np.r_[endpoints[endpoints < t_end], t_end]
    else:
        if (widths.ndim != 1 or not np.all(np.isfinite(widths)) or np.any(widths <= 0)
                or not np.isclose(widths.sum(), t_end, rtol=1e-12, atol=1e-14)):
            raise ValueError("dt widths must be finite and positive and sum to t_end")
        endpoints = np.cumsum(widths)
        endpoints[-1] = t_end
    # Splitting the step (lo, hi] that contains an event leaves a sliver when
    # the event lies within _EVENT_SNAP of that step's width from either end
    # (an event at 0.1 + 1e-10 with dt = 0.01 made a 1e-10 s step, and
    # variable-step BDF across a ~1e8 width ratio is unstable). Move that
    # endpoint onto the event instead; t_end itself is never moved.
    for event in events:
        index = int(np.searchsorted(endpoints, event))
        hi = endpoints[index]
        lo = endpoints[index - 1] if index else 0.0
        tolerance = _EVENT_SNAP * (hi - lo)
        if hi - event <= tolerance and index < endpoints.size - 1:
            endpoints = np.delete(endpoints, index)
        elif event - lo <= tolerance and index:
            endpoints = np.delete(endpoints, index - 1)
    endpoints = np.sort(np.r_[endpoints, events])
    return np.diff(np.r_[0.0, endpoints]), options
