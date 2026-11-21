"""Reader for GeN-Foam ``nuclearData`` dictionaries (OpenFOAM format).

GeN-Foam (EPFL/PSI) stores zone-wise multigroup data per state in SI units
(per metre, seconds per metre). This module turns one (state, zone) into a
GARNET :class:`~garnet.materials.Material` and
:class:`~garnet.materials.Kinetics`, with the conventions of GeN-Foam's own
S_N solver (``src/classes/neutronics/SN``):

* ``scatteringMatrixP<l>`` entry (J, I) scatters FROM group J TO group I
  (``readNuclearData.H``: ``sigmaFromTo[l][J][I] = matrix(J, I)``), the
  Legendre moment WITHOUT the (2l+1) factor (the solver applies it), as GARNET's
  ``sigma_s_moments[l, from, to]``;
* ``sigmaRemoval`` is the total minus within-group P0 scattering: the S_N
  collision term is ``sigmaRemoval + scatteringMatrixP0[g][g]`` (``fluxEqSN.H``);
* ``legendreMoments`` is the number of moments the solver uses (3 means
  P0-P2), whatever the file carries;
* fission: prompt ``chiPrompt (1 - beta) nuSigmaEff / k`` plus the delayed
  source ``chiDelayed sum_i lambda_i C_i``, at equilibrium in an eigenvalue
  solve, so the equilibrium spectrum is ``(1 - beta) chiPrompt + beta chiDelayed``;
* ``IV`` is 1/v in s/m, ``D`` in m, ``sigmaPow`` in J/m (kappa Sigma_f).

GARNET's ``Material.chi`` must sum to 1. When the combined spectrum does not
(rounding in the file), the sum is folded into ``nu_sigma_f``, which leaves the
eigenvalue unchanged; ``GenFoamZone.chi_sum`` reports it.

The data files are read in place, never copied (GeN-Foam's tutorials carry
their own licence). Only the dictionary syntax used by nuclearData files is
parsed: sub-dictionaries, ``key value;`` entries, ``nonuniform List<scalar> N
(...)`` fields, ``N M ( (...) ... )`` matrices and named-dictionary lists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .materials import Kinetics, Material

__all__ = ["GenFoamZone", "parse_foam_dict", "read_nuclear_data"]

_TOKEN = re.compile(r'"[^"]*"|[{}();]|[^\s{}();]+')


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


class _Parser:
    def __init__(self, text: str):
        self.tok = _TOKEN.findall(_strip_comments(text))
        self.i = 0

    def peek(self):
        return self.tok[self.i] if self.i < len(self.tok) else None

    def next(self):
        t = self.peek()
        if t is None:
            raise ValueError("unexpected end of the dictionary")
        self.i += 1
        return t

    def expect(self, t):
        got = self.next()
        if got != t:
            raise ValueError(f"expected {t!r}, got {got!r} (token {self.i})")

    def dictionary(self, closing):
        out = {}
        while self.peek() != closing:
            if self.peek() is None:
                if closing is None:
                    break
                raise ValueError("unterminated dictionary")
            key = self.next()
            if key in ("{", "}", "(", ")", ";"):
                raise ValueError(f"unexpected {key!r} where a keyword was expected")
            if self.peek() == "{":
                self.next()
                out[key] = self.dictionary("}")
                self.expect("}")
            else:
                vals = []
                while self.peek() != ";":
                    vals.append(self.value())
                self.expect(";")
                out[key] = _interpret(vals)
        return out

    def value(self):
        t = self.next()
        if t == "(":
            return self.list_()
        if t in ("{", "}", ")", ";"):
            raise ValueError(f"unexpected {t!r} in a value")
        return t

    def list_(self):
        items, named = [], {}
        while self.peek() != ")":
            if self.peek() is None:
                raise ValueError("unterminated list")
            t = self.next()
            if t == "(":
                items.append(self.list_())
            elif self.peek() == "{":            # named dictionary in a list
                self.next()
                named[t] = self.dictionary("}")
                self.expect("}")
            else:
                items.append(t)
        self.expect(")")
        if named and items:
            raise ValueError("a list mixes named dictionaries and values")
        return named if named else items


def _num(x):
    return float(x)


def _interpret(vals):
    """A value as a float, an array (field or matrix), a dict of named
    dictionaries, or the raw token list."""
    if len(vals) == 1 and isinstance(vals[0], str):
        try:
            return _num(vals[0])
        except ValueError:
            return vals[0]
    if len(vals) == 1 and isinstance(vals[0], dict):
        return vals[0]
    if (len(vals) == 4 and vals[0] == "nonuniform" and vals[1].startswith("List<")
            and isinstance(vals[3], list)):
        n = int(vals[2])
        a = np.array([_num(v) for v in vals[3]])
        if a.size != n:
            raise ValueError(f"field declares {n} values, has {a.size}")
        return a
    if len(vals) == 2 and vals[0] == "uniform":
        return _num(vals[1])
    if len(vals) == 3 and isinstance(vals[2], list) and all(isinstance(r, list) for r in vals[2]):
        n, m = int(vals[0]), int(vals[1])
        a = np.array([[_num(v) for v in r] for r in vals[2]])
        if a.shape != (n, m):
            raise ValueError(f"matrix declares {n} x {m}, has {a.shape}")
        return a
    if len(vals) == 1 and isinstance(vals[0], list):
        return np.array([_num(v) for v in vals[0]])
    return vals


def parse_foam_dict(text: str) -> dict:
    """Parse OpenFOAM dictionary text (the subset used by nuclearData)."""
    return _Parser(text).dictionary(None)


@dataclass
class GenFoamZone:
    """One zone of one state of a GeN-Foam nuclearData file, in SI units as
    read (``data``), with conversions to GARNET objects (per cm, cm/s)."""

    state: str
    zone: str
    data: dict
    energy_groups: int
    prec_groups: int
    legendre_moments: int
    chi_sum: float = field(init=False)

    def __post_init__(self):
        G, I = self.energy_groups, self.prec_groups
        need = ["IV", "D", "nuSigmaEff", "sigmaRemoval", "chiPrompt", "chiDelayed",
                "Beta", "lambda", "scatteringMatrixP0"]
        missing = [k for k in need if k not in self.data]
        if missing:
            raise KeyError(f"zone {self.zone!r} of state {self.state!r} lacks {missing}")
        for k in ("IV", "D", "nuSigmaEff", "sigmaRemoval", "chiPrompt", "chiDelayed"):
            if np.shape(self.data[k]) != (G,):
                raise ValueError(f"{k} must have {G} values, got {np.shape(self.data[k])}")
        for k in ("Beta", "lambda"):
            if np.shape(self.data[k]) != (I,):
                raise ValueError(f"{k} must have {I} values, got {np.shape(self.data[k])}")
        for l in range(self.available_order + 1):
            if np.shape(self.data[f"scatteringMatrixP{l}"]) != (G, G):
                raise ValueError(f"scatteringMatrixP{l} must be {G} x {G}")
        self.chi_sum = float(self.chi_total_raw.sum())

    @property
    def available_order(self) -> int:
        """Highest scatteringMatrixP<l> present in the file."""
        l = 0
        while f"scatteringMatrixP{l + 1}" in self.data:
            l += 1
        return l

    @property
    def solver_order(self) -> int:
        """The order GeN-Foam's S_N uses: legendreMoments - 1 (capped by the file)."""
        return min(self.legendre_moments - 1, self.available_order)

    @property
    def beta_total(self) -> float:
        return float(np.sum(self.data["Beta"]))

    @property
    def chi_total_raw(self) -> np.ndarray:
        b = self.beta_total
        return (1.0 - b) * self.data["chiPrompt"] + b * self.data["chiDelayed"]

    def moments(self, order: int) -> np.ndarray:
        """Scattering moments [l, from, to] in 1/cm, l = 0..order."""
        return np.stack([self.data[f"scatteringMatrixP{l}"] for l in range(order + 1)]) / 100.0

    def material(self, scattering_order: int | None = None, name: str | None = None) -> Material:
        """The zone as a GARNET Material (1/cm, cm). ``scattering_order``
        defaults to the order GeN-Foam's S_N uses (``solver_order``); 0 gives
        an isotropic material carrying the true total (not transport-corrected)."""
        L = self.solver_order if scattering_order is None else int(scattering_order)
        if not 0 <= L <= self.available_order:
            raise ValueError(f"scattering order {L} not in the file (P0-P{self.available_order})")
        d = self.data
        s0 = d["scatteringMatrixP0"] / 100.0
        removal = d["sigmaRemoval"] / 100.0
        out = s0.sum(axis=1) - np.diag(s0)
        chi = self.chi_total_raw / self.chi_sum
        kappa = d["sigmaPow"] / 100.0 if "sigmaPow" in d else None
        return Material(diffusion=d["D"] * 100.0,
                        sigma_a=removal - out,
                        nu_sigma_f=d["nuSigmaEff"] / 100.0 * self.chi_sum,
                        sigma_s=s0,
                        chi=chi,
                        total=removal + np.diag(s0),
                        kappa_fission=kappa,
                        name=name or f"{self.zone}",
                        sigma_s_moments=self.moments(L) if L > 0 else None)

    def kinetics(self) -> Kinetics:
        """Speeds (cm/s), delayed fractions, decay constants, delayed spectrum."""
        d = self.data
        return Kinetics(velocities=100.0 / d["IV"], beta=d["Beta"].copy(),
                        decay=d["lambda"].copy(), chi_delayed=d["chiDelayed"].copy())


def read_nuclear_data(path: str | Path) -> dict[tuple[str, str], GenFoamZone]:
    """Read a GeN-Foam nuclearData file: {(state, zone): GenFoamZone}.

    Files with a ``states ( name { zones ( ... ) } )`` list and older files
    with a top-level ``zones ( ... )`` list (state ``"reference"``) are read."""
    d = parse_foam_dict(Path(path).read_text())
    G = int(d["energyGroups"])
    I = int(d["precGroups"])
    L = int(d.get("legendreMoments", 1))
    if "states" in d:
        states = d["states"]
    elif "zones" in d:
        states = {"reference": {"zones": d["zones"]}}
    else:
        raise ValueError("no 'states' or 'zones' list in the nuclearData file")
    if not isinstance(states, dict):
        raise ValueError("'states' must be a list of named dictionaries")
    out = {}
    for s, body in states.items():
        zones = body.get("zones", {})
        if not isinstance(zones, dict):
            raise ValueError(f"state {s!r}: 'zones' must be a list of named dictionaries")
        for z, data in zones.items():
            out[(s, z)] = GenFoamZone(s, z, data, G, I, L)
    return out
