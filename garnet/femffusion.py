"""Readers for FEMFFUSION cross-section files.

FEMFFUSION (https://github.com/Zonni/FEMFFUSION) defines its materials in two
file formats; both are parsed here into GARNET ``Material`` lists so that
SPH-corrected constants produced with FEMFFUSION tooling drop straight into
the GARNET solvers. The semantics below mirror FEMFFUSION's own parsers
(``src/io/materials.cc`` and ``src/io/input_mat.cc``), which were read to pin
down the conventions:

- ``read_xsec`` -- the two-group ``.xsec`` format (``parse_xsec_2g``): a
  ``Materials`` section mapping assemblies to material ids, an ``XSecs``
  section with two lines per material

  .. code-block:: text

      id  Sigma_tr1  Sigma_a1  nuSigma_f1  Sigma_f1  Sigma_1->2
          Sigma_tr2  Sigma_a2  nuSigma_f2  Sigma_f2

  and optional ``Precursors`` / ``Velocity`` kinetics blocks. The fission
  spectrum is chi = (1, 0) by convention and D_g = 1/(3 Sigma_tr,g). The
  ``Sigma_f`` column is a plain fission cross section; it is stored as
  ``kappa_fission = energy_per_fission * Sigma_f`` (default 200 MeV in J, the
  same energy units as the library readers' ``kappaFission``). The
  group-independent kappa cancels in :func:`garnet.power.power_density`'s
  normalization to rated power, so only the Sigma_f shape matters. A 3-D
  file splits ``Materials`` into ``PLANE n`` blocks, one map per axial plane.

- ``read_material_xml`` -- the multigroup XML materials file
  (``input.mat.xml``): ``<materials ngroups="G">`` with one ``<mix>`` per
  material. In the file ``SigmaS`` is stored with row = TO group and
  column = FROM group (FEMFFUSION transposes on read, materials.cc:2502);
  it is transposed here to GARNET's ``sigma_s[from, to]``. The diffusion
  coefficient uses ``SigmaTR`` when present, else ``SigmaT``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .materials import Kinetics, Material

_KEYWORDS = {"Materials", "XSecs", "Precursors", "Velocity"}


@dataclass
class XsecFile:
    """Contents of a FEMFFUSION two-group .xsec file.

    materials : list of Material; index i holds the file's material id i+1.
    core_map  : the raw ``Materials`` section as ragged rows of ids (the row
                label is dropped). Geometry placement -- rectangular or
                hexagonal stagger -- is up to the caller, as in FEMFFUSION.
                For a 3-D file (``PLANE n`` headers) it is instead a list of
                such maps, one per plane, bottom (``PLANE 1``) first.
    kinetics  : Kinetics from the Precursors/Velocity blocks, or None.
    n_planes  : number of ``PLANE`` blocks; 0 for a 2-D file.
    """

    materials: list
    core_map: list
    kinetics: Kinetics | None
    n_planes: int = 0


def _content_lines(path):
    """Yield (stripped) non-empty, non-comment lines."""
    for raw in open(path).read().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            yield line


# Recoverable energy per fission (~200 MeV), in J.
ENERGY_PER_FISSION_J = 200.0e6 * 1.602176634e-19


def read_xsec(path, name_prefix: str = "",
              energy_per_fission: float = ENERGY_PER_FISSION_J) -> XsecFile:
    """Parse a FEMFFUSION two-group ``.xsec`` (Valkin/xsec_2g) file.

    ``energy_per_fission`` converts the file's Sigma_f column into
    ``kappa_fission`` (energy units, J by default).
    """
    lines = list(_content_lines(path))
    materials, core_map, kinetics, n_planes = [], [], None, 0
    beta = decay = velocities = None

    i = 0
    while i < len(lines):
        tok = lines[i].split()
        key = tok[0]
        if key == "Materials":
            i += 1
            planes = []
            while i < len(lines) and lines[i].split()[0] not in _KEYWORDS:
                row = lines[i].split()
                if row[0].upper() == "PLANE":
                    # A plane header, not a map row: reading it as one put a
                    # stray [n] row between planes and merged them all.
                    if len(row) != 2 or int(row[1]) != len(planes) + 1:
                        raise ValueError(f"{path}: expected 'PLANE {len(planes) + 1}', "
                                         f"got {lines[i]!r}")
                    if core_map and not planes:
                        raise ValueError(f"{path}: map rows before the first PLANE")
                    planes.append([])
                    i += 1
                    continue
                ids = [int(v) for v in row[1:]]  # drop row label
                (planes[-1] if planes else core_map).append(ids)
                i += 1
            if planes:
                core_map, n_planes = planes, len(planes)
        elif key == "XSecs":
            n_mats = int(tok[1])
            i += 1
            for m in range(1, n_mats + 1):
                first = lines[i].split()
                if int(first[0]) != m:
                    raise ValueError(f"{path}: expected material {m}, got {first[0]}")
                tr1, a1, nsf1, sf1, s12 = (float(v) for v in first[1:6])
                second = lines[i + 1].split()
                tr2, a2, nsf2, sf2 = (float(v) for v in second[:4])
                materials.append(Material(
                    name=f"{name_prefix}{m}",
                    diffusion=[1.0 / (3.0 * tr1), 1.0 / (3.0 * tr2)],
                    sigma_a=[a1, a2], nu_sigma_f=[nsf1, nsf2],
                    sigma_s=[[0.0, s12], [0.0, 0.0]], chi=[1.0, 0.0],
                    total=[tr1, tr2],
                    # None on non-fissile rows, like the library readers.
                    kappa_fission=([energy_per_fission * sf1,
                                    energy_per_fission * sf2]
                                   if sf1 > 0 or sf2 > 0 else None)))
                i += 2
        elif key == "Precursors":
            n_prec = int(tok[1])
            rows = [lines[i + 1 + k].split() for k in range(n_prec)]
            beta = [float(r[1]) for r in rows]
            decay = [float(r[2]) for r in rows]
            i += 1 + n_prec
        elif key == "Velocity":
            velocities = [float(v) for v in lines[i + 1].split()]
            i += 2
        else:
            i += 1

    if beta is not None and velocities is not None:
        kinetics = Kinetics(velocities=velocities, beta=beta, decay=decay)
    return XsecFile(materials=materials, core_map=core_map, kinetics=kinetics,
                    n_planes=n_planes)


def _xml_rows(element) -> np.ndarray:
    """Parse an XML block of ';'-terminated rows of floats."""
    rows = [r.split() for r in element.text.split(";") if r.strip()]
    return np.array([[float(v) for v in r] for r in rows])


def read_material_xml(path) -> list:
    """Parse a FEMFFUSION multigroup XML materials file (``input.mat.xml``).

    Returns the materials ordered by ``mix id``. Upscatter is preserved;
    ``total`` is set from SigmaT so the SP3 solver uses the file's data.
    """
    root = ET.parse(path).getroot()
    G = int(root.attrib["ngroups"])
    mixes = sorted(root.iter("mix"), key=lambda m: int(m.attrib["id"]))
    out = []
    for mix in mixes:
        def rows(tag, required=False):
            el = mix.find(tag)
            if el is None:
                if required:
                    raise ValueError(f"{path}: mix {mix.attrib['id']} lacks <{tag}>")
                return None
            return _xml_rows(el)

        sigma_t = rows("SigmaT")
        sigma_tr = rows("SigmaTR")
        sigma_a = rows("SigmaA", required=True).reshape(G)
        # File stores SigmaS[to, from]; GARNET wants sigma_s[from, to].
        sigma_s = rows("SigmaS", required=True).reshape(G, G).T
        nu_sig_f = rows("NuSigF", required=True).reshape(G)
        chi_el = rows("Chi")
        transport = sigma_tr if sigma_tr is not None else sigma_t
        if transport is None:
            raise ValueError(f"{path}: mix {mix.attrib['id']} needs SigmaT or SigmaTR")
        transport = transport.reshape(G)
        name_el = mix.find("name")
        chi = None
        if chi_el is not None and chi_el.sum() > 0:
            chi = chi_el.reshape(G) / chi_el.sum()
        out.append(Material(
            name=(name_el.text.strip() if name_el is not None and name_el.text
                  else f"mix-{mix.attrib['id']}"),
            diffusion=1.0 / (3.0 * transport),
            sigma_a=sigma_a, nu_sigma_f=nu_sig_f, sigma_s=sigma_s, chi=chi,
            total=sigma_t.reshape(G) if sigma_t is not None else None))
    return out
