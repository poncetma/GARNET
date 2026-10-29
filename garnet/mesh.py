"""Unstructured finite-volume diffusion on a Gmsh mesh.

Reads a Gmsh 2.2 ``.msh`` file -- 2D triangle/quad cells, or 3D tetrahedra,
hexahedra and prisms -- and solves the multigroup k-eigenvalue diffusion problem
on the arbitrary geometry it describes, the same job FEMFFUSION or GeN-Foam do
from a mesh rather than from a structured lattice. This is the general-geometry
escape hatch: where the structured Cartesian/hex operators need a lattice, this
one needs only cells, their vertices, and per-cell materials.

The scheme is a cell-centred two-point-flux finite volume: for a face shared by
cells i and j, the coupling is A_face / (d_i/D_i + d_j/D_j) with A_face the
shared face measure (edge length in 2D, polygon area in 3D) and d_i, d_j the
normal distances from the centroids to the face; boundary faces get the
Robin/albedo term (vacuum alpha = 1/2). On non-orthogonal meshes (general
triangulations, tets, hanging nodes) a deferred non-orthogonal correction built
from least-squares cell gradients restores consistency (see
:class:`UnstructuredDiffusionSolver`). The 2D and 3D assemblers produce the
same face quantities, so the operator and solver below are dimension-agnostic.
The connectivity is irregular, but
the within-group operator is still applied matrix-free as a pure *gather*: the
faces are stored as a row-wise ELLPACK adjacency (each cell's neighbours padded
to the maximum degree), and the apply reads each cell's neighbour fluxes and
combines them with per-slot weights -- no scatter, hence no GPU atomics, and a
coalesced write, the same access pattern that makes the structured stencils fast
on CUDA. The single NumPy/CuPy code path runs on CPU or GPU and reuses the
structured solvers' Jacobi/Neumann-PCG, putting the general-geometry track on the
GPU alongside the structured lattices rather than the earlier SciPy sparse direct
factorisation. Power iteration drives the outer fission source.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .backend import asnumpy, device_name, get_backend, synchronize
from .linalg import get_linear_solver, neumann_preconditioner


_VACUUM_ALPHA = 0.5


@dataclass
class Mesh:
    """Cells and faces of a 2D or 3D finite-volume mesh.

    ``area`` holds the cell measure (area in 2D, volume in 3D) and each face its
    measure (edge length in 2D, face area in 3D); the two-point-flux operator
    consumes these identically in either dimension.

    The assemblers also record each face's unit normal (owner -> neighbour for
    interior faces, outward for boundary faces) and centre. The solver needs
    them to split the face distance per side and to correct for
    non-orthogonality; a hand-built Mesh without them falls back to the plain
    centroid-distance two-point flux.
    """

    coords: np.ndarray              # (n_nodes, 2) or (n_nodes, 3)
    cells: list                     # list of node-index tuples (per cell)
    cell_tag: np.ndarray            # (n_cells,) physical/elementary tag per cell
    centroid: np.ndarray            # (n_cells, 2) or (n_cells, 3)
    area: np.ndarray                # (n_cells,) cell measure (2D area / 3D volume)
    faces: list                     # (i, j, face_measure, normal_distance) interior
    bfaces: list                    # (i, face_measure, centroid_to_face_normal_distance) boundary
    face_normal: np.ndarray = None  # (n_faces, dim) unit normal, i -> j
    face_center: np.ndarray = None  # (n_faces, dim) face (or face-overlap) centre
    bface_normal: np.ndarray = None  # (n_bfaces, dim) outward unit normal
    bface_center: np.ndarray = None  # (n_bfaces, dim) boundary face centre

    @property
    def n_cells(self) -> int:
        return len(self.cells)


# Gmsh element type -> node count, split by dimension.
_GMSH_2D = {2: 3, 3: 4}                          # triangle, quad
_GMSH_3D = {4: 4, 5: 8, 6: 6}                     # tetrahedron, hexahedron, prism
# Area/volume element types we recognise but cannot use (higher-order elements,
# pyramids): refused rather than silently dropped from the mesh.
_GMSH_UNSUPPORTED = {7: "5-node pyramid", 9: "6-node triangle", 10: "9-node quad",
                     11: "10-node tetrahedron", 12: "27-node hexahedron",
                     13: "18-node prism", 14: "14-node pyramid", 16: "8-node quad",
                     17: "20-node hexahedron", 18: "15-node prism",
                     19: "13-node pyramid"}


def read_gmsh(path: str) -> Mesh:
    """Parse a Gmsh 2.x ASCII mesh and assemble a Mesh.

    Handles 2D meshes of triangles (type 2) / quads (type 3) and 3D meshes of
    tetrahedra (4) / hexahedra (5) / prisms (6). If any 3D volume element is
    present the mesh is built in 3D (surface elements are ignored); otherwise the
    2D path is used. Each element line is ``id type ntags tag... node...``: the
    tag count is read from the line (0, 1, 2, or the 3+ a partitioned mesh
    writes), the first tag (the physical group) becomes the cell tag, and node
    ids are mapped through the $Nodes table, so they need not be contiguous.
    MSH 4 and binary files are refused (export with ``-format msh22``).
    """
    with open(path) as fh:
        lines = [ln.strip() for ln in fh.read().splitlines()]
    if "$MeshFormat" in lines:
        head = lines[lines.index("$MeshFormat") + 1].split()
        version, file_type = head[0], (head[1] if len(head) > 1 else "0")
        if int(float(version)) != 2:
            raise ValueError(f"{path}: Gmsh MSH {version} is not supported; only "
                             "the ASCII 2.x format is read (export with "
                             "`gmsh -format msh22` or Mesh.MshFileVersion = 2.2)")
        if file_type != "0":
            raise ValueError(f"{path}: binary MSH files are not supported; "
                             "write ASCII (Mesh.Binary = 0)")
    ni = lines.index("$Nodes")
    nn = int(lines[ni + 1])
    coords = np.zeros((nn, 3))
    node_index = {}                              # Gmsh node id -> row of coords
    for k in range(nn):
        t = lines[ni + 2 + k].split()
        node_index[int(t[0])] = k
        coords[k] = (float(t[1]), float(t[2]), float(t[3]))
    ei = lines.index("$Elements")
    ne = int(lines[ei + 1])
    elems = [[int(x) for x in lines[ei + 2 + k].split()] for k in range(ne)]
    for t in elems:
        if t[1] in _GMSH_UNSUPPORTED:
            raise ValueError(f"{path}: element {t[0]} is a {_GMSH_UNSUPPORTED[t[1]]} "
                             f"(Gmsh type {t[1]}); only first-order triangles, "
                             "quads, tetrahedra, hexahedra and prisms are supported")
    is_3d = any(t[1] in _GMSH_3D for t in elems)
    table = _GMSH_3D if is_3d else _GMSH_2D
    cells, tags = [], []
    for t in elems:
        nnode = table.get(t[1])
        if nnode is None:
            continue
        ntags = t[2]
        nodes = t[3 + ntags:]
        if len(nodes) != nnode:
            raise ValueError(f"{path}: element {t[0]} (type {t[1]}, {ntags} tags) "
                             f"lists {len(nodes)} nodes, expected {nnode}")
        try:
            cells.append(tuple(node_index[g] for g in nodes))
        except KeyError as err:
            raise ValueError(f"{path}: element {t[0]} references node {err.args[0]}, "
                             "which is not in $Nodes") from None
        tags.append(t[3] if ntags else 0)        # first tag (physical / assembly id)
    if is_3d:
        return assemble_mesh_3d(coords, cells, tags)
    return assemble_mesh(coords, cells, tags)


def _planar_coords(coords, cells):
    """2D coordinates of a planar 3D surface mesh.

    Gmsh writes x, y, z for every node; a 2D mesh drawn in the xy plane just
    drops z, but one drawn in any other plane must be rotated into it first or
    every area and distance comes out wrong. The plane is fitted (SVD) through
    the nodes the cells use and the nodes are expressed in an orthonormal basis
    of it -- an isometry, so the finite-volume geometry is exact. Non-coplanar
    nodes (a curved shell) are refused.
    """
    used = np.unique(np.fromiter((n for ns in cells for n in ns), dtype=np.int64))
    if used.size == 0:
        return coords[:, :2].copy()
    P = coords[used]
    scale = max(float(np.ptp(P, axis=0).max()), 1e-300)
    if np.ptp(P[:, 2]) <= 1e-9 * scale:
        return coords[:, :2].copy()
    origin = P.mean(0)
    _, sv, vt = np.linalg.svd(P - origin, full_matrices=False)
    if sv[2] > 1e-9 * scale * np.sqrt(len(P)):
        raise ValueError("2D elements are not coplanar (the mesh is a curved "
                         "surface); only planar 2D meshes are supported")
    return (coords - origin) @ vt[:2].T


def _side_distances(normal, center, ci, cj, centroid):
    """Normal distances from the two cell centres to a face, each positive for a
    convex cell; a non-positive one means a non-convex or inverted cell, whose
    two-point flux has no meaning."""
    di = float(normal @ (center - centroid[ci]))
    dj = float(normal @ (centroid[cj] - center)) if cj is not None else 1.0
    if di <= 0.0 or dj <= 0.0:
        raise ValueError(f"cell {ci if di <= 0.0 else cj} has its centroid on the "
                         "far side of one of its faces (non-convex or inverted cell)")
    return di, dj


def _cluster_keys(normals, offsets, scale):
    """Hashable (normal, offset) keys grouping faces that lie in one line/plane."""
    canon = normals.copy()
    for k in range(len(canon)):
        n = canon[k]
        lead = n[np.argmax(np.abs(n) > 1e-9)]
        if lead < 0.0:
            canon[k] = -n
    off = np.einsum("fd,fd->f", canon, offsets)
    return ([tuple(np.round(n, 7)) + (round(o / scale, 7),)
             for n, o in zip(canon, off)], canon)


def _clip_convex(poly, clip):
    """Sutherland-Hodgman: the part of convex polygon ``poly`` inside convex,
    counter-clockwise polygon ``clip`` (both (m, 2) arrays)."""
    out = list(poly)
    for k in range(len(clip)):
        a, b = clip[k], clip[(k + 1) % len(clip)]
        e = b - a
        inp, out = out, []
        if not inp:
            break

        def inside(p):
            return e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]) >= 0.0

        for m in range(len(inp)):
            p, q = inp[m], inp[(m + 1) % len(inp)]
            pin, qin = inside(p), inside(q)
            if pin:
                out.append(p)
            if pin != qin:
                dp, dq = e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]), \
                    e[0] * (q[1] - a[1]) - e[1] * (q[0] - a[0])
                out.append(p + (q - p) * (dp / (dp - dq)))
    return np.array(out)


def _polygon_area_centroid_2d(P):
    """Area and centroid of a simple polygon (shoelace); signed area > 0 for CCW."""
    x, y = P[:, 0], P[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    a2 = float(cr.sum())
    if abs(a2) <= 0.0:
        return 0.0, P.mean(0)
    return 0.5 * a2, np.array([((x + xn) * cr).sum(), ((y + yn) * cr).sum()]) / (3.0 * a2)


def _match_nonconforming(one, normals, centers, measures, owner_centroid, scale, dim,
                         polys=None):
    """Pair one-cell faces that overlap across a non-conforming interface.

    ``one`` lists the faces touched by a single cell. A face of a locally refined
    region meets its coarse neighbour face without sharing its node set (2:1,
    1:3, 1:4, multi-level, or arbitrary non-matching), so the two show up here
    as separate one-cell faces lying in the same line (2D) or plane (3D), with
    their cells on opposite sides and positive overlap. Each such overlap is an
    interior face carrying the overlap measure; whatever part of a one-cell face
    no opposite face covers is a true boundary. Faces are grouped by their
    line/plane, and only groups with cells on both sides (never the outer
    boundary of a convex domain) are searched.

    Returns (pairs, leftovers): pairs = [(fa, fb, measure, centre)], leftovers =
    {f: [(measure, centre), ...]} for every face with a nonzero overlap (faces
    absent from it are wholly boundary).
    """
    if not one:
        return [], {}
    nrm = np.array([normals[f] for f in one])
    ctr = np.array([centers[f] for f in one])
    keys, canon = _cluster_keys(nrm, ctr, scale)
    groups = defaultdict(list)
    for k, key in enumerate(keys):
        groups[key].append(k)
    tol = 1e-9 * scale
    pairs, covered = [], defaultdict(list)
    for members in groups.values():
        if len(members) < 2:
            continue
        n = canon[members[0]]
        side = [float(n @ (owner_centroid[one[k]] - ctr[k])) > 0.0 for k in members]
        if all(side) or not any(side):
            continue
        if dim == 2:
            u = np.array([-n[1], n[0]])                    # direction along the line
            iv = []
            for k in members:
                p, q = polys[one[k]]
                s0, s1 = sorted((float(u @ p), float(u @ q)))
                iv.append((s0, s1, k))
            pos = sorted((iv[m] for m in range(len(members)) if side[m]))
            neg = sorted((iv[m] for m in range(len(members)) if not side[m]))
            a = b = 0
            while a < len(pos) and b < len(neg):
                lo, hi = max(pos[a][0], neg[b][0]), min(pos[a][1], neg[b][1])
                if hi - lo > tol:
                    ka, kb = pos[a][2], neg[b][2]
                    base = ctr[ka] - (u @ ctr[ka]) * u     # point on the line, s = 0
                    mid = base + 0.5 * (lo + hi) * u
                    pairs.append((one[ka], one[kb], hi - lo, mid))
                    covered[one[ka]].append((lo, hi, u))
                    covered[one[kb]].append((lo, hi, u))
                if pos[a][1] < neg[b][1]:
                    a += 1
                else:
                    b += 1
        else:
            # Project to in-plane coordinates and clip convex polygons.
            e1 = np.cross(n, [1.0, 0.0, 0.0])
            if np.linalg.norm(e1) < 0.5:
                e1 = np.cross(n, [0.0, 1.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            poly2 = {}
            for k in members:
                P = polys[one[k]] @ np.c_[e1, e2]
                if _polygon_area_centroid_2d(P)[0] < 0.0:
                    P = P[::-1]
                poly2[k] = P
            pos = [k for m, k in enumerate(members) if side[m]]
            neg = [k for m, k in enumerate(members) if not side[m]]
            neg_box = sorted((poly2[k][:, 0].min(), poly2[k][:, 0].max(), k) for k in neg)
            for ka in pos:
                Pa = poly2[ka]
                xa0, xa1 = Pa[:, 0].min(), Pa[:, 0].max()
                ya0, ya1 = Pa[:, 1].min(), Pa[:, 1].max()
                for x0, x1, kb in neg_box:
                    if x0 > xa1 - tol:
                        break
                    Pb = poly2[kb]
                    if (x1 < xa0 + tol or Pb[:, 1].max() < ya0 + tol
                            or Pb[:, 1].min() > ya1 - tol):
                        continue
                    clip = _clip_convex(Pa, Pb)
                    if len(clip) < 3:
                        continue
                    A, c2 = _polygon_area_centroid_2d(clip)
                    if A <= tol * scale:
                        continue
                    off = float(n @ ctr[ka])
                    c3 = c2[0] * e1 + c2[1] * e2 + off * n
                    pairs.append((one[ka], one[kb], A, c3))
                    covered[one[ka]].append((A, c3))
                    covered[one[kb]].append((A, c3))

    leftovers = {}
    for f, parts in covered.items():
        rest = []
        if dim == 2:
            p, q = polys[f]
            u = parts[0][2]                    # the line direction the s were taken in
            s0, s1 = sorted((float(u @ p), float(u @ q)))
            base = p - (u @ p) * u
            cur = s0
            for lo, hi, _ in sorted(parts, key=lambda iv: iv[0]):
                if lo - cur > tol:
                    rest.append((lo - cur, base + 0.5 * (cur + lo) * u))
                cur = max(cur, hi)
            if s1 - cur > tol:
                rest.append((s1 - cur, base + 0.5 * (cur + s1) * u))
        else:
            A_left = measures[f] - sum(A for A, _ in parts)
            if A_left > 1e-9 * measures[f]:
                c_left = (measures[f] * centers[f]
                          - sum(A * c for A, c in parts)) / A_left
                rest.append((A_left, c_left))
        leftovers[f] = rest
    return pairs, leftovers


def _finish_faces(dim, centroid, interior, boundary, nonconf):
    """Common tail of the 2D and 3D assemblers.

    interior : [(i, j, measure, unit normal, centre)] conforming shared faces.
    boundary : [(i, measure, unit normal, centre)] one-cell faces (normal any sign).
    nonconf  : True to search the one-cell faces for non-conforming overlaps.
    Orients every normal (owner -> neighbour / outward), splits matched
    non-conforming faces, and returns the Mesh face lists and geometry arrays.
    """
    # Length scale for the line/plane grouping tolerances: the extent of the
    # cell and face centres (a single cell has zero centroid extent).
    pts = np.vstack([centroid] + [np.reshape(f[-2], (1, dim)) for f in interior]
                    + [np.reshape(b[3], (1, dim)) for b in boundary])
    scale = float(np.ptp(pts, axis=0).max()) if len(pts) else 0.0
    scale = scale if scale > 0.0 else 1.0
    faces, fn, fc, bfaces, bn, bc = [], [], [], [], [], []

    def add_interior(i, j, A, n, x):
        if n @ (x - centroid[i]) < 0.0:
            n = -n
        di, dj = _side_distances(n, x, i, j, centroid)
        faces.append((i, j, A, di + dj))
        fn.append(n)
        fc.append(x)

    def add_boundary(i, A, n, x):
        if n @ (x - centroid[i]) < 0.0:
            n = -n
        di, _ = _side_distances(n, x, i, None, centroid)
        bfaces.append((i, A, di))
        bn.append(n)
        bc.append(x)

    for i, j, A, n, x in interior:
        add_interior(i, j, A, n, x)
    if nonconf:
        owner = np.array([b[0] for b in boundary], dtype=np.int64)
        normals = [b[2] for b in boundary]
        centers = [b[3] for b in boundary]
        measures = [b[1] for b in boundary]
        polys = [b[4] for b in boundary]
        pairs, leftovers = _match_nonconforming(
            list(range(len(boundary))), normals, centers, measures,
            centroid[owner] if len(owner) else centroid, scale, dim, polys)
        # (owner_centroid is indexed by position in ``one``, which is identity here.)
        for fa, fb, A, x in pairs:
            add_interior(int(owner[fa]), int(owner[fb]), A, normals[fa], x)
        for k, (i, A, n, x, _) in enumerate(boundary):
            if k in leftovers:
                for Ar, xr in leftovers[k]:
                    add_boundary(i, Ar, n, xr)
            else:
                add_boundary(i, A, n, x)
    else:
        for i, A, n, x, _ in boundary:
            add_boundary(i, A, n, x)
    as_arr = lambda v: np.array(v, dtype=float).reshape(len(v), dim)
    return faces, as_arr(fn), as_arr(fc), bfaces, as_arr(bn), as_arr(bc)


def assemble_mesh(coords, cells, tags) -> Mesh:
    """Build a Mesh (centroids, areas, interior/boundary faces) from polygons.

    coords : (n_nodes, 2) node coordinates ((n_nodes, 3) is accepted for a planar
    mesh in any orientation). cells : list of node-index tuples, each a convex
    polygon. tags : one integer per cell (material/assembly id). A face is a
    boundary face when only one cell borders it.

    Nonconforming (locally-refined) interfaces are supported in general: two
    one-cell edges that are collinear, overlap with positive length and have
    their cells on opposite sides are coupled through an interior face carrying
    the overlap length -- the standard conservative two-point-flux treatment.
    That covers 2:1 hanging nodes, 1:3, 1:4, multi-level refinement and
    arbitrary non-matching edges; any uncovered remainder stays a boundary.
    Conforming meshes are unaffected (their one-cell edges are the outer
    boundary, with every cell on the inside).
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 2 and coords.shape[1] == 3:
        coords = _planar_coords(coords, cells)
    cells = [tuple(int(n) for n in c) for c in cells]
    nc = len(cells)
    centroid = np.zeros((nc, 2))
    area = np.zeros(nc)
    edge_cells = defaultdict(list)
    for c, ns in enumerate(cells):
        a_signed, centroid[c] = _polygon_area_centroid_2d(coords[list(ns)])
        area[c] = abs(a_signed)
        if not area[c] > 0.0:
            raise ValueError(f"cell {c} {ns} has zero area (degenerate or repeated nodes)")
        m = len(ns)
        for a in range(m):
            edge_cells[tuple(sorted((ns[a], ns[(a + 1) % m])))].append(c)

    def edge_geom(e):
        p0, p1 = coords[e[0]], coords[e[1]]
        t = p1 - p0
        L = float(np.hypot(*t))
        return L, np.array([t[1], -t[0]]) / L, 0.5 * (p0 + p1), (p0, p1)

    interior, boundary = [], []
    for e, lst in edge_cells.items():
        L, n, x, seg = edge_geom(e)
        if len(lst) == 2:
            interior.append((lst[0], lst[1], L, n, x))
        elif len(lst) == 1:
            boundary.append((lst[0], L, n, x, seg))
        else:
            raise ValueError(f"edge {e} is shared by {len(lst)} cells "
                             "(overlapping or non-manifold mesh)")
    faces, fn, fc, bfaces, bn, bcen = _finish_faces(2, centroid, interior, boundary, True)
    return Mesh(coords=coords, cells=cells, cell_tag=np.asarray(tags), centroid=centroid,
               area=area, faces=faces, bfaces=bfaces, face_normal=fn, face_center=fc,
               bface_normal=bn, bface_center=bcen)


# Local face node-orderings per 3D element type, keyed by node count. Orientation
# is irrelevant here: face areas and cell volumes are taken in magnitude and a
# face is identified by the *set* of its nodes, so any consistent enumeration of
# each element's bounding faces works. (Gmsh 4=tet, 5=hex, 6=prism/wedge.)
_TET_FACES = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
_HEX_FACES = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
              (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
_PRISM_FACES = [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]
_FACE_TEMPLATES = {4: _TET_FACES, 8: _HEX_FACES, 6: _PRISM_FACES}


def _cross3(a, b):
    """Cross product of two 3-vectors (np.cross's generic axis handling costs
    ~20x more per call, and the assembler makes several per cell)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def _poly_area_centroid_3d(pts):
    """Area, area-weighted centroid and unit normal of a planar polygon in 3D
    (fan triangulation; the normal is the area-weighted mean, so a slightly
    warped quad gets its average plane)."""
    p0 = pts[0]
    atot = 0.0
    csum = np.zeros(3)
    nsum = np.zeros(3)
    for i in range(1, len(pts) - 1):
        cr = _cross3(pts[i] - p0, pts[i + 1] - p0)
        a = 0.5 * float(np.sqrt(cr @ cr))
        atot += a
        csum += a * (p0 + pts[i] + pts[i + 1]) / 3.0
        nsum += 0.5 * cr
    nlen = float(np.linalg.norm(nsum))
    normal = nsum / nlen if nlen > 0 else np.zeros(3)
    return atot, (csum / atot if atot > 0 else pts.mean(0)), normal


def _cell_volume_centroid_3d(cell_pts, faces_local):
    """Volume and volume-weighted centroid of a star-convex polyhedron.

    Decomposes into tetrahedra from a seed point (the vertex mean) to a fan
    triangulation of every bounding face. Exact for convex tets/hexes/prisms.
    """
    g = cell_pts.mean(0)
    vtot = 0.0
    csum = np.zeros(3)
    for f in faces_local:
        fp = cell_pts[list(f)]
        p0 = fp[0]
        for i in range(1, len(fp) - 1):
            v = abs(float(_cross3(fp[i] - g, fp[i + 1] - g) @ (p0 - g))) / 6.0
            vtot += v
            csum += v * (g + p0 + fp[i] + fp[i + 1]) / 4.0
    return vtot, (csum / vtot if vtot > 0 else g)


def assemble_mesh_3d(coords, cells, tags) -> Mesh:
    """Build a 3D Mesh (volumes, centroids, interior/boundary faces) from cells.

    coords : (n_nodes, 3) node coordinates. cells : list of node-index tuples,
    each a tetrahedron (4 nodes), prism/wedge (6) or hexahedron (8) in Gmsh
    ordering. tags : one integer per cell (material id). Two cells share a face
    when they share the same set of face nodes; a face touched by one cell is a
    boundary face -- unless it is coplanar with, overlaps, and faces the cell
    across one or more other one-cell faces (a non-conforming interface from
    local refinement, hanging nodes on faces or edges). Those are clipped
    against each other (convex polygon intersection) and coupled through the
    overlap area, as the 2D assembler does with collinear edges.

    The returned Mesh carries the same two-point-flux quantities the 2D path
    does -- so `Mesh.area` holds the cell *volume* and each face its *area* and
    normal centroid-to-centroid (or centroid-to-face) distance -- and drives the
    exact same operator and solver.
    """
    coords = np.asarray(coords, dtype=float)
    cells = [tuple(int(n) for n in c) for c in cells]
    nc = len(cells)
    centroid = np.zeros((nc, 3))
    volume = np.zeros(nc)
    face_cells = defaultdict(list)      # sorted global node key -> [(cell, global face nodes)]
    for c, ns in enumerate(cells):
        tmpl = _FACE_TEMPLATES.get(len(ns))
        if tmpl is None:
            raise ValueError(f"cell {c} has {len(ns)} nodes; expected a tet (4), "
                             "prism (6) or hex (8)")
        cell_pts = coords[list(ns)]
        volume[c], centroid[c] = _cell_volume_centroid_3d(cell_pts, tmpl)
        if not volume[c] > 0.0:
            raise ValueError(f"cell {c} {ns} has zero volume (degenerate or repeated nodes)")
        for fl in tmpl:
            gface = tuple(ns[k] for k in fl)
            face_cells[tuple(sorted(gface))].append((c, gface))

    interior, boundary = [], []
    for lst in face_cells.values():
        gface = lst[0][1]
        pts = coords[list(gface)]
        A, fcent, n = _poly_area_centroid_3d(pts)
        if len(lst) == 2:
            interior.append((lst[0][0], lst[1][0], A, n, fcent))
        elif len(lst) == 1:
            boundary.append((lst[0][0], A, n, fcent, pts))
        else:
            raise ValueError(f"face {lst[0][1]} is shared by {len(lst)} cells "
                             "(non-manifold or overlapping mesh)")
    faces, fn, fc, bfaces, bn, bcen = _finish_faces(3, centroid, interior, boundary, True)
    return Mesh(coords=coords, cells=cells, cell_tag=np.asarray(tags),
                centroid=centroid, area=volume, faces=faces, bfaces=bfaces,
                face_normal=fn, face_center=fc, bface_normal=bn, bface_center=bcen)


@dataclass
class MeshResult:
    k_eff: float
    flux: np.ndarray                # (G, n_cells)
    converged: bool
    outer_iterations: int
    solve_seconds: float
    device: str = "cpu (numpy)"
    inner_iterations: int = 0
    k_history: list = None


def _build_ell(face_i, face_j, n):
    """Row-wise (ELLPACK) adjacency from a symmetric face list.

    Each interior face (i, j) couples both endpoints, so it appears once in row
    i (neighbour j) and once in row j (neighbour i). The rows are padded to the
    maximum cell degree K and stored column-major, (K, n): ``nbr[s, c]`` is the
    s-th neighbour cell of c and ``fidx[s, c]`` the face feeding that slot (-1 on
    padding). Padding neighbours point at the cell itself with a zero weight, so
    they are inert. This lets the operator apply be a *gather* (read each
    neighbour's flux) with a coalesced write, instead of a *scatter* (atomic adds
    on the GPU) -- the same access pattern that makes the structured stencils
    fast on CUDA.
    """
    face_count = face_i.size
    owner = np.concatenate([face_i, face_j])
    neighbor = np.concatenate([face_j, face_i])
    face_indices = np.concatenate([np.arange(face_count), np.arange(face_count)]).astype(np.int64)
    neighbor_count = np.bincount(owner, minlength=n)
    max_neighbors = max(int(neighbor_count.max()) if face_count else 0, 1)
    offset = np.zeros(n, dtype=np.int64)
    if n:
        offset[1:] = np.cumsum(neighbor_count)[:-1]
    order = np.argsort(owner, kind="stable")
    sorted_owner = owner[order]
    slot = np.arange(owner.size) - offset[sorted_owner]           # 0..deg-1 within each row
    neighbors = np.tile(np.arange(n, dtype=np.int64), (max_neighbors, 1))      # (K, n), self-padded
    face_map = np.full((max_neighbors, n), -1, dtype=np.int64)
    if face_count:
        neighbors[slot, sorted_owner] = neighbor[order]
        face_map[slot, sorted_owner] = face_indices[order]
    return neighbors, face_map


class _MeshGroupOperator:
    """Matrix-free within-group FV operator on an unstructured mesh.

    Exposes the same ``apply`` / ``inv_diag`` interface as the structured
    :class:`~garnet.operator.GroupOperator`, so it drops straight into the shared
    Jacobi/Neumann-PCG. The connectivity is an explicit ELLPACK adjacency
    (:func:`_build_ell`) rather than a shifted stencil, but the apply is still a
    pure *gather*: for every cell read its neighbours' flux and combine with the
    per-slot weights, ``A phi = diag*phi - sum_s w[s]*phi[nbr[s]]``. No scatter,
    so no GPU atomics and the write is coalesced. The assembled operator --
    volume-integrated harmonic-mean two-point-flux leakage, plus removal and the
    Robin boundary term on the diagonal -- is symmetric and diagonally dominant,
    hence SPD, so CG converges.
    """

    def __init__(self, xp, nbr, w_ell, diag):
        self.xp = xp
        self.nbr = nbr                # (K, n) int32 neighbour cell per slot
        self.w_ell = w_ell            # (K, n) coupling weight per slot (0 on padding)
        self.diag = diag              # (n,) assembled diagonal
        self.inv_diag = 1.0 / diag    # Jacobi preconditioner

    def apply(self, phi):
        gathered = phi[self.nbr]                      # (K, n) neighbour flux
        return self.diag * phi - (self.w_ell * gathered).sum(axis=0)


class UnstructuredDiffusionSolver:
    """Multigroup k-eigenvalue diffusion FV solver on an arbitrary Gmsh mesh.

    materials      : list of Material (all same group count, up- or down-scatter).
    cell_material  : (n_cells,) index into `materials` for each cell.
    alpha_boundary : Robin coefficient on every boundary face (0.5 = vacuum,
                     0 = reflective, or a custom albedo).
    device         : "auto" (GPU if present) | "gpu" | "cpu".
    precond_degree : Neumann-polynomial preconditioner degree for the inner CG
                     (0 = plain Jacobi, the default), as on the structured
                     solvers.
    linear_solver  : "cg" (default), "gmres", or "bicgstab", as on the
                     structured solvers.
    nonorthogonal_correction : apply the deferred non-orthogonal correction
                     (default True; only active when the mesh is non-orthogonal).

    Face flux. For a face with unit normal n and centre x_f between cells i and
    j, let d_i = n.(x_f - x_i), d_j = n.(x_j - x_f) be the normal distances and
    x_i', x_j' the points at those distances either side of the face along n
    (through the point where the centroid line crosses the face). The flux is
    the two-point flux between those two points,

        J = A (phi(x_i') - phi(x_j')) / (d_i/D_i + d_j/D_j),

    which is exact for a linear flux on any mesh. phi(x_i') = phi_i + g_i.r_i,
    with r_i = x_i' - x_i the (tangential) offset, splits it into an implicit
    symmetric two-point term and an explicit term in the cell gradients g. On an
    orthogonal mesh (rectangles, equilateral triangles, extruded wedges) r = 0
    and the scheme is the classical TPFA. On a skewed one (triangulations,
    hanging nodes, tets) the explicit term is the standard over-relaxed
    non-orthogonal correction: the gradients are least-squares reconstructions
    (exact for linear fields, with the Robin law closing boundary cells) and the
    term is lagged into the source of each inner solve, so the operator the CG
    sees stays symmetric and matrix-free. Without it the bare TPFA is
    inconsistent on such meshes (diagonal-cut squares: -316 pcm at n = 64, not
    converging). Boundary faces use the normal distance d_i, so the Robin term
    is consistent without any correction.
    """

    def __init__(self, mesh: Mesh, materials, cell_material,
                 alpha_boundary=_VACUUM_ALPHA, device="auto",
                 precond_degree=0, dtype=np.float64, linear_solver="cg",
                 nonorthogonal_correction=True):
        self.xp = xp = get_backend(device)
        self.device = device_name(xp)
        self.dtype = dtype
        self._linsolve = get_linear_solver(linear_solver)
        self.mesh = mesh
        self.mats = list(materials)
        self.cm = cm = np.asarray(cell_material)
        self.alpha = float(alpha_boundary)
        if not self.alpha >= 0.0:
            raise ValueError(f"alpha_boundary must be >= 0 (inf = zero flux), got {alpha_boundary!r}")
        self.G = G = self.mats[0].n_groups
        n = mesh.n_cells

        cell_measure = np.asarray(mesh.area, dtype=dtype)
        D = [np.array([self.mats[m].diffusion[g] for m in cm], dtype=dtype) for g in range(G)]
        removal = [np.array([self.mats[m].removal[g] for m in cm], dtype=dtype) for g in range(G)]

        # Face and boundary connectivity as flat arrays (host; moved to device).
        face_owner = np.array([f[0] for f in mesh.faces], dtype=np.int64)
        face_neighbor = np.array([f[1] for f in mesh.faces], dtype=np.int64)
        face_measure = np.array([f[2] for f in mesh.faces], dtype=dtype)
        center_distance = np.array([f[3] for f in mesh.faces], dtype=dtype)
        boundary_owner = np.array([b[0] for b in mesh.bfaces], dtype=np.int64)
        boundary_measure = np.array([b[1] for b in mesh.bfaces], dtype=dtype)
        boundary_distance = np.array([b[2] for b in mesh.bfaces], dtype=dtype)
        # Per-side normal distances: from the face geometry when the assembler
        # recorded it, else (a hand-built Mesh) half the centroid distance each.
        geom = mesh.face_normal is not None and mesh.bface_normal is not None
        cen = np.asarray(mesh.centroid, dtype=float)
        if geom:
            dim = cen.shape[1]
            fnorm = np.asarray(mesh.face_normal, dtype=float).reshape(len(mesh.faces), dim)
            fcen = np.asarray(mesh.face_center, dtype=float).reshape(len(mesh.faces), dim)
            dist_owner = np.einsum("fd,fd->f", fnorm, fcen - cen[face_owner])
            dist_neighbor = np.einsum("fd,fd->f", fnorm, cen[face_neighbor] - fcen)
        else:
            dist_owner = dist_neighbor = 0.5 * center_distance

        # ELLPACK adjacency (geometry, shared across groups); moved to device once.
        nbr, fmap = _build_ell(face_owner, face_neighbor, n)
        nbr_dev = xp.asarray(nbr.astype(np.int32))
        fpad = np.where(fmap >= 0, fmap, 0)                    # safe index for padding

        self.ops = []
        w_host = []
        for g in range(G):
            Dg = D[g]
            diffusion_owner, diffusion_neighbor = Dg[face_owner], Dg[face_neighbor]
            # Series resistance of the two half-cells: the harmonic mean of D
            # weighted by each side's normal distance (equal halves on a
            # uniform orthogonal mesh give the familiar 2 D_i D_j/(D_i+D_j) A/d).
            face_w = face_measure / (dist_owner / diffusion_owner
                                     + dist_neighbor / diffusion_neighbor)
            # per-slot weights (0 where padded); diag from the same face weights.
            w_ell = (np.where(fmap >= 0, face_w[fpad], 0.0).astype(dtype)
                     if face_w.size else np.zeros(fmap.shape, dtype=dtype))
            w_host.append(w_ell)
            diag = removal[g] * cell_measure
            diag = diag + np.bincount(face_owner, weights=face_w, minlength=n)
            diag = diag + np.bincount(face_neighbor, weights=face_w, minlength=n)
            if self.alpha != 0.0:
                Db = Dg[boundary_owner]
                if np.isinf(self.alpha):                 # zero-flux (Dirichlet) limit
                    boundary_weight = Db * boundary_measure / boundary_distance
                else:
                    boundary_weight = (self.alpha * Db * boundary_measure
                                       / (boundary_distance * self.alpha + Db))
                diag = diag + np.bincount(boundary_owner, weights=boundary_weight, minlength=n)
            self.ops.append(_MeshGroupOperator(
                xp, nbr_dev, xp.asarray(w_ell), xp.asarray(diag)))
        self.preconds = [neumann_preconditioner(op.apply, op.inv_diag,
                                                 int(precond_degree))
                         for op in self.ops]

        self._nbr = nbr_dev
        self._noc = None
        if geom and nonorthogonal_correction and face_owner.size:
            self._noc = self._nonorthogonal_setup(
                cen, nbr, fmap, face_owner, fnorm, fcen, dist_owner, dist_neighbor,
                boundary_owner, D, w_host)

        # Source data, device-resident.
        self.area = xp.asarray(cell_measure)
        self.nsf = [xp.asarray(np.array([self.mats[m].nu_sigma_f[g] for m in cm], dtype=dtype))
                    for g in range(G)]
        self.chi = [xp.asarray(np.array([self.mats[m].chi[g] for m in cm], dtype=dtype))
                    for g in range(G)]
        # scattering g'->g (both down- and up-scatter); lagged through the source.
        self.scat = {}
        for gf in range(G):
            for gt in range(G):
                if gt != gf:
                    col = np.array([self.mats[m].sigma_s[gf, gt] for m in cm], dtype=dtype)
                    if np.any(col):
                        self.scat[(gf, gt)] = xp.asarray(col)

    def _nonorthogonal_setup(self, cen, nbr, fmap, face_owner, fnorm, fcen,
                             dist_owner, dist_neighbor, boundary_owner, D, w_host):
        """Per-group device arrays for the deferred non-orthogonal correction,
        or None when the mesh is orthogonal (every centroid line parallel to its
        face normal), where the correction vanishes identically.

        Gradient: g_c = G0[:, c] phi_c + sum_s Gs[:, s, c] phi[nbr[s, c]], the
        least-squares fit to the neighbour differences (rows weighted by
        1/|dx|) plus, for each boundary face, the Robin law applied to the
        linear reconstruction, alpha (phi_c + g.dx_b) + D g.n = 0 (exact for
        a linear flux satisfying the boundary condition, and enough to pin the
        gradient of a corner cell with a single neighbour).
        Correction: the explicit part of the outward face flux,
        w (t g_c + (1 - t) g_nbr) . d_tan per slot, with d_tan the component of
        the centroid line tangential to the face and t the owner's fraction of
        the normal distance. The ELLPACK layout keeps both a pure gather.
        """
        xp, dtype = self.xp, self.dtype
        n, K = nbr.shape[1], nbr.shape[0]
        dim = cen.shape[1]
        valid = fmap >= 0
        fpad = np.where(valid, fmap, 0)
        dx = cen[nbr] - cen[None, :, :]                     # (K, n, dim) slot vectors
        dx[~valid] = 0.0
        nrm = fnorm[fpad]                                   # (K, n, dim), sign irrelevant
        dn = np.einsum("knd,knd->kn", nrm, dx)
        dtan = dx - nrm * dn[..., None]
        dlen = np.linalg.norm(dx, axis=-1)
        rel = np.linalg.norm(dtan, axis=-1) / np.where(valid, dlen, 1.0)
        if not np.any(rel[valid] > 1e-8):
            return None
        # Owner-side fraction t of the normal centroid distance for each slot.
        own_side = face_owner[fpad] == np.arange(n)[None, :]
        d_self = np.where(own_side, dist_owner[fpad], dist_neighbor[fpad])
        t = np.where(valid, d_self / np.where(valid, dist_owner[fpad] + dist_neighbor[fpad], 1.0), 0.0)
        dtan[~valid] = 0.0

        # Interior least-squares rows u_s = dx/|dx|, right-hand side (phi_s - phi_c)/|dx|.
        inv_len = np.where(valid, 1.0 / np.where(valid, dlen, 1.0), 0.0)
        u = dx * inv_len[..., None]                         # (K, n, dim)
        M_int = np.einsum("kni,knj->nij", u, u)
        mesh = self.mesh
        bnorm = np.asarray(mesh.bface_normal, dtype=float).reshape(len(mesh.bfaces), dim)
        bdx = np.asarray(mesh.bface_center, dtype=float).reshape(len(mesh.bfaces), dim) \
            - cen[boundary_owner]
        blen = np.linalg.norm(bdx, axis=1)
        alpha = self.alpha
        out = []
        for g in range(self.G):
            Db = D[g][boundary_owner].astype(float)
            if np.isinf(alpha):
                a_row = bdx / blen[:, None]
                beta = -1.0 / blen
            else:
                s = alpha * blen + Db
                a_row = (alpha * bdx + Db[:, None] * bnorm) / s[:, None]
                beta = -alpha / s
            M = M_int.copy()
            np.add.at(M, boundary_owner, np.einsum("bi,bj->bij", a_row, a_row))
            bvec = np.zeros((n, dim))
            np.add.at(bvec, boundary_owner, a_row * beta[:, None])
            Minv = np.linalg.pinv(M)                        # (n, dim, dim)
            Gs = np.einsum("nij,knj->ikn", Minv, u * inv_len[..., None])    # (dim, K, n)
            G0 = -Gs.sum(axis=1) + np.einsum("nij,nj->in", Minv, bvec)     # (dim, n)
            w = w_host[g].astype(float)
            corrA = np.einsum("kn,kn,knd->dn", w, t, dtan)                # (dim, n)
            corrB = np.einsum("kn,kn,knd->dkn", w, 1.0 - t, dtan)         # (dim, K, n)
            out.append(tuple(xp.asarray(a.astype(dtype)) for a in (G0, Gs, corrA, corrB)))
        return out

    def _correction(self, g, phi):
        """Explicit (non-orthogonal) part of the leakage of group g at flux phi."""
        G0, Gs, corrA, corrB = self._noc[g]
        nb = self._nbr
        grad = G0 * phi + (Gs * phi[nb]).sum(axis=1)                  # (dim, n)
        return (corrA * grad).sum(axis=0) + (corrB * grad[:, nb]).sum(axis=(0, 1))

    def solve(self, tol_k=1e-7, tol_source=1e-8, max_outer=1000,
              inner_rtol_floor=1e-11, anderson_depth=8) -> MeshResult:
        """Power iteration on the fission source, Anderson-accelerated.

        The fission-source map ``F -> (1/k) sum_g nuSigma_f,g A_g^-1 chi_g F`` is a
        fixed point whose plain power iteration converges at the dominance ratio;
        for a loosely-coupled core (ratio near 1) that is hundreds of outers.
        Anderson acceleration mixes a short history of source residuals (the same
        scheme used in the transient step and the SPH solve), collapsing the slow
        modes so the eigenvector converges in far fewer outers. ``anderson_depth``
        <= 1 recovers the plain power iteration.
        """
        xp, G = self.xp, self.G
        synchronize(xp)
        t0 = time.perf_counter()
        n = self.mesh.n_cells
        area = self.area
        phi = [xp.ones(n, dtype=self.dtype) for _ in range(G)]
        k = 1.0
        F = sum(self.nsf[g] * phi[g] for g in range(G))
        F = F * (n / float(xp.sum(F)))                    # normalize source, sum = n
        conv = False
        inner_total = 0
        src_err = 1.0
        k_hist = []
        hist = []                                         # (F_in, raw_iterate) for Anderson
        for outer in range(1, max_outer + 1):
            # Inner CG tolerance tracks the *outer residual* (src_err): loose early,
            # tight late. It must not be tied to tol_source -- doing so pinned the
            # inner solve loose whenever tol_source was set large (converge on k
            # only), so k stalled at the inner-solve noise floor and Anderson drifted.
            rtol = min(1e-3, max(0.1 * src_err, inner_rtol_floor))
            for g in range(G):
                q = self.chi[g] / k * F * area
                for (gf, gt), col in self.scat.items():
                    if gt == g:
                        q = q + col * phi[gf] * area
                if self._noc is not None:
                    # Non-orthogonal flux correction, lagged at the latest iterate.
                    q = q - self._correction(g, phi[g])
                phi[g], n_it = self._linsolve(self.ops[g].apply, q, phi[g],
                                              self.ops[g].inv_diag, xp,
                                              rtol=rtol, precond=self.preconds[g])
                inner_total += n_it

            Fn = sum(self.nsf[g] * phi[g] for g in range(G))
            total_new = float(xp.sum(Fn))
            k_new = k * total_new / n                     # F has sum n
            g_F = Fn * (n / total_new)                    # raw power iterate, sum = n
            r = g_F - F
            src_err = float(xp.sqrt(xp.sum(r * r) / xp.sum(g_F ** 2)))
            dk = abs(k_new - k)
            k = k_new
            k_hist.append(k)
            if dk < tol_k and src_err < tol_source and outer > 5:
                F = g_F
                conv = True
                break

            # Anderson update: F <- residual-minimizing mix of the recent iterates.
            F_new = g_F
            if anderson_depth > 1:
                hist.append((F, g_F))
                if len(hist) > anderson_depth:
                    hist.pop(0)
                if len(hist) >= 2:
                    res = [Gj - Sj for Sj, Gj in hist]
                    dres = [res[i] - res[-1] for i in range(len(res) - 1)]
                    m = len(dres)
                    A = np.array([[float(xp.sum(dres[i] * dres[j])) for j in range(m)]
                                  for i in range(m)])
                    b = np.array([-float(xp.sum(dres[i] * res[-1])) for i in range(m)])
                    A[np.diag_indices(m)] += 1e-12 * (np.trace(A) + 1e-300)
                    try:
                        gamma = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        gamma = None
                    if gamma is not None and np.all(np.abs(gamma) < 1e4):
                        for j in range(m):
                            F_new = F_new + float(gamma[j]) * (hist[j][1] - hist[-1][1])
            F = F_new * (n / float(xp.sum(F_new)))        # renormalize, sum = n

        synchronize(xp)
        flux = np.array([asnumpy(phi[g]) for g in range(G)])
        return MeshResult(k_eff=k, flux=flux, converged=conv, outer_iterations=outer,
                          solve_seconds=time.perf_counter() - t0, device=self.device,
                          inner_iterations=inner_total, k_history=k_hist)
