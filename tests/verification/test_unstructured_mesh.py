"""Unstructured finite-volume solver: cross-solver equivalence and exact
invariants.

On a Cartesian grid of quads the unstructured FV must reproduce the
structured Cartesian solver (same discretization, same mesh), and reflective
boundaries must give k = k_inf regardless of geometry. The VVER-440 Gmsh-mesh
validation lives in tests/validation/test_vver440.py.
"""

import numpy as np
import pytest

from garnet import (DiffusionEigenSolver, Grid, ONE_GROUP_DEMO, PWR_TWO_GROUP,
                   k_bare_box, k_infinite)
from garnet.mesh import (UnstructuredDiffusionSolver, assemble_mesh,
                        assemble_mesh_3d, read_gmsh)

# A large Robin coefficient makes the boundary flux vanish (Dirichlet limit),
# so a bare box converges to the zero-flux analytic k_bare_box.
_ZERO_FLUX = 1e8


def _hex_grid(nx, ny, nz, Lx, Ly, Lz):
    """A structured nx*ny*nz grid of hexahedra as an unstructured 3D mesh."""
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    nid, coords = {}, []

    def gid(i, j, k):
        if (i, j, k) not in nid:
            nid[(i, j, k)] = len(coords); coords.append((i * dx, j * dy, k * dz))
        return nid[(i, j, k)]

    cells = [(gid(i, j, k), gid(i + 1, j, k), gid(i + 1, j + 1, k), gid(i, j + 1, k),
              gid(i, j, k + 1), gid(i + 1, j, k + 1), gid(i + 1, j + 1, k + 1), gid(i, j + 1, k + 1))
             for i in range(nx) for j in range(ny) for k in range(nz)]
    return assemble_mesh_3d(coords, cells, [0] * len(cells))


def _tet_cube(n, L):
    """An n^3 lattice with each cube split into 6 tetrahedra (a mesh of tets)."""
    dx = L / n
    nid, coords = {}, []

    def gid(i, j, k):
        if (i, j, k) not in nid:
            nid[(i, j, k)] = len(coords); coords.append((i * dx, j * dx, k * dx))
        return nid[(i, j, k)]

    tets = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
            (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]   # 6-tet cube split
    cells = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                corner = [gid(i, j, k), gid(i + 1, j, k), gid(i + 1, j + 1, k),
                          gid(i, j + 1, k), gid(i, j, k + 1), gid(i + 1, j, k + 1),
                          gid(i + 1, j + 1, k + 1), gid(i, j + 1, k + 1)]
                cells += [tuple(corner[t] for t in tet) for tet in tets]
    return assemble_mesh_3d(coords, cells, [0] * len(cells))


def _cartesian_quad_mesh(n, L):
    dx = L / n
    nid = {}
    coords = []

    def gid(i, j):
        if (i, j) not in nid:
            nid[(i, j)] = len(coords)
            coords.append((i * dx, j * dx))
        return nid[(i, j)]

    cells = [(gid(i, j), gid(i + 1, j), gid(i + 1, j + 1), gid(i, j + 1))
             for i in range(n) for j in range(n)]
    return assemble_mesh(coords, cells, [0] * len(cells))


def test_unstructured_matches_structured_on_quads():
    # Bare homogeneous square, vacuum boundary. The unstructured FV on a quad
    # grid must equal the structured Cartesian FV (2D: reflective z) to tight
    # tolerance -- same discretization, same mesh.
    n, L = 24, 90.0
    mesh = _cartesian_quad_mesh(n, L)
    k_u = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], np.zeros(mesh.n_cells, int),
                                      alpha_boundary=0.5).solve(tol_k=1e-9).k_eff
    k_s = DiffusionEigenSolver(
        Grid(shape=(n, n, 1), size=(L, L, L / n)), ONE_GROUP_DEMO,
        bc=("vacuum", "vacuum", "reflective"), device="cpu",
    ).solve(tol_k=1e-9, tol_source=1e-8).k_eff
    assert k_u == pytest.approx(k_s, abs=2e-4)


def test_unstructured_matches_tri_on_hpmr_staircase():
    # The two HP-MR paths -- structured triangular lattice (TriDiffusionEigenSolver)
    # and the assembled unstructured mesh (UnstructuredDiffusionSolver) -- rasterize
    # the SAME triangles with the same centroid-painted absorber, so they must
    # produce the same k. This ties the general-geometry solver to the fast
    # structured one on the geometry that actually matters, drums in and out.
    from garnet.benchmarks.hpmr import build_hpmr2d, hpmr_locally_refined_mesh
    from garnet.tri import TriDiffusionEigenSolver

    for angle in (0.0, 120.0):
        p = build_hpmr2d(refine=4, drum_angle_deg=angle, absorber="raster")
        k_tri = TriDiffusionEigenSolver(
            p.grid, p.materials, p.material_map, active=p.active, mask_bc=p.mask_bc
        ).solve(tol_k=1e-9, tol_source=1e-8).k_eff

        mesh, cm, mats, alpha = hpmr_locally_refined_mesh(
            refine=4, drum_angle_deg=angle, refine_drums=False)
        k_mesh = UnstructuredDiffusionSolver(mesh, mats, cm, alpha).solve(tol_k=1e-9).k_eff

        assert k_tri == pytest.approx(k_mesh, abs=1e-6), (angle, k_tri, k_mesh)


def _dense_operator(mesh, cm, mats, alpha):
    """The within-group group-0 operator as an explicit dense matrix, assembled
    straight from the face/boundary lists (an independent reference for the
    matrix-free apply)."""
    n = mesh.n_cells
    D = np.array([mats[m].diffusion[0] for m in cm])
    A = np.zeros((n, n))
    for f, (i, j, L, d) in enumerate(mesh.faces):
        # series half-cell resistances along the face normal
        nf, x = mesh.face_normal[f], mesh.face_center[f]
        di, dj = nf @ (x - mesh.centroid[i]), nf @ (mesh.centroid[j] - x)
        assert di + dj == pytest.approx(d)
        w = L / (di / D[i] + dj / D[j])
        A[i, i] += w; A[j, j] += w
        A[i, j] -= w; A[j, i] -= w
    for i, L, db in mesh.bfaces:
        A[i, i] += alpha * D[i] * L / (db * alpha + D[i])
    A[np.diag_indices(n)] += np.array([mats[m].removal[0] for m in cm]) * mesh.area
    return A


def _assert_apply_matches_dense(mesh, cm, mats, alpha):
    solver = UnstructuredDiffusionSolver(mesh, mats, cm, alpha, device="cpu")
    A = _dense_operator(mesh, cm, mats, alpha)
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = rng.standard_normal(mesh.n_cells)
        assert np.allclose(solver.ops[0].apply(x), A @ x, rtol=0, atol=1e-9)


def test_matrix_free_apply_equals_assembled_operator():
    # The matrix-free within-group apply (a gather over the ELLPACK adjacency)
    # must equal an explicitly assembled dense operator built from the same
    # face/boundary weights -- this pins the gather independent of the eigensolve.
    # A locally-refined mesh exercises the 2:1 hanging-node faces too.
    from garnet.benchmarks.hpmr import hpmr_locally_refined_mesh
    mesh, cm, mats, alpha = hpmr_locally_refined_mesh(
        refine=3, drum_angle_deg=90.0, refine_drums=True)
    _assert_apply_matches_dense(mesh, cm, mats, alpha)


def test_matrix_free_apply_equals_assembled_operator_3d():
    # Same equivalence in 3D, on a mesh mixing hexes and tetrahedra so the
    # polygon face areas (quad and triangle) both feed the operator correctly --
    # pins the 3D face-weight assembly, not only the volumes the geometry tests
    # already check.
    hexes = _hex_grid(3, 3, 3, 15.0, 15.0, 15.0)
    tets = _tet_cube(2, 10.0)
    for mesh in (hexes, tets):
        cm = np.zeros(mesh.n_cells, int)
        _assert_apply_matches_dense(mesh, cm, [ONE_GROUP_DEMO], 0.5)


def test_unstructured_reflective_is_kinf():
    # Reflective (alpha=0) on all faces -> k = k_inf, independent of geometry.
    mesh = _cartesian_quad_mesh(10, 50.0)
    res = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], np.zeros(mesh.n_cells, int),
                                      alpha_boundary=0.0).solve(tol_k=1e-10)
    assert res.k_eff == pytest.approx(k_infinite(ONE_GROUP_DEMO), abs=1e-6)


def test_3d_cell_volumes_and_faces():
    # A 2x2x2 grid of unit hexes: each volume 1, total 8, and the six outer
    # sides carry 4 boundary faces each (24), the rest interior.
    mesh = _hex_grid(2, 2, 2, 2.0, 2.0, 2.0)
    assert mesh.n_cells == 8
    assert np.allclose(mesh.area, 1.0)
    assert mesh.area.sum() == pytest.approx(8.0)
    assert len(mesh.bfaces) == 24                       # 6 faces * 4 per face
    assert all(A == pytest.approx(1.0) for _, A, _ in mesh.bfaces)   # unit face area
    # a single regular tetrahedron: volume = 1/6 of the unit corner box
    tet = assemble_mesh_3d([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                           [(0, 1, 2, 3)], [0])
    assert tet.area[0] == pytest.approx(1.0 / 6.0)
    assert len(tet.bfaces) == 4 and len(tet.faces) == 0


def test_unstructured_3d_matches_structured_on_hexes():
    # Bare homogeneous cube, vacuum boundary: the 3D unstructured FV on a hex
    # grid must equal the structured Cartesian solver on the same grid.
    n, L = 12, 60.0
    mesh = _hex_grid(n, n, n, L, L, L)
    k_u = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], np.zeros(mesh.n_cells, int),
                                      alpha_boundary=0.5).solve(tol_k=1e-9).k_eff
    k_s = DiffusionEigenSolver(Grid(shape=(n, n, n), size=(L, L, L)), ONE_GROUP_DEMO,
                               bc="vacuum", device="cpu").solve(tol_k=1e-9, tol_source=1e-8).k_eff
    assert k_u == pytest.approx(k_s, abs=1e-5)


def test_unstructured_3d_reflective_is_kinf():
    mesh = _hex_grid(6, 6, 6, 30.0, 30.0, 30.0)
    res = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], np.zeros(mesh.n_cells, int),
                                      alpha_boundary=0.0).solve(tol_k=1e-10)
    assert res.k_eff == pytest.approx(k_infinite(ONE_GROUP_DEMO), abs=1e-6)


def test_unstructured_3d_converges_to_analytic_second_order():
    # Bare homogeneous box with vanishing boundary flux: the 3D FV k_eff must
    # converge to the exact analytic k_bare_box at second order in the mesh
    # spacing (error drops ~4x when the cell count per side doubles) -- this
    # checks the discretization error itself, not just cross-solver agreement.
    L = 90.0
    k_exact = k_bare_box(ONE_GROUP_DEMO, (L, L, L))
    err = {}
    for n in (16, 32):
        mesh = _hex_grid(n, n, n, L, L, L)
        k = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], np.zeros(mesh.n_cells, int),
                                        alpha_boundary=_ZERO_FLUX).solve(tol_k=1e-9,
                                                                         tol_source=1e-8).k_eff
        err[n] = abs(k - k_exact)
    assert err[32] < 2e-4
    ratio = err[16] / err[32]
    assert 3.5 < ratio < 4.5, f"expected ~4x error drop (2nd order), got {ratio:.2f} ({err})"


def test_unstructured_3d_two_group_matches_structured():
    # Multigroup (fission + down-scatter) in 3D: the hex-mesh solver must equal
    # the structured Cartesian solver on the same grid, vacuum boundary.
    n, L = 12, 60.0
    mesh = _hex_grid(n, n, n, L, L, L)
    k_u = UnstructuredDiffusionSolver(mesh, [PWR_TWO_GROUP], np.zeros(mesh.n_cells, int),
                                      alpha_boundary=0.5).solve(tol_k=1e-9, tol_source=1e-8).k_eff
    k_s = DiffusionEigenSolver(Grid(shape=(n, n, n), size=(L, L, L)), PWR_TWO_GROUP,
                               bc="vacuum", device="cpu").solve(tol_k=1e-9, tol_source=1e-8).k_eff
    assert k_u == pytest.approx(k_s, abs=1e-5)


def test_tet_mesh_reflective_is_kinf():
    # Physics on tetrahedra (not just their volumes): a cube split into tets,
    # reflective on all sides -> k = k_inf, independent of the (skewed) cell
    # shapes, so the tet face coupling and volumes are mutually consistent.
    mesh = _tet_cube(4, 20.0)
    res = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], np.zeros(mesh.n_cells, int),
                                      alpha_boundary=0.0).solve(tol_k=1e-10)
    assert res.k_eff == pytest.approx(k_infinite(ONE_GROUP_DEMO), abs=1e-6)


def test_mixed_element_mesh_reflective_is_kinf():
    # One hexahedron beside two prisms (a square split into two triangles, both
    # extruded), sharing a quad interface. Exercises face de-duplication across
    # element types -- a hex quad face matching a prism side face -- which the
    # single-type meshes never hit. Reflective -> k_inf, and the volumes sum.
    pts = [(0, 0), (1, 0), (1, 1), (0, 1), (2, 0), (2, 1)]     # A B C D E F
    coords = [(x, y, z) for z in (0.0, 1.0) for (x, y) in pts]  # level0: 0-5, level1: 6-11
    hexc = (0, 1, 2, 3, 6, 7, 8, 9)                            # ABCD extruded
    wedge1 = (1, 4, 5, 7, 10, 11)                              # BEF extruded
    wedge2 = (1, 5, 2, 7, 11, 8)                               # BFC extruded
    mesh = assemble_mesh_3d(coords, [hexc, wedge1, wedge2], [0, 0, 0])
    assert mesh.area.sum() == pytest.approx(2.0)               # 1 + 0.5 + 0.5
    assert len(mesh.faces) == 2                                # hex|wedge2 and wedge1|wedge2
    res = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], np.zeros(mesh.n_cells, int),
                                      alpha_boundary=0.0).solve(tol_k=1e-10)
    assert res.k_eff == pytest.approx(k_infinite(ONE_GROUP_DEMO), abs=1e-6)


def test_read_gmsh_routes_3d_tets(tmp_path):
    # Two tetrahedra sharing the face {2,3,4}: read_gmsh must detect the 3D
    # volume elements, build a 3D Mesh, and find one interior face.
    msh = tmp_path / "twotet.msh"
    msh.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n5\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n5 1 1 1\n$EndNodes\n"
        "$Elements\n2\n1 4 2 1 1 1 2 3 4\n2 4 2 1 1 2 3 4 5\n$EndElements\n")
    mesh = read_gmsh(str(msh))
    assert mesh.n_cells == 2
    assert mesh.coords.shape[1] == 3
    assert np.all(mesh.area > 0)
    assert len(mesh.faces) == 1                         # the shared face {2,3,4}


# ---- non-conforming interfaces (general hanging nodes) ----------------------

def _nonconforming_squares(levels, h=10.0, ny=3):
    """Columns of squares, column c refined 1:levels[c] (so neighbouring
    columns meet at 1:3, 1:4, multi-level ... hanging nodes). Returns
    (coords, cells)."""
    nodes, coords, cells = {}, [], []

    def node(x, y):
        k = (round(x, 9), round(y, 9))
        if k not in nodes:
            nodes[k] = len(coords)
            coords.append((x, y))
        return nodes[k]

    for c, r in enumerate(levels):
        s = h / r
        for i in range(r):
            for j in range(ny * r):
                x, y = c * h + i * s, j * s
                cells.append((node(x, y), node(x + s, y), node(x + s, y + s), node(x, y + s)))
    return np.array(coords), cells


def _conforming_equivalent(coords, cells):
    """Insert every node lying inside a cell edge into that cell's polygon:
    the same geometry, now conforming (polygons with collinear vertices)."""
    out = []
    for ns in cells:
        poly = []
        for a in range(len(ns)):
            p, q = coords[ns[a]], coords[ns[(a + 1) % len(ns)]]
            e = q - p
            L2 = e @ e
            t = (coords - p) @ e / L2
            cr = np.abs(e[0] * (coords[:, 1] - p[1]) - e[1] * (coords[:, 0] - p[0]))
            inside = np.where((t > 1e-9) & (t < 1 - 1e-9) & (cr < 1e-9 * L2))[0]
            poly.append(ns[a])
            poly += [int(k) for k in inside[np.argsort(t[inside])]]
        out.append(tuple(poly))
    return out


@pytest.mark.parametrize("levels", [(1, 3), (1, 4), (1, 2, 4), (3, 1, 2), (2, 3)])
def test_nonconforming_matches_conforming_equivalent_vacuum(levels):
    # With vacuum boundaries every lost coarse-fine coupling would act as an
    # internal vacuum surface; the reflective k_inf check cannot see that.
    # Only 2:1 midpoints used to be matched: 1:3, 1:4, multi-level and 2:3
    # interfaces were silently cut.
    coords, cells = _nonconforming_squares(levels)
    nc_mesh = assemble_mesh(coords, cells, [0] * len(cells))
    cf_mesh = assemble_mesh(coords, _conforming_equivalent(coords, cells), [0] * len(cells))
    assert len(nc_mesh.faces) == len(cf_mesh.faces)
    assert sum(b[1] for b in nc_mesh.bfaces) == pytest.approx(
        2 * (len(levels) + 3) * 10.0)                   # outer perimeter only
    cm = np.zeros(len(cells), int)
    k = [UnstructuredDiffusionSolver(m, [ONE_GROUP_DEMO], cm, alpha_boundary=0.5,
                                     device="cpu").solve(tol_k=1e-11, tol_source=1e-10).k_eff
         for m in (nc_mesh, cf_mesh)]
    assert k[0] == pytest.approx(k[1], abs=1e-9)


def _hex_box(x0, x1, nx, y0, y1, ny, z0, z1, nz, nodes, coords):
    def node(x, y, z):
        key = (round(x, 9), round(y, 9), round(z, 9))
        if key not in nodes:
            nodes[key] = len(coords)
            coords.append((x, y, z))
        return nodes[key]
    xs, ys, zs = (np.linspace(a, b, m + 1) for a, b, m in ((x0, x1, nx), (y0, y1, ny), (z0, z1, nz)))
    cells = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                a, b = xs[i], xs[i + 1]
                c, d = ys[j], ys[j + 1]
                e, f = zs[k], zs[k + 1]
                cells.append((node(a, c, e), node(b, c, e), node(b, d, e), node(a, d, e),
                              node(a, c, f), node(b, c, f), node(b, d, f), node(a, d, f)))
    return cells


def test_nonconforming_3d_faces_are_coupled():
    # A 2x2x2 block of coarse hexes beside a 6x6x6 block of fine ones (1:3 in
    # both face directions): every coarse face on the interface meets 9 fine
    # faces through clipped overlaps, conserving area, and no internal
    # boundary is left. Reflective -> k_inf; vacuum k close to the all-fine one.
    nodes, coords = {}, []
    cells = (_hex_box(0, 20, 2, 0, 20, 2, 0, 20, 2, nodes, coords)
             + _hex_box(20, 40, 6, 0, 20, 6, 0, 20, 6, nodes, coords))
    mesh = assemble_mesh_3d(coords, cells, [0] * len(cells))
    coarse_fine = [f for f in mesh.faces if (f[0] < 8) != (f[1] < 8)]
    assert len(coarse_fine) == 4 * 9
    assert sum(f[2] for f in coarse_fine) == pytest.approx(400.0)
    assert sum(b[1] for b in mesh.bfaces) == pytest.approx(2 * (800 + 800 + 400))
    cm = np.zeros(mesh.n_cells, int)
    k_inf = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], cm, alpha_boundary=0.0,
                                        device="cpu").solve(tol_k=1e-11).k_eff
    assert k_inf == pytest.approx(k_infinite(ONE_GROUP_DEMO), abs=1e-8)
    def k_vac(a, b):
        nodes, coords = {}, []
        cells = (_hex_box(0, 20, a, 0, 20, a, 0, 20, a, nodes, coords)
                 + _hex_box(20, 40, b, 0, 20, b, 0, 20, b, nodes, coords))
        m = assemble_mesh_3d(coords, cells, [0] * len(cells))
        return UnstructuredDiffusionSolver(m, [ONE_GROUP_DEMO], np.zeros(m.n_cells, int),
                                           alpha_boundary=0.5, device="cpu").solve(tol_k=1e-11).k_eff
    # Vacuum: between the all-coarse and all-fine meshes (a cut interface
    # would act as an internal vacuum and drop k far below both), and the
    # mirror image of the mesh gives the same k.
    k_nc = k_vac(2, 6)
    assert k_vac(6, 6) < k_nc < k_vac(2, 2)
    assert k_nc == pytest.approx(k_vac(6, 2), abs=1e-8)


# ---- low-severity geometry / boundary items ---------------------------------

def test_infinite_alpha_is_the_zero_flux_limit():
    mesh = _cartesian_quad_mesh(8, 40.0)
    cm = np.zeros(mesh.n_cells, int)
    k_inf_alpha = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], cm, alpha_boundary=np.inf,
                                              device="cpu").solve(tol_k=1e-11).k_eff
    k_big = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], cm, alpha_boundary=1e12,
                                        device="cpu").solve(tol_k=1e-11).k_eff
    assert np.isfinite(k_inf_alpha)
    assert k_inf_alpha == pytest.approx(k_big, abs=1e-8)
    with pytest.raises(ValueError, match="alpha_boundary"):
        UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], cm, alpha_boundary=-1.0)


def test_quad_centroid_is_the_area_centroid():
    # A trapezoid: its vertex mean (1.5, 0.5) is not its centroid.
    mesh = assemble_mesh([(0, 0), (4, 0), (1, 1), (0, 1)], [(0, 1, 2, 3)], [0])
    assert mesh.area[0] == pytest.approx(2.5)
    # composite: rectangle [0,1]x[0,1] (area 1) + triangle (1,0),(4,0),(1,1) (area 1.5)
    cx = (1 * 0.5 + 1.5 * 2.0) / 2.5
    cy = (1 * 0.5 + 1.5 * (1 / 3)) / 2.5
    assert mesh.centroid[0] == pytest.approx([cx, cy])
