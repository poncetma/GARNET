"""Gmsh 2.x reader robustness.

Element lines are ``id type ntags tag... node...``. The reader used to assume
exactly two tags and contiguous node ids: partitioned (4+ tags) or 3-tag files
gave degenerate zero-area cells, a 1-tag file gave 2-node triangles, and
non-contiguous ids raised IndexError. MSH 4 is not read and must say so, and a
2D mesh drawn outside the xy plane must keep its true areas.
"""

import numpy as np
import pytest

from garnet.mesh import read_gmsh


def _msh(ntags, ids=(1, 2, 3, 4), z=(0.0, 0.0, 0.0, 0.0), version="2.2 0 8",
         xy=((0, 0), (1, 0), (1, 1), (0, 1))):
    tags = " ".join(["7"] + ["1"] * (ntags - 1)) if ntags else ""
    nodes = "\n".join(f"{i} {x} {y} {zz}" if len(p) == 2 else f"{i} {p[0]} {p[1]} {p[2]}"
                      for i, p, zz in zip(ids, xy, z) for x, y in [p[:2]])
    a, b, c, d = ids
    return (f"$MeshFormat\n{version}\n$EndMeshFormat\n$Nodes\n4\n{nodes}\n$EndNodes\n"
            f"$Elements\n3\n"
            f"1 1 {ntags} {tags} {a} {b}\n"                       # a line element: ignored
            f"2 2 {ntags} {tags} {a} {b} {c}\n"
            f"3 2 {ntags} {tags} {a} {c} {d}\n$EndElements\n")


@pytest.mark.parametrize("ntags", [0, 1, 2, 3, 4, 6])
def test_any_tag_count(tmp_path, ntags):
    f = tmp_path / "m.msh"
    f.write_text(_msh(ntags))
    mesh = read_gmsh(str(f))
    assert mesh.n_cells == 2
    assert all(len(c) == 3 for c in mesh.cells)
    assert np.allclose(mesh.area, 0.5)
    assert list(mesh.cell_tag) == ([7, 7] if ntags else [0, 0])
    assert len(mesh.faces) == 1 and len(mesh.bfaces) == 4


def test_noncontiguous_node_ids(tmp_path):
    f = tmp_path / "m.msh"
    f.write_text(_msh(2, ids=(10, 20, 35, 41)))
    mesh = read_gmsh(str(f))
    assert mesh.area.sum() == pytest.approx(1.0)
    assert len(mesh.faces) == 1


def test_undefined_node_is_a_clear_error(tmp_path):
    f = tmp_path / "m.msh"
    f.write_text(_msh(2).replace("3 2 2 7 1 1 3 4", "3 2 2 7 1 1 3 99"))
    with pytest.raises(ValueError, match="node 99"):
        read_gmsh(str(f))


def test_msh4_and_binary_are_refused(tmp_path):
    f = tmp_path / "m.msh"
    f.write_text(_msh(2, version="4.1 0 8"))
    with pytest.raises(ValueError, match="MSH 4.1 is not supported"):
        read_gmsh(str(f))
    f.write_text(_msh(2, version="2.2 1 8"))
    with pytest.raises(ValueError, match="binary"):
        read_gmsh(str(f))


def test_second_order_elements_are_refused(tmp_path):
    f = tmp_path / "m.msh"
    f.write_text(_msh(2).replace("2 2 2 7 1 1 2 3", "2 9 2 7 1 1 2 3 1 2 3"))
    with pytest.raises(ValueError, match="6-node triangle"):
        read_gmsh(str(f))


def test_planar_mesh_outside_xy_plane(tmp_path):
    # The unit square drawn in the x-z plane (y = 0) and tilted: dropping z
    # would collapse it; the reader must rotate it into its own plane.
    f = tmp_path / "m.msh"
    xz = ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))
    f.write_text(_msh(2, xy=xz))
    mesh = read_gmsh(str(f))
    assert mesh.area.sum() == pytest.approx(1.0)
    c = np.cos(0.3)
    s = np.sin(0.3)
    tilted = tuple((x, y * c - z * s, y * s + z * c + 5.0) for x, y, z in xz)
    f.write_text(_msh(2, xy=tilted))
    mesh = read_gmsh(str(f))
    assert np.allclose(mesh.area, 0.5)
    assert sum(b[1] for b in mesh.bfaces) == pytest.approx(4.0)


def test_curved_2d_mesh_is_refused(tmp_path):
    f = tmp_path / "m.msh"
    f.write_text(_msh(2, z=(0.0, 0.0, 0.0, 0.4)))
    with pytest.raises(ValueError, match="coplanar"):
        read_gmsh(str(f))
