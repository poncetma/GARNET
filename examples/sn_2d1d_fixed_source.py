"""Exercise the CPU planar S_N / axial diffusion prototype with supplied data.

Install GARNET from the checkout, then run::

    python3 examples/sn_2d1d_fixed_source.py --compare-3d
    python3 examples/sn_2d1d_fixed_source.py --case c5g7 --compare-3d
    python3 examples/sn_2d1d_fixed_source.py --eigenvalue --case c5g7 --compare-3d

The C5G7 option uses existing seven-group material tables in a SMALL DERIVED
triangular extrusion, not the official C5G7 geometry. The default calculation
has a prescribed source; --eigenvalue instead performs fission power iteration.
Full 3D S_N differences measure model/discretization differences, not a failure
of the hybrid equations' algebraic verification. No nuclear-data files needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy

from garnet import Material, TriGrid, TriSNTransportSolver
from garnet.sn_2d1d import SN2D1DSolver


def make_problem(case="synthetic", layers=4, height=6.0):
    """Small radial/axial heterogeneity, with an absorber in the upper half."""
    grid = TriGrid((3, 3, 2, layers), side=1.5, height=height)
    if case == "c5g7":
        from garnet.benchmarks.c5g7 import c5g7_materials
        tables = c5g7_materials()
        materials = [tables[name] for name in ("UO2", "Water", "Control Rod")]
    else:
        materials = []
        for name, absorption, scattering in (
                ("medium", [.08, .11], [[.18, .04], [.007, .27]]),
                ("reflector", [.02, .03], [[.25, .05], [.003, .35]]),
                ("absorber", [.95, 1.285], [[.20, .05], [.015, .30]])):
            total = np.asarray(absorption) + np.sum(scattering, axis=1)
            materials.append(Material(name=name, diffusion=1/(3*total), total=total,
                                       sigma_a=absorption, sigma_s=scattering,
                                       nu_sigma_f=[0., 0.]))
    material_map = np.zeros(grid.shape, int)
    material_map[:, 0, :, :] = 1
    material_map[1, 1, :, layers//2:] = 2
    source = np.zeros((materials[0].n_groups,) + grid.shape)
    source[0] = np.linspace(.5, 1.0, layers)
    return grid, materials, material_map, source


def full_3d_balance(transport, flux, source):
    """Check the original transport equations using their conservative currents."""
    grid = transport.grid
    defects, scales = [], []
    current_sweep = (transport._sweep_currents_scb_3d if transport.scheme == "scb"
                     else transport._sweep_currents_3d)
    for group in range(transport.G):
        complete_source = source[group] + transport.ss_self[group] * flux[group]
        for from_group in range(transport.G):
            transfer = transport.scatter[from_group][group]
            if transfer is not None:
                complete_source += transfer * flux[from_group]
        _, currents = current_sweep(group, complete_source.ravel())
        divergence = np.zeros(grid.n_cells)
        for face, current in zip(transport._faces_3d(), currents):
            cells, _, _, _, _, face_to_volume, _, _ = face
            np.add.at(divergence, cells.ravel(), (current * face_to_volume).ravel())
        collision = transport.st[group] * flux[group]
        divergence = divergence.reshape(grid.shape)
        defects.append(divergence + collision - complete_source)
        scales.append(np.abs(divergence) + np.abs(collision) + np.abs(complete_source))
    residual = np.max(np.abs(defects)) / max(float(np.max(scales)), 1e-300)
    return float(residual), float(np.sum(defects) * grid.cell_volume)


def full_3d_reference(grid, materials, material_map, source, scheme, *, tol=1e-10):
    """Reuse original full-3D group solves and current folds at matching angles."""
    started = time.perf_counter()
    transport = TriSNTransportSolver(
        grid, materials, material_map, n_polar=4, n_azi=8, scheme=scheme,
        require_fissile=False, acceleration="dsa", outer_acceleration="power")
    setup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    flux = np.zeros_like(source)
    for iteration in range(1, 501):
        previous = flux.copy()
        for group in range(transport.G):
            group_source = source[group].copy()
            for from_group in range(transport.G):
                transfer = transport.scatter[from_group][group]
                if transfer is not None:
                    group_source += transfer * flux[from_group]
            flux[group] = transport._solve_group(
                group, group_source.ravel(), flux[group].ravel(), tol*.01).reshape(grid.shape)
        if np.max(np.abs(flux-previous)) <= tol*.01 * max(np.max(np.abs(flux)), 1e-300):
            break
    residual, integrated_defect = full_3d_balance(transport, flux, source)
    return flux, dict(
        converged=bool(iteration < 500 and residual <= tol),
        energy_iterations=iteration, balance_residual=float(residual),
        integrated_balance_defect=integrated_defect,
        setup_seconds=setup_seconds, solve_seconds=time.perf_counter()-started,
        n_sweeps=transport._sweep_count,
        quadrature=dict(mu=transport.mu.tolist(), eta=transport.eta.tolist(),
                        xi=transport.xi.tolist(), weights=transport.w.tolist()),
    )


def full_3d_eigen_reference(grid, materials, material_map, scheme):
    """Compare with the existing full-3D eigenvalue solver at matching angles."""
    started = time.perf_counter()
    transport = TriSNTransportSolver(grid, materials, material_map, n_polar=4, n_azi=8, scheme=scheme)
    setup_seconds = time.perf_counter()-started
    started = time.perf_counter()
    result = transport.solve(tol_k=1e-10, tol_source=1e-9)
    production = np.sum(transport.nsf * result.flux, axis=0)
    flux = result.flux / (production.sum() * grid.cell_volume)
    production = np.sum(transport.nsf * flux, axis=0)
    residual, integrated_defect = full_3d_balance(transport, flux, transport.chi*production/result.k_eff)
    return flux, dict(
        converged=bool(result.converged and residual <= 1e-8),
        k_eff=result.k_eff, outer_iterations=result.outer_iterations,
        balance_residual=residual, integrated_balance_defect=integrated_defect,
        setup_seconds=setup_seconds, solve_seconds=time.perf_counter()-started,
        n_sweeps=transport._sweep_count,
        quadrature=dict(mu=transport.mu.tolist(), eta=transport.eta.tolist(),
                        xi=transport.xi.tolist(), weights=transport.w.tolist()))


def run(case="synthetic", layers=4, height=6.0, schemes=("step", "scb"), compare_3d=False,
        eigenvalue=False):
    grid, materials, material_map, source = make_problem(case, layers, height)
    if eigenvalue and case == "synthetic":
        materials[0] = materials[0].replace(nu_sigma_f=[.025, .08], chi=[.8, .2])
    material_data = [dict(name=mat.name, total=mat.sigma_t.tolist(),
                          absorption=mat.sigma_a.tolist(), scattering=mat.sigma_s.tolist(),
                          production=mat.nu_sigma_f.tolist(), chi=mat.chi.tolist()) for mat in materials]
    material_json = json.dumps(material_data, sort_keys=True).encode()
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    report = dict(
        study=("Eigenvalue CPU planar S_N / axial diffusion prototype" if eigenvalue
               else "Fixed-source CPU planar S_N / axial diffusion prototype"),
        calculation="eigenvalue" if eigenvalue else "fixed-source",
        case=case, official_benchmark_geometry=False, git_head=revision,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        geometry=dict(shape=grid.shape, side_cm=grid.side, height_cm=grid.height,
                      radial_bc="vacuum", axial_bc="vacuum", layer_width_cm=grid.dz),
        data=dict(group_order="descending energy", scattering="source_group, target_group",
                  correction="use supplied isotropic transport totals; D=1/(3*total)",
                  source_units="neutrons / (cm^3 s); angular weights sum to one",
                  normalization=("unit volume-integrated fission production" if eigenvalue
                                 else "absolute prescribed source; fission multiplication disabled"),
                  url=("https://github.com/mit-crpg/OpenMOC/blob/develop/sample-input/c5g7-mgxs-hdf5.py"
                       if case == "c5g7" else None),
                  report_table=("NEA/NSC/DOC(2005)16, Appendix A, tables 1 and 2"
                                if case == "c5g7" else "explicit synthetic constants in this example"),
                  material_sha256=hashlib.sha256(material_json).hexdigest(),
                  material_map_sha256=hashlib.sha256(material_map.astype("<i8").tobytes()).hexdigest(),
                  source_sha256=(None if eigenvalue else hashlib.sha256(
                      source.astype("<f8").tobytes()).hexdigest()),
                  materials=material_data),
        source_files_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
            "garnet/sn_2d1d.py", "garnet/_sn_plane.py", "garnet/axial_diffusion.py",
            "garnet/tri_sn.py", "garnet/sn.py", "garnet/stencil.py", "garnet/solver.py",
            "garnet/materials.py", "garnet/blend.py", "garnet/benchmarks/c5g7.py",
            "examples/sn_2d1d_fixed_source.py")},
        tolerance=1e-10, results=[])
    if eigenvalue:
        report["eigenvalue_tolerances"] = dict(tol_k=1e-9, tol_source=1e-8, tol_inner=1e-10)
    if case == "c5g7":
        data_file = "garnet/benchmarks/_c5g7_data.py"
        report["source_files_sha256"][data_file] = hashlib.sha256((root/data_file).read_bytes()).hexdigest()
    for scheme in schemes:
        started = time.perf_counter()
        solver = SN2D1DSolver(grid, materials, material_map, scheme=scheme, n_polar=2, n_azi=8)
        setup_seconds = time.perf_counter() - started
        result = (solver.solve_eigenvalue(**report["eigenvalue_tolerances"]) if eigenvalue
                  else solver.solve(source, tol=report["tolerance"]))
        plane = solver.planes[0].transport
        row = dict(scheme=scheme, setup_seconds=setup_seconds,
                   quadrature=dict(mu=plane.mu.tolist(), eta=plane.eta.tolist(), weights=plane.w.tolist()))
        for name in ("converged", "reason", "iterations", "n_sweeps", "n_axial_solves", "solve_seconds",
                     "transport_residual", "balance_residual", "flux_disagreement",
                     "negative_effective_sources", "minimum_effective_source"):
            row[name] = getattr(result, name)
        if eigenvalue:
            for name in ("k_eff", "outer_iterations", "k_history", "k_change_history",
                         "source_change_history", "flux_change_history", "inner_iteration_history"):
                row[name] = getattr(result, name)
            row["integrated_fission_production"] = float(result.fission_source.sum() * grid.cell_volume)
        row["integrated_balance_defect"] = float(result.balance_defect.sum() * grid.cell_volume)
        if compare_3d:
            reference_flux, reference = (full_3d_eigen_reference(grid, materials, material_map, scheme)
                                         if eigenvalue else full_3d_reference(
                                             grid, materials, material_map, source, scheme))
            reference["uncertainty"] = "spatial/angular error unestimated; a same-mesh method comparison"
            row["full_3d_reference"] = reference
            row["flux_relative_l2_difference"] = float(
                np.linalg.norm(result.flux-reference_flux) / np.linalg.norm(reference_flux))
            if eigenvalue:
                row["reactivity_difference_pcm"] = 1e5 * (1/reference["k_eff"] - 1/result.k_eff)
        report["results"].append(row)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("synthetic", "c5g7"), default="synthetic")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--height", type=float, default=6.0)
    parser.add_argument("--scheme", choices=("step", "scb", "both"), default="both")
    parser.add_argument("--compare-3d", action="store_true")
    parser.add_argument("--eigenvalue", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    schemes = ("step", "scb") if args.scheme == "both" else (args.scheme,)
    report = run(args.case, args.layers, args.height, schemes, args.compare_3d, args.eigenvalue)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in report["results"]:
        if args.eigenvalue:
            print(f"{row['scheme']:4s}: k_eff={row['k_eff']:.9f}, {row['outer_iterations']} power updates")
        print(f"{row['scheme']:4s}: converged={row['converged']}, "
              f"{row['iterations']} iterations, balance={row['balance_residual']:.3e}, "
              f"flux disagreement={row['flux_disagreement']:.3e}")
        if args.compare_3d:
            print(f"      full-3D S_N converged={row['full_3d_reference']['converged']}, "
                  f"relative flux L2 difference={row['flux_relative_l2_difference']:.3%}")
            if args.eigenvalue:
                print(f"      full-3D k_eff={row['full_3d_reference']['k_eff']:.9f}, "
                      f"reactivity difference={row['reactivity_difference_pcm']:.1f} pcm")
    if not all(row["converged"] and row.get("full_3d_reference", {}).get("converged", True)
               for row in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
