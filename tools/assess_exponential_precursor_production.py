"""E5a: PARCS-style physical precursor production versus FTM2's default.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tools.assess_exponential_precursor_production

Every FTM2 configuration shares the transformed-theta neutron equation,
frequency predictor, controller and spatial path. Only ``precursor_production``
changes: linear transformed (default), linear physical, or quadratic physical
through three accepted endpoints. References are independent: exact matrix
exponentials of assembled kinetics generators (Radau across a ramp), and the
chunk-7 TWIGL physical-generator Radau reference. No heat equation enters.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import interp1d
from scipy.linalg import expm

from garnet import Kinetics, Material, Model
from garnet.timescheme import ExponentialTheta
from tools.assess_exponential_twigl import (Reference, TwiglSystem, error_gate, graded_widths,
                                            measure)


ROOT = Path(__file__).resolve().parents[1]
MODES = ExponentialTheta.precursor_productions
BUDGET = 1e-3
BETA6 = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
DECAY6 = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
# name: (final reactivity in dollars, ramp duration or 0 for a step, march end)
KINETICS_CASES = {'step-0.3$': (0.3, 0., 2.0), 'step-0.8$': (0.8, 0., 0.5),
                  'shutdown-5$': (-5.0, 0., 1.0), 'ramp-0.5$': (0.5, 0.2, 1.0)}
# chunk-7 0.1% minimum-solve BDF selections, rerun to confirm the comparator.
TWIGL_BDF = {('step', 2): (1e-4, 0.02), ('ramp', 2): (1e-3, 0.01),
             ('step', 4): (1e-5, 0.02), ('ramp', 4): (1e-3, 0.01)}


def exponential_integral(matrix, initial, duration):
    """Exact int_0^T expm(A s) y0 ds from one augmented matrix exponential."""
    n = matrix.shape[0]
    augmented = np.zeros((n + 1, n + 1))
    augmented[:n, :n], augmented[:n, n] = matrix * duration, initial * duration
    return expm(augmented)[:n, n]


def radau_integral(solution, component):
    """Three-point Gauss integration is exact for Radau's cubic dense segments."""
    nodes, weights = np.polynomial.legendre.leggauss(3)
    widths = np.diff(solution.t)
    times = solution.t[:-1, None] + 0.5 * widths[:, None] * (1 + nodes)
    values = solution.sol(times.ravel())[component].reshape(times.shape)
    return float(np.sum(0.5 * widths * (values @ weights)))


def common_times(duration, event=None):
    return np.unique(np.r_[np.linspace(0, duration, 2001),
                           np.geomspace(1e-9, min(0.02, duration), 401),
                           [] if event is None else [event]])


def configurations(duration, event, *, rtols, output_rtols, max_dt, graded_max):
    """(label, configuration, transient options) at identical neutron settings."""
    rows = []
    for mode in MODES:
        for rtol in rtols:
            for output in output_rtols:
                rows.append((f'ftm2-{mode}', dict(production=mode, rtol=rtol, output_rtol=output),
                             dict(time_scheme='ftm2', ftm_kwargs={'precursor_production': mode},
                                  adaptive_ftm=dict(rtol=rtol, output_rtol=output, estimator='history',
                                                    min_dt=1e-11, max_dt=max_dt, max_rejections=40))))
        widths = graded_widths(duration, graded_max, event)
        for frequency in (None, 0.):
            rows.append((f'ftm2-{mode}', dict(production=mode, frequency=frequency, shared_mesh=True),
                         dict(time_scheme='ftm2', dt=widths,
                              ftm_kwargs={'precursor_production': mode, 'frequency': frequency})))
    for rtol in rtols:
        rows.append(('adaptive-bdf', dict(rtol=rtol, max_dt=max_dt),
                     dict(time_scheme='bdf5', adaptive_bdf=dict(
                         rtol=rtol, min_dt=1e-11, max_dt=max_dt, max_rejections=40, automatic_order=True))))
    return rows


def work(result):
    return dict(accepted_steps=result.n_steps, rejected_steps=result.rejected_steps,
                implicit_solves=sum(result.feedback_iterations) + sum(result.rejected_feedback_iterations),
                krylov_iterations=result.total_inner_iterations,
                fgmres_iterations=sum(result.step_iterations) + sum(result.rejected_step_iterations),
                precursor_productions=dict(Counter(map(str, result.step_precursor_productions))))


def gated(row, keys):
    return all(np.max(row[key]) <= BUDGET for key in keys) and row['minimum_final_precursor'] >= 0


def fewest(rows, keys):
    """Fewest solves meeting the budget in every listed error, per method."""
    out = {}
    for row in rows:
        if gated(row, keys) and (row['method'] not in out
                                 or row['implicit_solves'] < out[row['method']]['implicit_solves']):
            out[row['method']] = row
    return {method: dict(configuration=row['configuration'], implicit_solves=row['implicit_solves'],
                         krylov_iterations=row['krylov_iterations'],
                         errors={key: float(np.max(row[key])) for key in keys})
            for method, row in out.items()}


def point_kinetics_case(name):
    dollars, ramp, duration = KINETICS_CASES[name]
    fuel = Material(name='fuel', diffusion=[1.3], sigma_a=[0.03], nu_sigma_f=[0.035])
    model = (Model((1, 1, 1), (1, 1, 1)).fill(fuel).set_boundary('reflective')
             .set_kinetics(Kinetics([2.2e5], BETA6, DECAY6)))
    steady = model.steady(device='cpu', tol_k=1e-13, tol_source=1e-12)
    production = 0.035 / steady.k_eff        # nuSigma_f/k0 = initial absorption
    generation = 1 / (2.2e5 * production)
    rho_final = dollars * BETA6.sum()
    # Absorption change sets rho = 1 - sigma_a/(nuSigma_f/k0) exactly for one cell.
    # A step acts from 0+, so the shared initial eigenpair stays unperturbed.
    rho = ((lambda t: rho_final * (t > 0)) if not ramp
           else (lambda t: rho_final * min(max(t, 0.) / ramp, 1.)))

    def matrix(r):
        A = np.zeros((7, 7))
        A[0, 0], A[0, 1:] = (r - BETA6.sum()) / generation, DECAY6
        A[1:, 0], A[1:, 1:] = BETA6 / generation, -np.diag(DECAY6)
        return A

    initial = np.r_[1., BETA6 / (generation * DECAY6)]
    event = ramp or None
    if ramp:
        first = solve_ivp(lambda t, y: matrix(rho(t)) @ y, (0, ramp), initial, method='Radau',
                          jac=lambda t, y: matrix(rho(t)), rtol=1e-12, atol=1e-14, dense_output=True)
        A, start = matrix(rho_final), first.y[:, -1]
        reference = lambda t: np.array([first.sol(s) if s <= ramp else expm(A * (s - ramp)) @ start
                                        for s in np.atleast_1d(t)])
        integral = radau_integral(first, 0) + exponential_integral(A, start, duration - ramp)[0]
    else:
        A = matrix(rho_final)
        reference = lambda t: np.array([expm(A * s) @ initial for s in np.atleast_1d(t)])
        integral = exponential_integral(A, initial, duration)[0]
    equilibrium = model.transient(1e-3, 1e-3, device='cpu', initial_steady=steady, tol_step=1e-12,
                                  multigroup_kwargs={'rtol': 1e-13}).precursors_numpy.ravel()
    samples = common_times(duration, event)
    final = reference(duration)[0]
    rows = []
    for method, configuration, options in configurations(
            duration, event, rtols=(1e-3, 1e-4, 1e-5), output_rtols=(1e-4, 1e-3), max_dt=0.05,
            graded_max=0.01):
        dt = options.pop('dt', 0.005)
        result = model.transient(duration, dt, device='cpu', initial_steady=steady,
            materials_at=lambda t: {'fuel': fuel.replace(sigma_a=[production * (1 - rho(t))])} if t > 0 else {},
            event_times=[event] if event else None, tol_step=1e-12,
            multigroup_kwargs={'rtol': 1e-13}, **options)
        times, power = np.asarray(result.times), np.asarray(result.power)
        grid = np.unique(np.r_[samples, times])
        reference_grid = reference(grid)[:, 0]
        ratio = result.precursors_numpy.ravel() / equilibrium / (final[1:] / initial[1:])
        row = dict(method=method, configuration=configuration, **work(result),
                   maximum_relative_power_error=float(np.max(np.abs(
                       np.interp(grid, times, power) / reference_grid - 1))),
                   maximum_endpoint_relative_power_error=float(np.max(np.abs(
                       power / reference(times)[:, 0] - 1))),
                   relative_power_integral_error=float(abs(trapezoid(power, times) / integral - 1)),
                   final_relative_precursor_error_by_family=np.abs(ratio - 1).tolist(),
                   minimum_final_precursor=float(result.precursors_numpy.min()))
        rows.append(row)
        print(name, method, configuration, f"P={row['maximum_relative_power_error']:.3e}",
              f"C={max(row['final_relative_precursor_error_by_family']):.3e}",
              f"solves={row['implicit_solves']}", flush=True)
    keys = ('maximum_relative_power_error', 'relative_power_integral_error',
            'final_relative_precursor_error_by_family')
    return name, dict(dollars=dollars, ramp_s=ramp, duration_s=duration, generation_s=generation,
                      reference='exact 7x7 expm' + (' after Radau ramp (rtol 1e-12)' if ramp else ''),
                      records=rows, fewest_solves_at_budget=fewest(
                          [r for r in rows if 'shared_mesh' not in r['configuration']], keys))


def localized_case(name='localized-six-family'):
    """Two-group, eight-cell slab: six families, distinct delayed spectra and a
    left-quarter thermal absorption step (about +0.40$) that tilts the shape."""
    diffusion, absorption = np.array([1.4, 0.4]), np.array([0.02, 0.1])
    fission, spectrum = np.array([0.005, 0.15]), np.array([0.9, 0.1])
    scatter, speeds = np.array([[0., 0.03], [0.002, 0.]]), np.array([1e7, 2.2e5])
    delayed = np.tile([[1., 0.]], (6, 1))
    delayed[3:] = [0.9, 0.1]
    cells, width, duration, factor = 8, 5., 1.0, 0.989
    fuel = Material(name='fuel', diffusion=diffusion, sigma_a=absorption, nu_sigma_f=fission,
                    sigma_s=scatter, chi=spectrum)
    left = fuel.replace(name='left')
    model = (Model((cells * width, 1, 1), (cells, 1, 1)).fill(fuel).add_box(left, x=(0, 2 * width))
             .set_boundary('reflective').set_kinetics(Kinetics(speeds, BETA6, DECAY6, chi_delayed=delayed)))
    steady = model.steady(device='cpu', tol_k=1e-13, tol_source=1e-12)
    changed = left.replace(sigma_a=absorption * [1., factor])
    perturbed_k = (Model((cells * width, 1, 1), (cells, 1, 1)).fill(fuel).add_box(changed, x=(0, 2 * width))
                   .set_boundary('reflective').steady(device='cpu', tol_k=1e-13, tol_source=1e-12).k_eff)
    flux0 = steady.flux_numpy.reshape(2, cells).copy()
    flux0 /= np.sum(fission[:, None] * flux0) / steady.k_eff
    source0 = fission @ flux0 / steady.k_eff
    initial = np.r_[flux0.ravel(), (BETA6[:, None] / DECAY6[:, None] * source0).ravel()]
    # Independent reflecting FV generator, uniform D so harmonic faces are D.
    laplace = (np.diag(np.r_[-1., -2. * np.ones(cells - 2), -1.]) + np.diag(np.ones(cells - 1), 1)
               + np.diag(np.ones(cells - 1), -1)) / width**2
    absorbed = np.broadcast_to(absorption[:, None], (2, cells)).copy()
    absorbed[1, :2] *= factor
    n = (2 + 6) * cells
    A = np.zeros((n, n))
    block = lambda i: slice(cells * i, cells * (i + 1))
    for g in range(2):
        A[block(g), block(g)] += speeds[g] * (diffusion[g] * laplace - np.diag(absorbed[g] + scatter[g].sum()))
        for h in range(2):
            A[block(g), block(h)] += speeds[g] * np.eye(cells) * (
                scatter[h, g] + (spectrum[g] - BETA6 @ delayed[:, g]) * fission[h] / steady.k_eff)
        for i in range(6):
            A[block(g), block(2 + i)] = speeds[g] * delayed[i, g] * DECAY6[i] * np.eye(cells)
    for i in range(6):
        A[block(2 + i), block(2 + i)] = -DECAY6[i] * np.eye(cells)
        for g in range(2):
            A[block(2 + i), block(g)] = BETA6[i] * fission[g] / steady.k_eff * np.eye(cells)
    reference = lambda t: np.array([expm(A * s) @ initial for s in np.atleast_1d(t)])
    power_of = lambda states: states[:, :2 * cells].reshape(-1, 2, cells).transpose(0, 2, 1) @ fission / steady.k_eff
    total = lambda states: power_of(states).sum(axis=1)
    integral = float(total(exponential_integral(A, initial, duration)[None])[0])
    samples = common_times(duration)
    final = reference(duration)[0]
    final_c = final[2 * cells:].reshape(6, cells)
    rows = []
    for method, configuration, options in configurations(
            duration, None, rtols=(1e-3, 1e-4, 1e-5), output_rtols=(1e-4, 1e-3), max_dt=0.05,
            graded_max=0.01):
        dt = options.pop('dt', 0.005)
        snapshots = [flux0.ravel()]
        result = model.transient(duration, dt, device='cpu', initial_steady=steady,
            materials_at=lambda t: {'left': changed} if t > 0 else {},
            on_step=lambda t, phi, power: snapshots.append(np.asarray(phi).ravel().copy()),
            tol_step=1e-12, multigroup_kwargs={'rtol': 1e-13}, **options)
        times, power = np.asarray(result.times), np.asarray(result.power)
        grid = np.unique(np.r_[samples, times])
        expected = reference(grid)
        flux = interp1d(times, np.asarray(snapshots), axis=0, assume_sorted=True)(grid).reshape(-1, 2, cells)
        expected_flux = expected[:, :2 * cells].reshape(-1, 2, cells)
        flux_rms = np.sqrt(np.sum((flux - expected_flux)**2, axis=-1) / np.sum(expected_flux**2, axis=-1))
        shape_rms = np.sqrt(np.sum((flux / np.interp(grid, times, power)[:, None, None]
                                    - expected_flux / total(expected)[:, None, None])**2, axis=-1)
                            / np.sum((expected_flux / total(expected)[:, None, None])**2, axis=-1))
        precursors = result.precursors_numpy.reshape(6, cells)
        row = dict(method=method, configuration=configuration, **work(result),
                   maximum_relative_power_error=float(np.max(np.abs(np.interp(grid, times, power) / total(expected) - 1))),
                   maximum_relative_flux_rms_error_by_group=np.max(flux_rms, axis=0).tolist(),
                   maximum_relative_shape_rms_error_by_group=np.max(shape_rms, axis=0).tolist(),
                   relative_power_integral_error=float(abs(trapezoid(power, times) / integral - 1)),
                   final_relative_precursor_rms_error_by_family=np.sqrt(
                       np.sum((precursors - final_c)**2, axis=1) / np.sum(final_c**2, axis=1)).tolist(),
                   minimum_final_precursor=float(precursors.min()))
        rows.append(row)
        print(name, method, configuration, f"P={row['maximum_relative_power_error']:.3e}",
              f"C={max(row['final_relative_precursor_rms_error_by_family']):.3e}",
              f"solves={row['implicit_solves']}", flush=True)
    keys = ('maximum_relative_power_error', 'maximum_relative_flux_rms_error_by_group',
            'maximum_relative_shape_rms_error_by_group', 'relative_power_integral_error',
            'final_relative_precursor_rms_error_by_family')
    shape_change = np.ptp(final[cells:2 * cells] / final[cells:2 * cells].mean())
    return name, dict(reactivity_dollars=float((perturbed_k - steady.k_eff) / perturbed_k / BETA6.sum()),
                      final_thermal_shape_peak_to_peak=float(shape_change), duration_s=duration,
                      reference='exact dense expm of an independently assembled 48x48 generator',
                      records=rows, fewest_solves_at_budget=fewest(
                          [r for r in rows if 'shared_mesh' not in r['configuration']], keys))


def twigl_case(kind, refinement):
    system = TwiglSystem.build(kind, refinement)
    reference = Reference.solve(system, kind, rtol=3e-11)
    event = 0.2 if kind == 'ramp' else None
    rows = []
    runs = configurations(0.5, event, rtols=(1e-3, 1e-4), output_rtols=(1e-4, 1e-3), max_dt=0.02,
                          graded_max=0.005)
    # Lagged shared meshes only; the chunk-7 BDF selection replaces the sweep.
    runs = [r for r in runs if r[0] != 'adaptive-bdf'
            and ('shared_mesh' not in r[1] or r[1]['frequency'] is None)]
    rtol, cap = TWIGL_BDF[kind, refinement]
    runs.append(('adaptive-bdf', dict(rtol=rtol, max_dt=cap, chunk7_selection=True),
                 dict(time_scheme='bdf5', adaptive_bdf=dict(rtol=rtol, min_dt=1e-11, max_dt=cap,
                                                            max_rejections=40, automatic_order=True))))
    for method, configuration, options in runs:
        dt = options.pop('dt', 0.005)
        snapshots = [system.initial[:2 * system.count].reshape(2, system.count)]
        result = system.model.transient(0.5, dt, at=system.problem.problem_at,
            initial_steady=system.steady, device='cpu', event_times=[event] if event else None,
            on_step=lambda t, phi, power: snapshots.append(np.asarray(phi).reshape(2, system.count).copy()),
            step_solver='monolithic', tol_step=1e-10, multigroup_kwargs=dict(rtol=1e-11), **options)
        row = measure(system, reference, result.times, result.power, np.asarray(snapshots),
                      result.precursors_numpy)
        row.update(method=method, configuration=configuration, **work(result),
                   meets_budget=bool(error_gate(row, BUDGET)))
        rows.append(row)
        print(f'twigl-{kind}-{refinement}', method, configuration,
              f"P={row['maximum_relative_power_error']:.3e}",
              f"C={max(row['final_relative_precursor_rms_error_by_family']):.3e}",
              f"solves={row['implicit_solves']}", flush=True)
    selected = {}
    for row in rows:
        if row['meets_budget'] and 'shared_mesh' not in row['configuration'] and (
                row['method'] not in selected or row['implicit_solves'] < selected[row['method']]['implicit_solves']):
            selected[row['method']] = row
    return f'twigl-{kind}-{refinement}', dict(
        k0=system.steady.k_eff, records=rows,
        fewest_solves_at_budget={m: dict(configuration=r['configuration'], implicit_solves=r['implicit_solves'],
                                         krylov_iterations=r['krylov_iterations'],
                                         maximum_relative_power_error=r['maximum_relative_power_error'])
                                 for m, r in selected.items()})


def run_task(task):
    kind, *args = task
    return {'kinetics': point_kinetics_case, 'localized': localized_case, 'twigl': twigl_case}[kind](*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts', nargs='+', choices=['kinetics', 'localized', 'twigl'],
                        default=['kinetics', 'localized', 'twigl'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmark-results/exponential-second-order/chunk8-precursor-production.json')
    args = parser.parse_args()
    tasks = ([('kinetics', name) for name in KINETICS_CASES] if 'kinetics' in args.parts else [])
    tasks += [('localized',)] if 'localized' in args.parts else []
    tasks += ([('twigl', kind, r) for r in (2, 4) for kind in ('step', 'ramp')]
              if 'twigl' in args.parts else [])
    sources = ['garnet/_exponential.py', 'garnet/timescheme.py', 'garnet/transient.py',
               'tools/assess_exponential_twigl.py', 'tools/assess_exponential_precursor_production.py']
    report = dict(date='2026-10-28', scope='E5a precursor production control at identical FTM2 neutron settings',
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         device='cpu'),
        budget=BUDGET, modes=list(MODES),
        controls='Adaptive FTM2 uses the history estimator; output_rtol 1e-4 and 1e-3. Shared graded meshes '
                 'isolate production formulas at identical endpoints (lagged and zero frequency).',
        work='All constituent and rejected solves, Krylov iterations; no timing claim',
        precursor_scope='Final precursor fields only',
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}, cases={})
    if args.output.exists():   # Merge parts run separately into one report.
        report['cases'] = json.loads(args.output.read_text())['cases']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                name, case = future.result()
            except Exception as error:
                failures.append((futures[future], error))
                print('FAILED', futures[future], repr(error), flush=True)
                continue
            report['cases'][name] = case
            args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')
            print('SAVED', name, flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} assessment tasks failed: {failures}')
    print('WROTE', args.output, flush=True)


if __name__ == '__main__':
    main()
