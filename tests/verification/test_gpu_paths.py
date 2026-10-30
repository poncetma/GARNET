"""GPU-only code paths, checked against their CPU or uncaptured counterparts.

Audit item O4: these paths run only under CuPy, so the CPU suite never
executes them. Each test compares the GPU result with an independent path:

* the fused SDPN block stack, forward and transposed, against the unfused
  per-pair reaction and the CPU operator, and through forward and adjoint
  SDP3 eigen-solves against the CPU solves;
* the per-material beta precursor start (a mapped device field) through a
  short transient against the CPU transient;
* PCG with CUDA-graph capture against the same solve without capture: same
  iterate, same iteration count, one capture reused by a second solve;
* GMRES with two-pass classical Gram-Schmidt (CGS2) on the device against
  modified Gram-Schmidt on the host, across restarts.

The module skips without CuPy or a CUDA device.
"""

import numpy as np
import pytest

from garnet import Grid, Kinetics, kernels
from garnet.linalg import PCGWorkspace, gmres, pcg
from garnet.operator import GroupOperator, SDPNGroupOperator
from tests.verification.test_sdpn_symmetric import _bl_problem
from tests.verification.test_transient_per_material import _model, _two_region

TIGHT = dict(tol_k=1e-10, tol_source=1e-9)


@pytest.fixture
def cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA runtime")
    return cp


def _sdp3_vacuum_operator(xp, seed=11):
    grid = Grid(shape=(9, 7, 5), size=(9.0, 7.0, 5.0))
    rng = np.random.default_rng(seed)
    D1, st, rem = (0.5 + rng.random(grid.shape), 1.0 + rng.random(grid.shape),
                   0.1 + 0.3 * rng.random(grid.shape))
    bc = (("reflective", "vacuum"), ("vacuum", "reflective"), "reflective")
    op = SDPNGroupOperator(xp, grid, xp.asarray(D1), xp.asarray(st), xp.asarray(rem),
                           order=3, bc=bc)
    return grid, op, rng


def test_gpu_fused_sdpn_block_matches_unfused_and_cpu(cuda):
    cp = cuda
    grid, op, rng = _sdp3_vacuum_operator(cp)
    _, cpu_op, _ = _sdp3_vacuum_operator(np)
    assert not op.symmetric                      # a real transpose
    u = rng.random((op.M,) + grid.shape)
    assert kernels.use_fused(cp, "block")
    for name in ("apply", "apply_transpose"):
        fused = cp.asnumpy(getattr(op, name)(cp.asarray(u)))
        previous = kernels.set_fused(False)
        try:
            unfused = cp.asnumpy(getattr(op, name)(cp.asarray(u)))
        finally:
            kernels.set_fused(previous)
        reference = getattr(cpu_op, name)(u)
        scale = np.abs(reference).max()
        np.testing.assert_allclose(fused, unfused, rtol=0, atol=1e-13 * scale)
        np.testing.assert_allclose(fused, reference, rtol=0, atol=1e-13 * scale)
    assert op._stacked_coupling is not None and op._stacked_coupling_t is not None


def test_gpu_sdp3_forward_and_adjoint_match_cpu(cuda):
    from garnet import SDP3EigenSolver

    grid, mats, mmap, bc = _bl_problem(20)
    solvers = {d: SDP3EigenSolver(grid, mats, material_map=mmap, bc=bc, device=d)
               for d in ("cpu", "gpu")}
    assert solvers["gpu"].device != "cpu"
    assert not all(op.symmetric for op in solvers["gpu"].ops)
    for adjoint in (False, True):
        result = {d: s.solve(adjoint=adjoint, **TIGHT) for d, s in solvers.items()}
        assert all(r.converged for r in result.values())
        assert result["gpu"].k_eff == pytest.approx(result["cpu"].k_eff, abs=1e-9)
        a, b = (np.asarray(result[d].flux_numpy) for d in ("cpu", "gpu"))
        np.testing.assert_allclose(b / np.linalg.norm(b), a / np.linalg.norm(a), atol=1e-7)


def test_gpu_per_material_beta_transient_matches_cpu(cuda):
    _, _, problem_at = _two_region()
    kinetics = Kinetics(velocities=np.tile([1.0e7, 2.0e5], (2, 1)),
                        beta=[[0.005, 0.0025], [0.006, 0.0020]], decay=[0.08, 0.4])
    power = {d: _model(problem_at, kinetics).transient(
        at=problem_at, device=d, t_end=0.05, dt=0.01).power for d in ("cpu", "gpu")}
    assert power["cpu"][-1] > 1.001                  # the step inserted reactivity
    np.testing.assert_allclose(power["gpu"], power["cpu"], rtol=1e-9)


def _device_stencil(cp, n=24):
    rng = np.random.default_rng(7)
    grid = Grid(shape=(n, n, n), size=(60.0, 60.0, 60.0))
    op = GroupOperator(cp, grid, cp.asarray(0.5 + rng.random(grid.shape)),
                       cp.asarray(0.02 + 0.05 * rng.random(grid.shape)))
    return op, cp.asarray(rng.random(grid.shape))


def test_gpu_pcg_graph_capture_matches_direct_iterations(cuda):
    cp = cuda
    op, b = _device_stencil(cp)
    kw = dict(rtol=1e-10, check_every=4)
    reference, reference_it = pcg(op.apply, b, cp.zeros_like(b), op.inv_diag, cp, **kw)
    workspace = PCGWorkspace.like(b, operator_out=True)
    solved, iterations = pcg(op.apply, b, cp.zeros_like(b), op.inv_diag, cp,
                             graph_block=4, workspace=workspace, **kw)
    assert workspace.graph_error is None, workspace.graph_error
    assert workspace.graph_captures == 1 and workspace.graph_replays > 0
    assert iterations == reference_it > 8
    expected = cp.asnumpy(reference)
    np.testing.assert_allclose(cp.asnumpy(solved), expected, rtol=0,
                               atol=1e-12 * np.abs(expected).max())
    # A second solve with the same operator replays the captured graph.
    replays = workspace.graph_replays
    _, again = pcg(op.apply, 2.0 * b, cp.zeros_like(b), op.inv_diag, cp,
                   graph_block=4, workspace=workspace, **kw)
    assert again == reference_it
    assert workspace.graph_captures == 1 and workspace.graph_replays > replays


def test_gpu_gmres_cgs2_matches_host_mgs(cuda):
    cp = cuda
    rng = np.random.default_rng(3)
    n = 400
    A = rng.random((n, n)) - 0.5
    A += np.diag(np.abs(A).sum(axis=1) * 0.3 + 1.0)   # non-normal, needs restarts
    b = rng.random(n)
    inv_diag = 1.0 / np.diag(A)
    kw = dict(rtol=1e-11, restart=10)
    host, host_it = gmres(lambda x: A @ x, b, np.zeros(n), inv_diag, np, **kw)
    A_d = cp.asarray(A)
    device, device_it = gmres(lambda x: A_d @ x, cp.asarray(b), cp.zeros(n),
                              cp.asarray(inv_diag), cp, **kw)
    assert host_it > 10                               # at least one restart
    assert abs(device_it - host_it) <= 1
    np.testing.assert_allclose(cp.asnumpy(device), host, rtol=1e-9)
    np.testing.assert_allclose(A @ cp.asnumpy(device), b, rtol=0,
                               atol=1e-10 * np.linalg.norm(b))
