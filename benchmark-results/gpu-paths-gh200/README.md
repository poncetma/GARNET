# GPU-only code paths on GH200 (audit item O4)

Run 2026-10-29 on Merlin7 GH200 nodes: NVIDIA GH200 120GB, driver 595.91.07,
aarch64. The environment was `garnet-gh-py313-v1` (Python 3.13, NumPy 2.3.2,
SciPy 1.16.3, CuPy 13.6.0, pytest 8.4.2). Both jobs ran a staged snapshot of
commit `20967d3` plus the uncommitted audit work of that day. Each log starts
with the snapshot's provenance. The staged `garnet/` and `tests/` were
identical to the working tree with every O-item change, including the O11
error-norm default.

## Targeted run: job 241680 (gh-hourly)

[tests/verification/test_gpu_paths.py](../../tests/verification/test_gpu_paths.py)
checks each audit path against an independent counterpart:

| Path | Check |
|---|---|
| Fused SDPN block stack, forward and **transposed** (`spn.py` `_coupling_stack`) | Non-symmetric SDP3 vacuum block: fused = unfused = CPU operator to 1e-13 for `apply` and `apply_transpose`; both stacks built |
| SDP3 forward and adjoint eigen-solves on the device | k within 1e-9 of CPU, normalized flux within 1e-7, Brantley–Larsen-style 20×20 core |
| Per-material β precursor start on CuPy (`transient.py`, mapped device field) | Two-region, two-group transient with distinct β rows: GPU power history = CPU to 1e-9 relative |
| PCG CUDA-graph capture (`graph_block`) | Same iterate (1e-12) and iteration count as direct PCG; one capture, replayed, and reused by a second solve |
| GMRES with CGS2 on the device | Matches host MGS GMRES to 1e-9, iteration count within one, across restarts |

The run also included the existing GPU-gated modules `test_cartesian_adf.py`
and `test_fused_kernels.py`, plus `test_linalg_krylov.py`:
**123 passed, 0 skipped, 0 failed** in 18 s.
Evidence: [gpu-paths-gh200-20261029.xml](gpu-paths-gh200-20261029.xml) and
[the log](gpu-paths-gh200-20261029.log).

## Full suite: job 241683 (gh-daily)

The full regression suite with slow tests included (`-o addopts=`) gave
**1283 passed, 4 skipped, 0 failed** in 8 min. The skips:

- preCICE, SymPy and PyAMG are not installed in the GH200 environment;
- one test targets CUDA-less machines only.

Evidence: [full-gh200-20261029.xml](full-gh200-20261029.xml) and
[the log](full-gh200-20261029.log).

## Reproduce

```bash
GARNET_REGRESSION_PARTITION=gh-hourly GARNET_REGRESSION_TIME=01:00:00 \
GARNET_REGRESSION_STAGE=/data/scratch/shared/poncet_m/garnet-gpu-paths-stage \
GARNET_PYTEST_TARGETS="tests/verification/test_gpu_paths.py tests/verification/test_cartesian_adf.py tests/verification/test_fused_kernels.py tests/verification/test_linalg_krylov.py" \
bash slurm/stage_and_submit_gpu_regression.sh            # targeted

GARNET_REGRESSION_PARTITION=gh-daily GARNET_REGRESSION_TIME=06:00:00 \
GARNET_REGRESSION_STAGE=/data/scratch/shared/poncet_m/garnet-gpu-full-stage \
bash slurm/stage_and_submit_gpu_regression.sh            # full suite
```

`slurm/run_garnet_gpu_regression.sh` picks the GH200 Python and its CUDA
libraries on aarch64 nodes; the A100 defaults are unchanged. The A100 has not
run these paths.
