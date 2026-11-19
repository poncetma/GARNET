---
name: verification-tests
description: How to develop verification tests and checks in GARNET (pytest tests, analytic benchmarks, consistency checks, diagnostic scripts that validate a new method or solver feature). Use whenever writing or extending a test, a verification script or an acceptance check, and before reporting that something "passes" or "agrees". Core rule - break every new test on purpose before trusting it.
---

# Developing verification tests

## The rule: break it on purpose

A new test is not trusted until it has been seen to **fail** on a deliberately
broken input. A check that cannot fail gives a false positive: a wrong sign,
an unused argument, a comparison of a quantity with itself, a tolerance wider
than the effect, or a symmetry that makes the broken and correct versions
agree. Before relying on a test (and before writing "agrees to 1e-10"):

1. **Introduce the defect the test is meant to catch.** Disable the new
   code path, drop a term, transpose the wrong index, use the old formula.
   Monkeypatching the method in a scratch script is usually enough.
2. **Run the test and confirm it fails, by a margin.** The broken version
   should miss by orders of magnitude more than the tolerance. Record the
   margin in the test's docstring or the report: "a transposition of P0 only
   misses by 312 pcm; the tolerance is 1e-6 pcm".
3. **Break each ingredient separately.** One check can pass while one of
   several parts is wrong. Break each part on its own.
4. **Make sure the case exercises the feature.** A one-group, symmetric,
   homogeneous or isotropic case often makes the new code a no-op. Use it
   for an exact identity, and add a case where the feature changes the
   answer: several groups with upscatter, anisotropy, heterogeneity, mixed
   cells, a non-convex boundary.
5. **Check the comparison is not trivial.** Compare against something
   computed independently: an analytic value, a different engine or
   scheme, a perturbation estimate, a conserved balance. Do not compare
   against the same code path. Confirm the compared quantities actually
   differ from a naive guess, e.g. the adjoint flux differs from the
   forward flux.

## Techniques that worked here

- **Negative controls by monkeypatching**, as in the S_N adjoint (2026-11-18).
  The correct adjoint gives k_adj − k_fwd = 1e-7 pcm. Without the
  transposition, or without the χ/νΣf swap, it gives −14,000 pcm.
  Transposing only P0 and not the higher moments gives +312 pcm.
- **Order of convergence as a test.** A first-order perturbation estimate
  must have an error that drops 10× when the perturbation drops 10×. That
  is stronger than a single tolerance.
- **Exact limits.** In one group the S_N problem is self-adjoint (adjoint
  flux = forward flux to 1e-15). With levels=0 the drum-band mesh reproduces
  the raster solve. Zero moments reproduce the isotropic solve. Use these as
  identities, but only next to a case where the feature matters.
- **Balances.** Production/k − absorption = boundary leakage closes to the
  solver tolerance. It catches missing face measures and wrong normal
  directions.
- **Reproduce the former implementation** cell for cell after a port or
  refactor. Snapshot the old output before changing the code.
- **Attribution.** When a composite result is off, interpolate or perturb one
  ingredient at a time with the rest exact. The contributions must add up to
  the total. If they don't, the attribution itself is wrong.

## Reporting

- State what was broken and by how much it then failed, next to the passing
  result.
- A tolerance without a demonstrated failing margin is not evidence.
- Keep the scratch scripts for negative controls in the scratchpad, and move
  the strongest one into the test module as a parametrized "broken" case when
  it is cheap.
