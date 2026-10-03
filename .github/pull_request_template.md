## Summary

<!-- What changed, and why? -->

## Verification

<!-- List exact commands and results. -->

- [ ] `ruff check .`
- [ ] `pytest -q`
- [ ] Documentation updated where needed
- [ ] `CHANGELOG.md` updated for user-visible changes

## Distributed or GPU evidence

<!-- If applicable: backend, world size, hardware, PyTorch/CUDA versions, tolerances, and result paths. Otherwise write "Not applicable." -->

## Checklist

- [ ] The change is focused and contains no credentials, private data, or machine-specific paths.
- [ ] New execution paths are compared element-wise with an independent full-batch oracle.
- [ ] Performance claims come from benchmark runs, not verification or profiler timings.
