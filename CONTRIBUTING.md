# Contributing to DistLLM

Thanks for helping improve DistLLM. The project values small, auditable changes and evidence-backed claims.

## Before you start

- Search existing issues before opening a new one.
- Use an issue to discuss changes that alter the objective contract, parallel topology, result schema, or supported PyTorch versions.
- Keep pull requests focused. Unrelated refactors make numerical review harder.
- Do not include private datasets, credentials, model weights, or machine-specific paths.

## Development setup

```bash
git clone https://github.com/Sentinel3252/DistLLM.git
cd DistLLM
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the local checks:

```bash
ruff check .
pytest -q
python -m build
twine check dist/*
```

CPU tests spawn real 2- and 4-process Gloo groups. GPU changes should additionally run the appropriate scripts in `scripts/` and state the hardware, PyTorch build, CUDA version, and exact command in the pull request.

## Correctness requirements

Changes to training or communication code must preserve these invariants:

- Loss is summed per micro-batch and divided by the supervised-token count once per optimizer window.
- DDP's gradient averaging is compensated exactly once.
- TP and PP do not multiply replicated token counts by world size.
- Empty windows skip the optimizer step.
- Performance claims are kept separate from verification or profiler runs.

Add a full-batch oracle comparison when introducing a new execution path. Comparing only a scalar loss or gradient norm is not sufficient; tests should check the affected parameters element-wise.

## Results and benchmarks

Generated output is normally not committed. A maintainer may accept a compact result file when it is stable, documents a supported configuration, and is referenced from `docs/validation.md`.

Benchmark pull requests must distinguish observed data from projections. Never infer GPU kernel overlap from host submission timing or `async_op=True` alone.

## Documentation and style

- Use clear names and short docstrings for non-obvious contracts.
- Update the README only for user-facing behavior; put methodology in `docs/`.
- Add an entry under `Unreleased` in `CHANGELOG.md` for user-visible changes.
- Keep claims reproducible and state unvalidated hardware paths explicitly.

## Pull requests

By submitting a contribution, you agree that it is licensed under the Apache License 2.0. The pull request template asks for the motivation, verification commands, and any hardware-specific evidence.

All contributors must follow the [Code of Conduct](CODE_OF_CONDUCT.md).
