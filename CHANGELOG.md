# Changelog

All notable changes to DistLLM will be documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CPU FP32 per-parameter oracle and communication-record regression coverage.
- Config, training-input, trace-parser, loss, and CLI boundary tests.
- Source-distribution installation and console-entry-point smoke checks.
- `distllm-reproduce --output` and explicit distributed test evidence export.
- Copyable Windows PowerShell setup and CPU validation examples.
- Local CPU launcher with an OS-assigned TCPStore port and explicit libuv opt-out.

- Public project documentation in English and Chinese.
- Apache-2.0 project license and third-party attribution notice.
- Contribution, support, security, and community conduct policies.
- GitHub CI, issue forms, pull request template, and Dependabot configuration.
- Package metadata and command-line entry points.

### Fixed

- Reject mismatched training tensors, invalid dimensions, and invalid benchmark timings.
- Preserve FP64 gradients in quality comparisons and fail invalid GPU study arguments early.
- Keep routine pytest runs from overwriting checked-in validation evidence.
- Reconcile historical summary maxima with per-rank reports and clarify result provenance.

### Changed

- Reframed the project as an auditable distributed-training reference implementation.
- Moved detailed methodology and performance evidence into dedicated documentation.

### Removed

- Repository-external résumé update utility.

## [0.1.0] - 2026-09-19

### Added

- Token-correct gradient accumulation reference path.
- DDP, MLP tensor-parallel, and GPipe-style pipeline-parallel implementations.
- Communication tracing, collective checks, scaling analysis, and GPU profiling scripts.
- Full-batch numerical verification across 2- and 4-process Gloo configurations.

[Unreleased]: https://github.com/Sentinel3252/DistLLM/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Sentinel3252/DistLLM/releases/tag/v0.1.0
