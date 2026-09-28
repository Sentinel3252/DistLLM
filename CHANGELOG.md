# Changelog

All notable changes to DistLLM will be documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Public project documentation in English and Chinese.
- Apache-2.0 project license and third-party attribution notice.
- Contribution, support, security, and community conduct policies.
- GitHub CI, issue forms, pull request template, and Dependabot configuration.
- Package metadata and command-line entry points.

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
