# Security policy

## Supported versions

Security fixes are applied to the latest code on the `main` branch. Until the project publishes stable releases, older commits are not maintained as separate security-support lines.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's private vulnerability reporting for this repository:

<https://github.com/Sentinel3252/DistLLM/security/advisories/new>

Include the affected revision, environment, reproduction steps, impact, and any suggested mitigation. You should receive an acknowledgement within seven days. The maintainers will coordinate validation, remediation, and disclosure with the reporter.

DistLLM does not load remote models or datasets by default. Reports about vulnerabilities in PyTorch, CUDA, NCCL, or other dependencies should also be disclosed to the relevant upstream project.
