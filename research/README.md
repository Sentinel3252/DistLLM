# Upstream provenance

This directory preserves the source evidence used for DistLLM's motivating case study. It is not part of the installable Python package.

## Snapshot

- Repository: [`allenai/open-instruct`](https://github.com/allenai/open-instruct)
- Commit: `b9269782a3d2c81f2e43ce14ba5290417923f4c0`
- Source path: `open_instruct/finetune.py`
- Local snapshot: `finetune.upstream.py`
- SHA-256: `f3ee4de900ea48cd157c5e7b26dccbc72e622136fd0493043ccca8b1a04b2527`
- Upstream issue: [`allenai/open-instruct#1728`](https://github.com/allenai/open-instruct/issues/1728)
- Snapshot review date: 2026-09-19

The upstream file is unmodified and is distributed under its original Apache-2.0 terms; see `LICENSE.open-instruct`.

## What the evidence supports

At the pinned revision, the ordinary causal-LM path uses `outputs.loss` and passes it to `accelerator.backward(loss)`. A separate sequence-parallel branch applies token weighting, but it does not establish a single supervised-token denominator over the complete ordinary accumulation window.

DistLLM independently reproduces and fixes the general mean-of-microbatch-means mechanism. It is not a patch applied to open-instruct, does not run the complete Tulu training stack, and does not attempt to explain every historical quality observation in the upstream issue.

Useful snapshot locations:

- line 828: enter `accelerator.accumulate(model)`;
- line 832: ordinary causal-LM forward;
- line 835: use `outputs.loss`;
- line 838: sequence-parallel token-weighting branch; and
- line 856: call `accelerator.backward(loss)`.

The issue state and upstream source can change. The commit, hash, and local license make this evidence reproducible without implying that the current upstream project still has the same behavior.
