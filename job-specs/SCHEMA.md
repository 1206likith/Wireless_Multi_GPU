# Job Spec Format

A job spec describes ONE training job that can run across the connected
GPU cluster. This is Layer 3's core scaffolding — designed now, wired
into the Layer 2 coordinator once networking (Layer 1) is proven.

## Hard security constraint (do not relax this)

Job specs reference a **specific, versioned training script already
present in this repo** (or a signed release of it) — they never contain
or fetch arbitrary code to execute. A worker only ever runs code that:
1. Was cloned from this repo at a specific, pinned commit/tag, AND
2. Is one of a small, reviewed set of entry-point scripts.

This is the same hard line established earlier in this project: no
live code-push channel from a coordinator to a worker, ever. Adding a
new job TYPE means adding a new reviewed script + spec to this repo and
cutting a new version — not something a coordinator can inject at
runtime.

## Spec fields (YAML)

```yaml
# job-specs/example-yolo-detection.yaml
name: yolo-detect-v4-ddp          # unique job identifier
version: 1                         # bump on any change to this spec
entry_point: scripts/36b_train_yolo_v4_ddp_wsl2.py  # must exist in THIS repo, pinned by commit
description: >
  Multi-node YOLOv8s detection training on yolo_detect_v4 dataset via
  DDP over WSL2+NCCL.

requires:
  min_gpu_memory_gb: 8
  dataset: yolo_detect_v4          # logical dataset name, resolved locally per-machine
  python_env: yolo_ddp_env          # venv name expected inside WSL2

roles:
  # How workers coordinate for this specific job. For DDP jobs, one
  # worker is rank 0 (master) and the rest are ranks 1..N-1.
  rank_assignment: coordinator-assigned  # vs. "manual" for the old 2-laptop-by-hand flow

runtime: wsl2  # optional, defaults to native. "wsl2" means worker_daemon.py must launch this
               # entry_point inside WSL2 via torchrun (NCCL is only available there, not on
               # native Windows PyTorch). The actual launch-strategy decision lives in
               # worker_daemon.py's WSL2_ENTRY_POINTS set (a reviewed allowlist, same reasoning
               # as ALLOWED_ENTRY_POINTS) -- this field documents intent, it does not by itself
               # change what the daemon does; a new WSL2 script needs adding to that set too.

env:
  # Environment variables the entry_point script expects, with the
  # coordinator filling in machine-specific values (IP, rank, etc) at
  # dispatch time -- these are NOT arbitrary, they're a fixed allow-list
  # per job spec.
  PER_GPU_BATCH: "{auto}"  # coordinator picks based on requires.min_gpu_memory_gb vs this machine's actual VRAM
```

## Why YAML, not Python

A job spec is inert data, not code. This is deliberate: a malformed or
even malicious job spec YAML can, at worst, point at a script that
doesn't exist (fails safely) or set an unexpected environment variable
for an ALREADY-REVIEWED script — it cannot introduce new code execution
paths the way a Python file could if the coordinator were allowed to
serve one directly.

## Status

Scaffolding only as of 2026-07-24 — no coordinator exists yet to parse/
dispatch these specs (see Layer 2). This file and format may change once
Layer 1 (networking) is proven and Layer 2 is actually built against it.
