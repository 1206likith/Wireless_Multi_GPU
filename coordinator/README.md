# Layer 2 — Coordinator + Worker Daemon

Minimal multi-machine dispatch for this project's 2-laptop hobby GPU
cluster. Not a scheduler, not HA, not auto-discovering — a static worker
list, one job at a time per worker, polled over plain HTTP.

## Running a worker

On every machine contributing a GPU (including the "main" machine):

```bash
# One-time: generate a shared secret this worker's dispatch endpoint requires.
mkdir -p coordinator/tokens
python -c "import secrets; print(secrets.token_hex(32))" > coordinator/tokens/<worker-name>.secret

python coordinator/worker_daemon.py --port 8770 --token-file coordinator/tokens/<worker-name>.secret
```

By default the daemon binds `127.0.0.1` only. Once Layer 1 (Tailscale +
WSL2 mirrored networking) is proven on that machine, pass its own
Tailscale IP explicitly via `--host 100.x.y.z` — never `0.0.0.0`, so this
never accidentally listens on the raw LAN/internet interface.

## Running the coordinator

On the one "main" machine:

```bash
# coordinator/workers.yaml lists every worker: name, host, port, token_file.
# token_file must contain the SAME secret that worker's daemon was started
# with -- copy it over out-of-band (e.g. a direct file copy), never send it
# over this project's own HTTP protocol.

python coordinator/coordinator.py status                                   # one-shot poll -> coordinator/status.json
python coordinator/coordinator.py watch --interval 10                       # poll continuously
python coordinator/coordinator.py dispatch job-specs/yolo-detect-v4-ddp.yaml # dispatch a real job
```

`status.json`'s shape mirrors `dashboard/index.html`'s hardcoded example
data (a machines list + a jobs list + cluster totals) so wiring the static
dashboard up to live data later is a small fetch+render change, not a
rewrite.

## Security model

- **entry_point allowlist**: `worker_daemon.py`'s `ALLOWED_ENTRY_POINTS` is
  the only set of scripts a dispatch can ever launch. A job spec pointing
  anywhere else — even a real, existing file elsewhere in the repo — is
  refused. Adding a new job type means adding its script to that set
  deliberately, not something a job spec can expand on its own. This is
  the same no-arbitrary-remote-execution line established earlier in the
  project (see `job-specs/SCHEMA.md`).
- **env allowlist**: only `ALLOWED_ENV_KEYS` from a job spec's `env` block
  or the coordinator's per-dispatch values are passed to the subprocess —
  everything else is dropped. An unrestricted env merge would let a caller
  set things like `PYTHONSTARTUP` or override `PATH` to influence what
  `sys.executable` actually loads, which would defeat the entry_point
  allowlist even without any literal code in the request body.
- **bearer-token auth on /dispatch**: every worker requires a shared
  secret (`--token-file`), checked with `hmac.compare_digest`. `/status`
  is unauthenticated (read-only, not sensitive beyond hostname/GPU/job
  name). `/dispatch` without a valid token returns 401.
- **loopback by default**: `--host` defaults to `127.0.0.1`. Binding to a
  real interface is an explicit, deliberate choice at daemon-start time.

These three (found by an automated security review during development,
fixed the same session) were real gaps in an earlier draft: an
unauthenticated dispatch endpoint bound to all interfaces, an entry_point
check that only verified "this file exists in the repo" rather than "this
file is one we actually reviewed for this," and an env-var merge that
would have let dispatch-time data smuggle in effective code execution.

## What's NOT built (honest gaps)

- **No auto-discovery.** `workers.yaml` is hand-edited. Fine for 2-3
  known machines; would need real work to scale further.
- **No real rank/bin-packing logic.** `coordinator.py` assigns ranks in
  worker-list order and only resolves the job spec's `{auto}` placeholder
  to a fixed fallback (`PER_GPU_BATCH=8`), not an actual VRAM-aware
  auto-tuner.
- **No per-epoch training progress.** The worker reports process
  liveness/exit code, not what epoch/mIoU/mAP the training script is
  actually at — the dashboard's "progress" column has nothing real to
  show yet without the training script itself writing progress
  somewhere the daemon can read.
- **No TLS.** Traffic is plain HTTP; relies entirely on Tailscale's own
  WireGuard encryption for transport security between machines, and on
  loopback-only binding for anything not yet on Tailscale.
- **Single coordinator, no HA.** If the coordinator's machine is off,
  nothing dispatches or polls. Acceptable for a 2-laptop hobby cluster.
- **Not yet tested cross-machine.** Everything above is verified via
  `test_e2e.py` on loopback only (no second laptop available at build
  time). The worker list in `workers.yaml` needs a real Tailscale IP
  substituted in for the second laptop, and that path is unverified
  until Layer 1 is proven end-to-end.

## Testing

```bash
python coordinator/test_e2e.py
```

Starts a worker daemon on loopback with a throwaway token, dispatches a
trivial no-op job (`test_noop_job.py`/`.yaml` — NOT the real training
script, to avoid colliding with actual training running on this GPU),
and asserts the full poll → dispatch → run → status → dashboard-shape
loop works, plus the entry_point-allowlist and auth rejections.
