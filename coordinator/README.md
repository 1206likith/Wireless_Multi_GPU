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

# Remote command execution -- see SECURITY MODEL below before using this.
python coordinator/coordinator.py health                                    # report problems, fix nothing
python coordinator/coordinator.py exec second-laptop "df -h"                # run any shell command remotely
python coordinator/coordinator.py exec --all "git pull"                     # ...on every configured worker

# Push a dataset directory from THIS machine to a worker over HTTP.
python coordinator/coordinator.py send-dataset second-laptop data/yolo_detect_v4 \
    --advertise-host 100.x.y.z   # THIS machine's own Tailscale IP -- required,
                                  # get it with `tailscale ip -4`
```

`status.json`'s shape mirrors `dashboard/index.html`'s hardcoded example
data (a machines list + a jobs list + cluster totals) so wiring the static
dashboard up to live data later is a small fetch+render change, not a
rewrite.

## Running the control API (for the web dashboard)

`dashboard/index.html` is a real web app, not a static read-only page —
it has buttons (dispatch, exec, health check) that need something to
call. `control_api.py` is a small HTTP wrapper around the SAME functions
`coordinator.py`'s CLI already uses (no new worker-facing behavior, no
new security surface on the worker side):

```bash
# One-time: generate this API's own bearer token (SEPARATE from any
# worker's dispatch_token.secret -- this one grants everything
# coordinator.py's CLI can do, including /exec on any configured worker).
python -c "import secrets; print(secrets.token_hex(32))" > coordinator/control_api_token.secret

python coordinator/control_api.py --port 8790 --token-file coordinator/control_api_token.secret

# Serve the dashboard itself over HTTP (fetch() needs a real origin, not file://):
cd dashboard && python -m http.server 8080
```

Then open `http://127.0.0.1:8080/index.html`, enter the control API's
base URL (`http://127.0.0.1:8790` by default) and the token from
`control_api_token.secret` into the dashboard's "Control API" panel, and
click Save. The token is stored only in that browser's own
`localStorage` — this project never bakes a secret into a static HTML
file. Endpoints: `GET /api/status` (unauthenticated, read-only, same
data as `status.json`), `POST /api/dispatch` `{job_spec: "<name>"}`,
`POST /api/exec` `{worker, command}`, `POST /api/health`,
`POST /api/send-dataset` `{worker, dataset_path, advertise_host}` — all
POST endpoints require the same bearer-token auth as `/dispatch` does on
a worker.

Binds `127.0.0.1` by default, same reasoning as `worker_daemon.py` —
only bind a real interface deliberately. If you want the dashboard
reachable from another machine on the tailnet, bind `control_api.py` to
this machine's Tailscale IP and set the dashboard's "API base URL" field
to that same IP.

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

### /exec and /dataset-pull — a deliberate exception to the above

`POST /exec` runs an ARBITRARY shell command sent by the coordinator on
the target worker, with no allowlist. This is a deliberate scope
expansion beyond the no-arbitrary-remote-execution design above, added
at the user's explicit request after the tradeoff was spelled out and
confirmed: **if the main laptop or its dispatch tokens are ever
compromised, that gives an attacker code execution on every connected
worker, not just one.** This is accepted as reasonable for a personal
2-3 laptop hobby cluster; it would NOT be reasonable for anything
shared/production. Kept as safe as a fundamentally unsafe feature can
be:
- Same bearer-token auth as `/dispatch`.
- Every command is logged to `coordinator/worker_exec.log` on the
  WORKER (not just the coordinator) before the response is sent, so
  there's always a local record even if the response never arrives.
- A hard 300s timeout so a bad command can't hang the worker forever.
- One command at a time per worker (a lock, not a queue) so two remote
  fixes can't race each other.

`POST /dataset-pull` is narrower in principle (it only ever runs a
fetch, not arbitrary code) but still worth knowing about: it recursively
downloads a directory listing via pure-Python `urllib` (no external
`wget`/`curl` dependency — an earlier version shelled out to `wget`,
which silently failed on native Windows where `wget` is only a
PowerShell alias, not a real executable `subprocess.run` can find; found
by testing, not review). `dest` is validated to stay inside the repo
root the same way `entry_point` is.

**Protect the main laptop and its worker tokens like root credentials
to the entire cluster** — that is genuinely what they are now.

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
- **Cross-machine status polling and remote exec ARE proven** (a real
  second laptop, over its real Tailscale IP, has successfully been
  polled via `/status` and driven via `/exec`/`/dataset-pull`) — but
  `test_e2e.py` itself still only runs the security/auth checks on
  loopback (no second machine available in CI/automated test runs), so
  it can't catch a real cross-machine regression on its own.
- **WSL2 dispatch gap: CLOSED.** `worker_daemon.py` now detects
  WSL2-only entry_points (`WSL2_ENTRY_POINTS`, currently
  `36b_train_yolo_v4_ddp_wsl2.py`) and launches them via
  `wsl.exe -d <distro> -- bash -lc "source venv && torchrun ..."`
  instead of `sys.executable` directly — see `build_launch_argv()`.
  Verified end-to-end on this laptop: a real `/dispatch` HTTP call
  launched `wsl.exe`, which ran `torchrun` inside WSL2's Ubuntu venv,
  which initialized and destroyed a real NCCL process group, exit
  code 0. This is now a permanent regression test in `test_e2e.py`
  (skips gracefully on a machine without WSL2/Ubuntu-24.04 set up).
  **Not yet proven cross-machine** — the smoke test is single-node
  (world_size=1, this laptop only); dispatching the real multi-hour
  YOLO DDP job across both laptops through the coordinator is the
  remaining unverified step, blocked on the second laptop's
  worker_daemon.py actually being reachable (see below).
- **`/api/status` is slow when a worker is unreachable.** Each worker
  poll has its own `HTTP_TIMEOUT_S` (5s) in `coordinator.py`; with one
  worker offline, `/api/status` takes ~5-8s to respond (waits out that
  worker's timeout before returning). Fine for a 10s dashboard poll
  interval with 2 workers, but would compound linearly with more
  offline workers — polls aren't parallelized. Not fixed, since it's not
  wrong, just slow; would need `concurrent.futures` or threads in
  `poll_worker`'s caller if this becomes a real problem at a larger
  worker count.
- **Auto-fix scope is intentionally narrow.** `coordinator.py heal`
  currently mirrors `health`'s findings rather than actually fixing
  anything automatically — see `cmd_heal`'s own docstring for exactly
  why each problem class hit this session (Tailscale account mismatches,
  admin-elevation-gated firewall rules, PyTorch version drift) needs a
  human at the keyboard rather than a script pretending otherwise. What
  `/exec` DOES let you do is run the fix yourself remotely instead of
  physically walking to the other laptop.

## Testing

```bash
python coordinator/test_e2e.py           # coordinator.py CLI <-> worker_daemon.py
python coordinator/test_control_api.py   # control_api.py HTTP wrapper <-> worker_daemon.py
```

`test_e2e.py` starts a worker daemon on loopback with a throwaway token,
dispatches a trivial no-op job (`test_noop_job.py`/`.yaml` — NOT the real
training script, to avoid colliding with actual training running on this
GPU), and asserts the full poll → dispatch → run → status →
dashboard-shape loop works, plus the entry_point-allowlist, WSL2
dispatch, and auth rejections.

`test_control_api.py` starts a worker daemon AND `control_api.py` on
loopback with throwaway tokens, then drives every `/api/*` endpoint
exactly like `dashboard/index.html`'s own JS does (real HTTP calls, same
payload shapes) — asserts real dispatch/exec/health work end-to-end
through the HTTP layer, plus auth rejection and job-spec-name validation
(path traversal, nonexistent names).
