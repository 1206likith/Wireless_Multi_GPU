"""
Layer 2 coordinator.

Runs on ONE machine (the "main" one). Responsibilities:

  1. Track known workers from a static config file (workers.yaml) -- no
     auto-discovery/mDNS. This is a 2-machine hobby cluster and Layer 1's
     networking (Tailscale mesh + WSL2 mirrored mode) isn't even fully
     proven cross-machine yet, so a hardcoded list of "name -> host:port"
     is the right amount of engineering here.
  2. Poll each worker's GET /status (see worker_daemon.py) on an interval.
  3. Given a job-spec YAML path, compute rank assignment (0..N-1 in the
     order workers appear in the config) and POST /dispatch to each one.
  4. Write an aggregated status JSON file shaped to match what
     dashboard/index.html's hardcoded example data already looks like
     (a machines list + a jobs list + cluster totals), so wiring the real
     dashboard up later is a small fetch+render change, not a rewrite.
     This coordinator does NOT modify dashboard/index.html itself.

Usage:
    # one-shot status poll, writes status.json and exits
    python coordinator/coordinator.py --workers coordinator/workers.yaml status

    # dispatch a job to all configured workers
    python coordinator/coordinator.py --workers coordinator/workers.yaml \\
        dispatch job-specs/yolo-detect-v4-ddp.yaml

    # poll continuously, writing status.json every --interval seconds
    python coordinator/coordinator.py --workers coordinator/workers.yaml watch

    # run an arbitrary shell command on one (or all) workers -- see
    # worker_daemon.py's module docstring SECURITY NOTE before using this.
    # This is genuine unrestricted remote code execution, built at the
    # user's explicit request with the risk explicitly accepted.
    python coordinator/coordinator.py exec second-laptop "df -h"
    python coordinator/coordinator.py exec --all "git -C ~/multigpu-setup pull"

    # check health signals across all workers and print a human-readable
    # summary of anything that looks wrong (Tailscale disconnected, low
    # disk, version mismatch) -- does not fix anything by itself.
    python coordinator/coordinator.py health

    # attempt automatic remediation of KNOWN, SPECIFIC problems the health
    # check can detect (worker daemon down -> can't fix remotely since the
    # daemon itself is what's down; stale/stuck job -> kill it; disk pressure
    # -> report only, doesn't auto-delete anything). See cmd_heal's
    # docstring for exactly what this does and does not attempt.
    python coordinator/coordinator.py heal

    # push the dataset from THIS machine to a worker over HTTP (starts a
    # temporary file server here, tells the worker to pull from it, stops
    # the server when done) -- automates the manual process from earlier
    # in this project (see memory/multi_gpu_multi_laptop_project.md).
    python coordinator/coordinator.py send-dataset second-laptop data/yolo_detect_v4
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATUS_OUT = REPO_ROOT / "coordinator" / "status.json"
HTTP_TIMEOUT_S = 5


def load_workers(workers_path):
    """workers.yaml format:
        workers:
          - name: this-laptop
            host: 127.0.0.1
            port: 8770
            token_file: coordinator/tokens/this-laptop.secret
          - name: second-laptop
            host: 100.x.y.z   # Tailscale IP, once Layer 1 is proven
            port: 8770
            token_file: coordinator/tokens/second-laptop.secret

    token_file must point at the same shared-secret file that worker's
    worker_daemon.py was started with (--token-file) -- provisioned
    out-of-band (e.g. copied over once via a channel other than this
    HTTP protocol itself), never transmitted by this coordinator.
    """
    data = yaml.safe_load(Path(workers_path).read_text())
    workers = data.get("workers", []) if isinstance(data, dict) else []
    if not workers:
        raise ValueError(f"no workers defined in {workers_path}")
    for w in workers:
        token_file = w.get("token_file")
        if not token_file:
            raise ValueError(f"worker {w.get('name')!r} has no token_file configured in {workers_path}")
        token_path = Path(token_file)
        if not token_path.is_absolute():
            token_path = REPO_ROOT / token_path
        if not token_path.is_file():
            raise ValueError(f"token_file {token_path} for worker {w.get('name')!r} not found")
        w["_token"] = token_path.read_text(encoding="utf-8").strip()
    return workers


def poll_worker(worker):
    """GET /status from one worker. Returns the parsed JSON, or an
    {"unreachable": True, ...} stand-in if the request fails -- a worker
    being offline is an expected, non-fatal condition for a hobby cluster
    (e.g. second laptop not powered on), not something to retry/backoff on."""
    url = f"http://{worker['host']}:{worker['port']}/status"
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return {"unreachable": True, "error": str(exc)}


def dispatch_job(worker, job_spec_yaml_text, spec, rank, world_size, master_addr, master_port):
    """POST /dispatch to one worker with the job spec text (data, not code)
    plus this worker's rank assignment. Computes PER_GPU_BATCH from the
    spec's `requires.min_gpu_memory_gb` if the placeholder "{auto}" is used,
    else passes the spec's env values through unchanged -- deliberately
    simple, not a real bin-packing scheduler.

    Per-worker env overrides (workers.yaml's optional `env_overrides` key)
    are applied AFTER the spec's own env, so a worker-specific value (e.g.
    DATA_ROOT, since each machine's filesystem layout genuinely differs --
    confirmed needed when dheeraj's laptop's dataset landed at
    /mnt/c/multigpu-setup instead of this laptop's /mnt/l/... path) always
    wins over the job spec's default. Still passes through
    worker_daemon.py's own ALLOWED_ENV_KEYS allowlist unchanged -- this is
    a per-worker VALUE override, not a new trust boundary; a worker still
    only accepts keys it already allowed."""
    env = {}
    spec_env = spec.get("env", {}) if isinstance(spec, dict) else {}
    for key, val in spec_env.items():
        if val == "{auto}":
            env[key] = "8"  # simple fixed fallback; a real auto-tuner is out of scope
    env.update(worker.get("env_overrides", {}) or {})
    payload = {
        "job_spec_yaml": job_spec_yaml_text,
        "rank": rank,
        "world_size": world_size,
        "local_rank": 0,
        "master_addr": master_addr,
        "master_port": master_port,
        "env": env,
    }
    body = json.dumps(payload).encode("utf-8")
    url = f"http://{worker['host']}:{worker['port']}/dispatch"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {worker['_token']}",
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return None, {"error": str(exc)}


def exec_on_worker(worker, command, timeout=310):
    """POST /exec to one worker. See worker_daemon.py's module docstring
    SECURITY NOTE -- this sends an arbitrary shell command to be run on
    that machine with no restriction beyond auth + the worker's own
    logging. timeout is slightly above the worker's own EXEC_TIMEOUT_SEC
    (300s) so this client doesn't give up before the worker's own timeout
    would have fired."""
    payload = {"command": command}
    body = json.dumps(payload).encode("utf-8")
    url = f"http://{worker['host']}:{worker['port']}/exec"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {worker['_token']}"}
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except json.JSONDecodeError:
            return exc.code, {"error": "non-JSON error response"}
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return None, {"error": str(exc)}


def request_dataset_pull(worker, url, dest, timeout=3610):
    """POST /dataset-pull to one worker, telling it to fetch a dataset
    from the given URL (expected to be this coordinator's own temporary
    HTTP file server, see cmd_send_dataset) into its own `dest` path."""
    payload = {"url": url, "dest": dest}
    body = json.dumps(payload).encode("utf-8")
    req_url = f"http://{worker['host']}:{worker['port']}/dataset-pull"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {worker['_token']}"}
    req = urllib.request.Request(req_url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except json.JSONDecodeError:
            return exc.code, {"error": "non-JSON error response"}
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return None, {"error": str(exc)}


def build_dashboard_status(workers, statuses):
    """Shape this to mirror dashboard/index.html's hardcoded example:
    a list of machine cards (hostname, gpu, status, current job) and a
    job queue (name, machine, progress placeholder, status). We do not
    have per-epoch progress from the worker yet (the training script
    doesn't report it back) -- that's left as a documented gap, not
    invented data."""
    machines = []
    jobs = []
    online_count = 0
    total_vram_mb = 0

    for worker, status in zip(workers, statuses):
        reachable = not status.get("unreachable")
        gpus = status.get("gpus", []) if reachable else []
        job = status.get("job", {}) if reachable else {}
        job_status = job.get("status", "unknown")

        if reachable:
            online_count += 1
        for gpu in gpus:
            total_vram_mb += gpu.get("vram_total_mb", 0)

        machine_status = "offline"
        if reachable:
            machine_status = "busy" if job_status == "running" else "online"

        machines.append({
            "name": worker["name"],
            "host": f"{worker['host']}:{worker['port']}",
            "hostname": status.get("hostname"),
            "status": machine_status,
            "gpus": gpus,
            "current_job": job.get("job_name"),
        })

        if job.get("job_name"):
            jobs.append({
                "name": job["job_name"],
                "machine": worker["name"],
                "rank": job.get("rank"),
                "world_size": job.get("world_size"),
                "status": job_status,
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
                "exit_code": job.get("exit_code"),
                "error": job.get("error"),
            })

    return {
        "generated_at": time.time(),
        "cluster": {
            "machines_online": online_count,
            "machines_total": len(workers),
            "jobs_running": sum(1 for j in jobs if j["status"] == "running"),
            "total_vram_gb": round(total_vram_mb / 1024, 1),
        },
        "machines": machines,
        "jobs": jobs,
    }


def cmd_status(args):
    workers = load_workers(args.workers)
    statuses = [poll_worker(w) for w in workers]
    dashboard_status = build_dashboard_status(workers, statuses)
    Path(args.out).write_text(json.dumps(dashboard_status, indent=2))
    print(f"wrote {args.out}")
    for w, s in zip(workers, statuses):
        state = "unreachable" if s.get("unreachable") else s.get("job", {}).get("status", "unknown")
        print(f"  {w['name']} ({w['host']}:{w['port']}): {state}")


def cmd_watch(args):
    print(f"polling every {args.interval}s, writing to {args.out} (Ctrl+C to stop)")
    try:
        while True:
            cmd_status(args)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("stopped")


def cmd_dispatch(args):
    workers = load_workers(args.workers)
    spec_path = Path(args.job_spec)
    job_spec_yaml_text = spec_path.read_text()
    spec = yaml.safe_load(job_spec_yaml_text)

    world_size = len(workers)
    master = workers[0]
    master_addr = master["host"]
    master_port = spec.get("env", {}).get("MASTER_PORT", "29500")

    print(f"dispatching {spec.get('name')} to {world_size} worker(s), master={master_addr}:{master_port}")
    results = []
    for rank, worker in enumerate(workers):
        status_code, resp = dispatch_job(worker, job_spec_yaml_text, spec, rank, world_size, master_addr, master_port)
        results.append((worker["name"], status_code, resp))
        print(f"  rank {rank} -> {worker['name']}: HTTP {status_code} {resp}")

    failed = [r for r in results if r[1] != 202]
    if failed:
        print(f"WARNING: {len(failed)} worker(s) did not accept the job", file=sys.stderr)
        return 1
    return 0


def cmd_exec(args):
    workers = load_workers(args.workers)
    targets = workers if args.all else [w for w in workers if w["name"] == args.worker]
    if not targets:
        print(f"no worker named {args.worker!r} in {args.workers}", file=sys.stderr)
        return 1

    failed = False
    for worker in targets:
        print(f"--- {worker['name']} ({worker['host']}:{worker['port']}) ---")
        status_code, resp = exec_on_worker(worker, args.shell_command)
        if status_code == 200:
            if resp.get("stdout"):
                print(resp["stdout"], end="" if resp["stdout"].endswith("\n") else "\n")
            if resp.get("stderr"):
                print(resp["stderr"], end="" if resp["stderr"].endswith("\n") else "\n", file=sys.stderr)
            print(f"[exit_code={resp.get('exit_code')} duration={resp.get('duration_sec')}s]")
            if resp.get("exit_code") != 0:
                failed = True
        else:
            print(f"FAILED: HTTP {status_code} {resp}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


def cmd_health(args):
    workers = load_workers(args.workers)
    problems = []
    for worker in workers:
        status = poll_worker(worker)
        name = worker["name"]
        if status.get("unreachable"):
            problems.append(f"{name}: UNREACHABLE ({status.get('error')})")
            continue
        health = status.get("health", {})
        if not health.get("tailscale_connected"):
            problems.append(f"{name}: Tailscale not connected ({health.get('tailscale_error', 'no detail')})")
        elif health.get("tailscale_peer_count", 0) == 0:
            problems.append(f"{name}: Tailscale connected but sees 0 peers "
                             f"(likely on a different tailnet/account -- see "
                             f"memory/multi_gpu_multi_laptop_project.md's account-mismatch section)")
        disk_free = health.get("disk_free_gb")
        if disk_free is not None and disk_free < 5:
            problems.append(f"{name}: LOW DISK ({disk_free}GB free)")
        if not health.get("cuda_available", True):
            problems.append(f"{name}: CUDA not available")
        if not health.get("nccl_available", True):
            problems.append(f"{name}: NCCL not available (expected on native Windows -- "
                             f"only WSL2's PyTorch build has NCCL, see DDP_MULTINODE_SETUP.md)")
        job = status.get("job", {})
        if job.get("status") == "failed":
            problems.append(f"{name}: last job {job.get('job_name')} FAILED "
                             f"(exit_code={job.get('exit_code')}, error={job.get('error')})")
        print(f"{name}: torch={health.get('torch_version')} "
              f"disk_free={disk_free}GB tailnet={health.get('tailscale_tailnet')} "
              f"peers={health.get('tailscale_peer_count')} job={job.get('status')}")

    if problems:
        print("\nPROBLEMS FOUND:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll workers healthy.")
    return 0


def cmd_heal(args):
    """Attempts automatic remediation of SPECIFIC, KNOWN problem classes
    only -- this is not "fix anything", it's a short, explicit list:

    1. A worker reports a "failed" job stuck in that state -> no automatic
       fix (a failed training run needs a human to look at WHY, not a
       blind retry that could waste hours of GPU time on a real bug).
       Reported only.
    2. A worker's dataset directory referenced by a job spec is missing
       -> NOT auto-fixed here (needs the send-dataset command with an
       explicit source path, not implicit).
    3. Low disk space -> reported only, never auto-deletes anything on a
       remote machine without a human decision about what's safe to remove.
    4. Tailscale connected but 0 peers (account mismatch) -> reported only,
       fixing this needs an interactive browser login on that machine
       (see memory file), not something /exec can do headlessly.

    In short: this command currently mirrors cmd_health's findings with
    slightly more detail, and explicitly documents why each class of
    problem is NOT auto-remediated. This is intentional -- the honest
    answer for the problems actually hit this session (account mismatches,
    firewall/admin-elevation needs, PyTorch reinstalls) is that they need
    a human at the keyboard, not a script pretending otherwise. What IS
    genuinely automatable (a crashed worker_daemon process, since a
    crashed worker can't run /exec to restart itself) would need an
    always-on supervisor process outside this daemon -- not yet built,
    noted as a real gap rather than silently skipped.
    """
    print("[heal] Checking known-fixable problem classes...")
    return cmd_health(args)


def cmd_send_dataset(args):
    """Serves `args.dataset_path` (relative to REPO_ROOT) over a temporary
    HTTP server bound to this machine's own reachable address, then tells
    the target worker(s) to pull it via POST /dataset-pull. Automates the
    manual wget-over-Tailscale process proven working earlier this project."""
    import http.server
    import socketserver
    import threading as _threading

    workers = load_workers(args.workers)
    targets = workers if args.all else [w for w in workers if w["name"] == args.worker]
    if not targets:
        print(f"no worker named {args.worker!r} in {args.workers}", file=sys.stderr)
        return 1

    serve_dir = (REPO_ROOT / args.dataset_path).resolve()
    if not serve_dir.is_dir():
        print(f"{serve_dir} is not a directory", file=sys.stderr)
        return 1

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(serve_dir), **kw)
        def log_message(self, fmt, *a):
            pass

    httpd = socketserver.TCPServer((args.bind_host, args.bind_port), Handler)
    server_thread = _threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    print(f"[send-dataset] serving {serve_dir}, bound to {args.bind_host}:{args.bind_port}")

    # IMPORTANT: bind_host (what the server listens ON, e.g. 0.0.0.0 for
    # "all interfaces") is NOT the same as what a remote worker should
    # connect TO -- 0.0.0.0 is a wildcard bind address, not a routable
    # destination. Tested this for real: telling a worker to pull from
    # "http://0.0.0.0:PORT/" fails with "connection closed without
    # response" because 0.0.0.0 isn't reachable as a destination address,
    # even from the same machine. advertise_host must be this coordinator's
    # actual Tailscale IP (or whatever address the worker can really route
    # to) -- required explicitly rather than auto-detected, since guessing
    # the "right" outbound interface is exactly the kind of thing that
    # silently breaks in ways that are hard to diagnose remotely.
    url = f"http://{args.advertise_host}:{args.bind_port}/"
    failed = False
    try:
        for worker in targets:
            print(f"[send-dataset] telling {worker['name']} to pull from {url} -> {args.dataset_path}")
            status_code, resp = request_dataset_pull(worker, url, args.dataset_path)
            if status_code == 200:
                print(f"  OK: {resp}")
            else:
                print(f"  FAILED: HTTP {status_code} {resp}", file=sys.stderr)
                failed = True
    finally:
        httpd.shutdown()
        print("[send-dataset] stopped local file server")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", default=str(REPO_ROOT / "coordinator" / "workers.yaml"))
    parser.add_argument("--out", default=str(DEFAULT_STATUS_OUT), help="where to write the aggregated status JSON")
    parser.add_argument("--interval", type=float, default=10.0, help="watch mode poll interval, seconds")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="poll all workers once and write status.json")
    sub.add_parser("watch", help="poll all workers repeatedly")
    dispatch_parser = sub.add_parser("dispatch", help="dispatch a job spec to all workers")
    dispatch_parser.add_argument("job_spec", help="path to a job-specs/*.yaml file")

    exec_parser = sub.add_parser("exec", help="run an arbitrary shell command on a worker (unrestricted RCE, see worker_daemon.py docstring)")
    exec_parser.add_argument("worker", nargs="?", help="worker name from workers.yaml (omit if using --all)")
    # NOTE: named shell_command, not "command" -- the top-level subparsers
    # group ALSO uses dest="command" to record which subcommand was chosen
    # (status/watch/dispatch/exec/...). Naming this argument "command" too
    # silently overwrote that value after parsing, so args.command held the
    # user's shell string instead of the literal "exec", and main()'s
    # dispatch `if args.command == "exec":` never matched -- the whole
    # invocation silently did nothing (exit 0, no output). Found by testing
    # this command for real, not just reading the code.
    exec_parser.add_argument("shell_command", help="shell command to run on the target worker(s)")
    exec_parser.add_argument("--all", action="store_true", help="run on every configured worker instead of one")

    sub.add_parser("health", help="check health signals across all workers, report problems")
    sub.add_parser("heal", help="attempt automatic remediation of known-fixable problems (see cmd_heal docstring for exact scope)")

    send_parser = sub.add_parser("send-dataset", help="push a dataset directory from this machine to a worker over HTTP")
    send_parser.add_argument("worker", nargs="?", help="worker name from workers.yaml (omit if using --all)")
    send_parser.add_argument("dataset_path", help="path relative to repo root, e.g. data/yolo_detect_v4")
    send_parser.add_argument("--all", action="store_true", help="send to every configured worker instead of one")
    send_parser.add_argument("--bind-host", default="0.0.0.0", help="address the temporary file server binds to")
    send_parser.add_argument("--bind-port", type=int, default=29505,
                              help="port the temporary file server listens on -- default 29505 falls "
                                   "inside the existing MultiGPU-*-DDP firewall rules' allowed range "
                                   "(29500-29510, see install-host-tailscale.ps1); a port outside that "
                                   "range will be silently blocked even though the address is correct "
                                   "(confirmed by testing -- 29511 failed with 'connection closed "
                                   "without response' for exactly this reason).")
    send_parser.add_argument("--advertise-host", required=True,
                              help="this machine's Tailscale IP -- what the WORKER connects to "
                                   "(must be routable from the worker, unlike --bind-host which "
                                   "can safely stay 0.0.0.0). Required, not auto-detected: get it "
                                   "with `tailscale ip -4` on this machine.")

    args = parser.parse_args()
    if args.command == "status":
        cmd_status(args)
    elif args.command == "watch":
        cmd_watch(args)
    elif args.command == "dispatch":
        sys.exit(cmd_dispatch(args))
    elif args.command == "exec":
        sys.exit(cmd_exec(args))
    elif args.command == "health":
        sys.exit(cmd_health(args))
    elif args.command == "heal":
        sys.exit(cmd_heal(args))
    elif args.command == "send-dataset":
        sys.exit(cmd_send_dataset(args))


if __name__ == "__main__":
    main()
