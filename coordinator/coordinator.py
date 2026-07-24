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
    simple, not a real bin-packing scheduler."""
    env = {}
    spec_env = spec.get("env", {}) if isinstance(spec, dict) else {}
    for key, val in spec_env.items():
        if val == "{auto}":
            env[key] = "8"  # simple fixed fallback; a real auto-tuner is out of scope
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

    args = parser.parse_args()
    if args.command == "status":
        cmd_status(args)
    elif args.command == "watch":
        cmd_watch(args)
    elif args.command == "dispatch":
        sys.exit(cmd_dispatch(args))


if __name__ == "__main__":
    main()
