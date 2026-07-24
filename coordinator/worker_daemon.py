"""
Layer 2 worker daemon.

Runs on EVERY machine that contributes a GPU to the cluster (including the
"main" machine, which runs this alongside coordinator.py). Responsibilities:

  1. Report status: expose a tiny stdlib-only HTTP server on GET /status
     that returns this machine's hostname, GPU name/VRAM, and current job
     state as JSON. No dependency beyond the standard library + torch
     (already installed for training) -- deliberately not using Flask/
     FastAPI/etc, since a single read-only JSON endpoint doesn't need a
     framework.
  2. Accept a job dispatch: POST /dispatch with a JSON body containing the
     job-spec YAML (as text) plus coordinator-assigned rank/world_size/
     master_addr/master_port. Validates the spec's entry_point against the
     HARD SECURITY CONSTRAINT from job-specs/SCHEMA.md: entry_point must be
     a path that already exists in THIS repo checkout, resolved and checked
     to stay inside the repo root (no "../" escapes, no absolute paths
     elsewhere on disk). This daemon never fetches, receives, or executes
     any code that isn't already sitting on disk as part of this repo --
     the POST body only ever supplies data (env var values, rank numbers),
     never code.
  3. Launch the job as a subprocess with the job spec's `env` block plus
     the coordinator-assigned rank/world_size/master_addr/master_port,
     and track its liveness/exit code for subsequent /status reports.

Only one job runs at a time per worker (this is a 2-machine hobby cluster,
not a scheduler) -- a dispatch while a job is already running is rejected.

Usage:
    python coordinator/worker_daemon.py [--port 8770] [--repo-root PATH]

Run this from inside the repo (or pass --repo-root) so relative entry_point
paths resolve correctly. On a real second laptop, this runs against that
machine's own checkout of this repo, at whatever commit was pulled -- the
coordinator never pushes code, only a job-spec's data fields (see dispatch
validation above).
"""
import argparse
import hmac
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 8770  # arbitrary, unlikely to collide with anything training-related

# Only these entry_point paths may ever be dispatched -- a security review
# correctly flagged that "exists inside the repo" alone is too weak a gate
# (any script later added to the repo would become launchable). This list
# is the actual reviewed set; adding a new job type means adding its script
# here deliberately, not something a job spec can expand on its own.
ALLOWED_ENTRY_POINTS = {
    "scripts/36b_train_yolo_v4_ddp_wsl2.py",
    "coordinator/test_noop_job.py",  # trivial local test job, see test_e2e.py
}

# Job-spec / coordinator-assigned env keys this daemon will actually pass
# through to the subprocess. Anything else is dropped, not merged --
# an unrestricted env merge lets a caller set PYTHONSTARTUP, override PATH,
# or otherwise influence what sys.executable actually loads, which defeats
# the entry_point allowlist even though the POST body itself contains no
# code. Extend this list only for keys the pinned training scripts genuinely
# read (see their own docstrings for the env vars they expect).
ALLOWED_ENV_KEYS = {
    "PER_GPU_BATCH", "MASTER_ADDR", "MASTER_PORT",
    "RANK", "WORLD_SIZE", "LOCAL_RANK", "DATA_ROOT",
}

# Shared-secret token required on /dispatch. Loaded from a file so the
# secret itself never has to be typed into a command line or committed to
# git; generate one with e.g. `python -c "import secrets;
# print(secrets.token_hex(32))" > coordinator/dispatch_token.secret` on each
# worker, out-of-band (not over this HTTP endpoint).
TOKEN_PATH = Path(__file__).resolve().parent / "dispatch_token.secret"


def get_gpu_info():
    """Return [{name, vram_total_mb, vram_used_mb}, ...] via nvidia-smi, or
    [] if nvidia-smi isn't on PATH (e.g. this worker has no GPU, or it's
    being polled from inside a WSL2 shell where the binary path differs)."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        out = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            name, total, used = parts
            gpus.append({
                "name": name,
                "vram_total_mb": int(float(total)),
                "vram_used_mb": int(float(used)),
            })
    return gpus


class JobState:
    """Tracks the single in-flight job (if any) for this worker. A plain
    object with a lock is enough here -- one job at a time, no queue."""

    def __init__(self):
        self.lock = threading.Lock()
        self.job_name = None
        self.proc = None
        self.started_at = None
        self.finished_at = None
        self.exit_code = None
        self.rank = None
        self.world_size = None
        self.error = None

    def snapshot(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is not None and self.exit_code is None:
                self.exit_code = self.proc.returncode
                self.finished_at = time.time()

            if self.job_name is None:
                status = "idle"
            elif self.proc is not None and self.proc.poll() is None:
                status = "running"
            elif self.exit_code == 0:
                status = "done"
            elif self.error is not None:
                status = "rejected"
            else:
                status = "failed"

            return {
                "status": status,
                "job_name": self.job_name,
                "rank": self.rank,
                "world_size": self.world_size,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "exit_code": self.exit_code,
                "error": self.error,
            }

    def try_start(self, job_name, argv, env, rank, world_size):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return False, "a job is already running on this worker"
            self.job_name = job_name
            self.rank = rank
            self.world_size = world_size
            self.started_at = time.time()
            self.finished_at = None
            self.exit_code = None
            self.error = None
            try:
                self.proc = subprocess.Popen(
                    argv, cwd=str(REPO_ROOT), env=env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                self.error = f"failed to launch subprocess: {exc}"
                self.proc = None
                return False, self.error
            return True, None


JOB = JobState()


def validate_entry_point(entry_point_str):
    """HARD CONSTRAINT (job-specs/SCHEMA.md): entry_point must be one of the
    explicitly reviewed ALLOWED_ENTRY_POINTS, and must resolve to a real
    file inside this repo checkout. A security review correctly flagged
    that "any path that exists in the repo" is too permissive a gate on its
    own (it would auto-allow any future file); the allowlist is the actual
    gate, existence-on-disk is just a sanity check. realpath is used on
    both sides so a symlink inside the repo can't point outside it and
    slip past the containment check. Returns the resolved absolute Path on
    success, raises ValueError with a clear reason on failure."""
    if not entry_point_str or entry_point_str not in ALLOWED_ENTRY_POINTS:
        raise ValueError(f"entry_point {entry_point_str!r} is not in the allowed set — refused")
    if Path(entry_point_str).is_absolute():
        raise ValueError(f"entry_point must be a relative path within the repo, got: {entry_point_str!r}")
    repo_root_real = Path(os.path.realpath(REPO_ROOT))
    candidate = Path(os.path.realpath(REPO_ROOT / entry_point_str))
    try:
        candidate.relative_to(repo_root_real)
    except ValueError:
        raise ValueError(f"entry_point {entry_point_str!r} resolves outside the repo root — refused")
    if not candidate.is_file():
        raise ValueError(f"entry_point {entry_point_str!r} does not exist in this repo checkout — refused")
    return candidate


class Handler(BaseHTTPRequestHandler):
    server_version = "WorkerDaemon/1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[worker_daemon] " + (fmt % args) + "\n")

    def _send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/status":
            self._send_json(404, {"error": "not found"})
            return
        self._send_json(200, {
            "hostname": socket.gethostname(),
            "gpus": get_gpu_info(),
            "job": JOB.snapshot(),
            "time": time.time(),
        })

    def _check_auth(self):
        """Require a bearer token matching this worker's dispatch_token.secret,
        compared with hmac.compare_digest to avoid a timing side-channel.
        A security review correctly flagged /dispatch as an unauthenticated
        remote-code-launch endpoint -- this is the fix. The token is
        provisioned out-of-band per worker, never sent by the coordinator
        over any channel this daemon itself exposes."""
        expected = self.server.dispatch_token
        if expected is None:
            return False
        got = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not got.startswith(prefix):
            return False
        return hmac.compare_digest(got[len(prefix):], expected)

    def do_POST(self):
        if self.path != "/dispatch":
            self._send_json(404, {"error": "not found"})
            return
        if not self._check_auth():
            self._send_json(401, {"error": "missing or invalid Authorization bearer token"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        job_spec_yaml = payload.get("job_spec_yaml")
        rank = payload.get("rank")
        world_size = payload.get("world_size")
        master_addr = payload.get("master_addr")
        master_port = payload.get("master_port")
        assigned_env = payload.get("env", {})  # coordinator-computed values, e.g. PER_GPU_BATCH

        if job_spec_yaml is None or rank is None or world_size is None:
            self._send_json(400, {"error": "job_spec_yaml, rank, world_size are required"})
            return

        try:
            spec = yaml.safe_load(job_spec_yaml)
        except yaml.YAMLError as exc:
            self._send_json(400, {"error": f"job spec is not valid YAML: {exc}"})
            return

        entry_point = spec.get("entry_point", "") if isinstance(spec, dict) else ""
        try:
            script_path = validate_entry_point(entry_point)
        except ValueError as exc:
            JOB.error = str(exc)
            self._send_json(403, {"error": str(exc)})
            return

        # Build the subprocess environment starting from a minimal explicit
        # base (not a copy of this daemon's own os.environ) plus ONLY keys
        # in ALLOWED_ENV_KEYS from the job spec / coordinator. A security
        # review correctly flagged the previous "merge everything the
        # caller sends" approach as an RCE path: an attacker-controlled env
        # var like PYTHONSTARTUP, PATH, or a *_OPTIONS var can change what
        # sys.executable actually loads/runs even though the POST body
        # contains no literal code. Only pass through what the pinned
        # training scripts are documented to actually read.
        env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""),
               "SystemRoot": os.environ.get("SystemRoot", "")}

        def _apply_allowed(source):
            for k, v in source.items():
                key = str(k)
                if key in ALLOWED_ENV_KEYS:
                    env[key] = str(v)

        if isinstance(spec.get("env"), dict):
            _apply_allowed(spec["env"])
        if isinstance(assigned_env, dict):
            _apply_allowed(assigned_env)

        env["RANK"] = str(rank)
        env["WORLD_SIZE"] = str(world_size)
        env["LOCAL_RANK"] = str(payload.get("local_rank", 0))
        if master_addr:
            env["MASTER_ADDR"] = str(master_addr)
        if master_port:
            env["MASTER_PORT"] = str(master_port)

        argv = [sys.executable, str(script_path)]
        ok, err = JOB.try_start(spec.get("name", entry_point), argv, env, rank, world_size)
        if not ok:
            self._send_json(409, {"error": err})
            return
        self._send_json(202, {"accepted": True, "job_name": spec.get("name"), "pid": JOB.proc.pid})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1",
                         help="bind address; defaults to loopback-only. Pass the "
                              "machine's own Tailscale IP explicitly (never 0.0.0.0) "
                              "once Layer 1 is proven, so this never listens on the "
                              "raw LAN/internet interface even by accident.")
    parser.add_argument("--token-file", default=str(TOKEN_PATH),
                         help="path to the shared-secret token file required on "
                              "/dispatch's Authorization header; generate with "
                              "`python -c \"import secrets; print(secrets.token_hex(32))\"`")
    args = parser.parse_args()

    token_path = Path(args.token_file)
    if not token_path.is_file():
        print(f"[worker_daemon] FATAL: token file {token_path} not found. "
              f"Generate one out-of-band first:\n"
              f'  python -c "import secrets; print(secrets.token_hex(32))" > {token_path}',
              file=sys.stderr)
        sys.exit(1)
    dispatch_token = token_path.read_text(encoding="utf-8").strip()
    if not dispatch_token:
        print(f"[worker_daemon] FATAL: token file {token_path} is empty.", file=sys.stderr)
        sys.exit(1)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.dispatch_token = dispatch_token
    print(f"[worker_daemon] listening on {args.host}:{args.port}, repo root {REPO_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
