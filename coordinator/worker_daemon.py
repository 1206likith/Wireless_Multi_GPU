"""
Layer 2 worker daemon.

Runs on EVERY machine that contributes a GPU to the cluster (including the
"main" machine, which runs this alongside coordinator.py). Responsibilities:

  1. Report status: expose a tiny stdlib-only HTTP server on GET /status
     that returns this machine's hostname, GPU name/VRAM, health-check
     signals (Tailscale connectivity, disk space, PyTorch version), and
     current job state as JSON. No dependency beyond the standard library
     + torch (already installed for training) -- deliberately not using
     Flask/FastAPI/etc, since this doesn't need a framework.
  2. Accept a job dispatch: POST /dispatch with a JSON body containing the
     job-spec YAML (as text) plus coordinator-assigned rank/world_size/
     master_addr/master_port. Validates the spec's entry_point against the
     allowlist in ALLOWED_ENTRY_POINTS.
  3. Launch the job as a subprocess with the job spec's `env` block plus
     the coordinator-assigned rank/world_size/master_addr/master_port,
     and track its liveness/exit code for subsequent /status reports.
  4. POST /exec -- run an ARBITRARY shell command sent by the coordinator
     and return its stdout/stderr/exit code. SECURITY NOTE, read this
     before touching this endpoint: this is a deliberate scope expansion
     beyond this project's earlier "coordinator never runs arbitrary code
     on a worker" design, made at the user's explicit request and with the
     tradeoff explicitly confirmed (main-laptop compromise = code execution
     on every connected worker). Kept as safe as a fundamentally unsafe
     feature can be: same bearer-token auth as /dispatch, every command is
     logged (to worker_exec.log) with timestamp + command + exit code
     before/after execution so there's always a local record, and a hard
     timeout so a bad command can't hang the worker forever. This is NOT
     sandboxed, NOT restricted to an allowlist -- it is real, unrestricted
     remote code execution. Treat the main laptop and its dispatch tokens
     accordingly (protect them like root credentials to every machine in
     the cluster).
  5. POST /dataset-pull -- fetches a dataset from a given HTTP URL (the
     main laptop is expected to be running scripts/47_dataset_server.py
     or equivalent) into this worker's data/ directory, using the same
     wget-over-Tailscale pattern manually proven working earlier this
     project (see memory/multi_gpu_multi_laptop_project.md). Automates
     what was previously a manual multi-step process.

Only one training job runs at a time per worker (this is a small hobby
cluster, not a scheduler) -- a dispatch while a job is already running is
rejected. /exec commands do NOT go through this same-job-at-a-time gate
(they're meant for quick fixes, not long training runs) but ARE logged and
rate-limited to one at a time via their own lock, so two /exec calls can't
stomp on each other.

Usage:
    python coordinator/worker_daemon.py [--port 8770] [--repo-root PATH]

Run this from inside the repo (or pass --repo-root) so relative entry_point
paths resolve correctly. On a real second laptop, this runs against that
machine's own checkout of this repo, at whatever commit was pulled.
"""
import argparse
import hmac
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
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
    "coordinator/test_noop_job_wsl2.py",  # WSL2-dispatch smoke test, see that file's docstring
}

# entry_points that must be launched inside WSL2 via torchrun rather than
# with sys.executable directly. sys.executable is native Windows Python,
# which has no NCCL backend at all (see 36b_..._wsl2.py's docstring) -- a
# job spec targeting this script CANNOT run as a plain subprocess launch;
# it has to cross into the WSL2 Linux environment where NCCL is actually
# available. Kept as an explicit set (not spec-driven) for the same reason
# ALLOWED_ENTRY_POINTS is a fixed set: which launch strategy applies to a
# script is a reviewed decision, not something a job spec should toggle.
WSL2_ENTRY_POINTS = {
    "scripts/36b_train_yolo_v4_ddp_wsl2.py",
    "coordinator/test_noop_job_wsl2.py",
}
WSL2_DISTRO = os.environ.get("MULTIGPU_WSL_DISTRO", "Ubuntu-24.04")
WSL2_VENV = os.environ.get("MULTIGPU_WSL_VENV", "~/yolo_ddp_env")
WSL2_NPROC_PER_NODE = 1  # one GPU per machine in this cluster


def _windows_path_to_wsl(win_path):
    """Translate an absolute Windows path (e.g. L:\\Likith\\...) to its
    WSL2 mount equivalent (/mnt/l/Likith/...) -- confirmed this is how
    this machine's WSL2 actually reaches the repo (see
    36b_train_yolo_v4_ddp_wsl2.py's DATA_ROOT docstring: "/mnt/l/..." is
    the real, tested mount path, not assumed from docs)."""
    resolved = Path(win_path).resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = "/".join(resolved.parts[1:])
    return f"/mnt/{drive}/{rest}"

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

# Shared-secret token required on /dispatch, /exec, /dataset-pull. Loaded
# from a file so the secret itself never has to be typed into a command
# line or committed to git; generate one with e.g. `python -c "import
# secrets; print(secrets.token_hex(32))" > coordinator/dispatch_token.secret`
# on each worker, out-of-band (not over this HTTP endpoint).
TOKEN_PATH = Path(__file__).resolve().parent / "dispatch_token.secret"

EXEC_LOG_PATH = Path(__file__).resolve().parent / "worker_exec.log"
EXEC_TIMEOUT_SEC = 300  # hard cap so a hung command can't block this worker forever
EXEC_LOCK = threading.Lock()  # one /exec at a time, so two remote fixes can't race


def log_exec(command, exit_code, duration_s, stdout_len, stderr_len):
    """Append-only local audit trail for every /exec command run on this
    worker. Since /exec is genuinely unrestricted remote code execution
    (see module docstring), this log is the only local record of what the
    main laptop has done to this machine -- check it if anything looks
    wrong here that you didn't expect."""
    line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} exit={exit_code} "
            f"duration={duration_s:.1f}s stdout_bytes={stdout_len} "
            f"stderr_bytes={stderr_len} cmd={command!r}\n")
    with open(EXEC_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


class _DirListingParser(HTMLParser):
    """Parses the <li><a href="...">...</a></li> directory-listing HTML
    that Python's http.server.SimpleHTTPRequestHandler generates (verified
    format by actually serving a directory and curling it, not assumed
    from docs). Collects every href attribute from an <a> tag."""

    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href":
                    self.hrefs.append(value)


def _safe_join(dest_dir, name, dest_root_real):
    """Resolve dest_dir/name and verify it stays inside dest_root_real.
    SECURITY: a security review correctly flagged path traversal /
    arbitrary file write here. `name` comes from a REMOTE HTTP server's
    directory-listing response (an href attribute) -- _handle_dataset_pull
    accepts any URL in its POST body, so the content of that listing is
    NOT trustworthy by construction. The prior code extracted a "filename"
    via `href.rstrip('/').rsplit('/', 1)[-1]` BEFORE url-decoding, which
    looks safe (a literal '../foo' collapses to 'foo') but percent-encoded
    traversal sequences survive that split untouched and only become real
    '../' after urllib.parse.unquote() runs afterward -- confirmed
    exploitable: an href of '..%2f..%2f..%2fevil.txt' decodes to
    '../../../evil.txt' AFTER the split, escaping dest_dir entirely, which
    would let a malicious or compromised dataset-server URL write files
    anywhere the worker process has permission to write. Raises ValueError
    if the resolved path would escape dest_root_real; caller must not
    proceed with the write on that exception."""
    candidate = Path(os.path.realpath(dest_dir / name))
    try:
        candidate.relative_to(dest_root_real)
    except ValueError:
        raise ValueError(f"entry {name!r} resolves outside the dataset destination root — refused")
    return candidate


def _recursive_http_fetch(base_url, dest_dir, _depth=0, _dest_root_real=None):
    """Recursively downloads every file under an http.server-style
    directory listing at base_url into dest_dir. Pure Python (urllib +
    html.parser), no external binary dependency -- see _handle_dataset_pull
    for why that matters (a shelled-out `wget` call silently failed on
    native Windows, where wget is only a PowerShell alias, not a real
    executable). Returns (files_written, bytes_written)."""
    if _depth > 10:
        raise RuntimeError("directory nesting exceeded 10 levels -- possible listing loop, aborting")
    if _dest_root_real is None:
        _dest_root_real = Path(os.path.realpath(dest_dir))

    with urllib.request.urlopen(base_url, timeout=30) as resp:
        html_text = resp.read().decode("utf-8", errors="replace")

    parser = _DirListingParser()
    parser.feed(html_text)

    files_written = 0
    bytes_written = 0
    for href in parser.hrefs:
        if href in ("../", "..", "/"):
            continue
        child_url = urllib.parse.urljoin(base_url, href)
        name = urllib.parse.unquote(href.rstrip("/").rsplit("/", 1)[-1])
        if not name:
            continue
        if href.endswith("/"):
            sub_dest = _safe_join(dest_dir, name, _dest_root_real)
            sub_dest.mkdir(parents=True, exist_ok=True)
            f, b = _recursive_http_fetch(child_url, sub_dest, _depth + 1, _dest_root_real)
            files_written += f
            bytes_written += b
        else:
            dest_file = _safe_join(dest_dir, name, _dest_root_real)
            with urllib.request.urlopen(child_url, timeout=60) as file_resp:
                data = file_resp.read()
            dest_file.write_bytes(data)
            files_written += 1
            bytes_written += len(data)

    return files_written, bytes_written


def health_check():
    """Best-effort signals for known failure classes hit this session
    (Tailscale account/connection issues, disk space, PyTorch/CUDA version
    drift) so the coordinator has real, specific signal to act on instead
    of just "unreachable". Every check is wrapped so one failing check
    doesn't prevent the others from reporting."""
    health = {}

    try:
        tailscale_exe = shutil.which("tailscale") or r"C:\Program Files\Tailscale\tailscale.exe"
        out = subprocess.run([tailscale_exe, "status", "--json"], capture_output=True,
                              text=True, timeout=5)
        if out.returncode == 0:
            info = json.loads(out.stdout)
            health["tailscale_connected"] = True
            health["tailscale_tailnet"] = (info.get("CurrentTailnet") or {}).get("Name")
            health["tailscale_peer_count"] = len(info.get("Peer") or {})
        else:
            health["tailscale_connected"] = False
    except Exception as exc:
        health["tailscale_connected"] = False
        health["tailscale_error"] = str(exc)

    try:
        usage = shutil.disk_usage(str(REPO_ROOT))
        health["disk_free_gb"] = round(usage.free / 1e9, 1)
        health["disk_total_gb"] = round(usage.total / 1e9, 1)
    except Exception as exc:
        health["disk_error"] = str(exc)

    try:
        import torch
        health["torch_version"] = torch.__version__
        health["cuda_available"] = torch.cuda.is_available()
        health["nccl_available"] = torch.distributed.is_nccl_available()
    except Exception as exc:
        health["torch_error"] = str(exc)

    return health


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


_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9.-]{1,255}$")


def _validate_wsl_launch_value(name, value, kind):
    """SECURITY: build_launch_argv interpolates these values directly into
    a `bash -lc "..."` string run inside WSL2 -- a security review
    correctly flagged this as command injection, since these values come
    from the /dispatch HTTP payload's `env` dict, which is only gated by
    a shared bearer token, not otherwise trusted input. quoting alone
    (shlex.quote) would stop shell metacharacters, but these fields are
    also structurally constrained (a hostname, a port number, a batch
    size, a filesystem path) -- validating the actual shape is a
    stronger and clearer guarantee than escaping, and matches this
    project's existing allowlist-over-blocklist approach (see
    ALLOWED_ENTRY_POINTS, ALLOWED_ENV_KEYS). Raises ValueError with a
    clear reason on failure; caller must not proceed to build inner_cmd
    on that exception."""
    if kind == "host":
        try:
            ipaddress.ip_address(value)
            return value
        except ValueError:
            pass
        if _HOSTNAME_RE.match(value):
            return value
        raise ValueError(f"{name} {value!r} is not a valid IP address or hostname")
    if kind == "port":
        if not str(value).isdigit() or not (1 <= int(value) <= 65535):
            raise ValueError(f"{name} {value!r} is not a valid port number")
        return str(int(value))
    if kind == "int":
        if not str(value).isdigit():
            raise ValueError(f"{name} {value!r} is not a valid integer")
        return str(int(value))
    if kind == "path":
        # DATA_ROOT becomes a `DATA_ROOT=<value>` shell-env assignment
        # ahead of torchrun -- reject anything with shell metacharacters,
        # whitespace, or a leading '-' (which could be misread as a flag
        # by something downstream). This does not need to be a real path
        # on this machine (WSL2 paths differ from the Windows daemon's
        # own filesystem), just free of shell-breaking characters.
        if not re.match(r"^[A-Za-z0-9._/\-]{1,4096}$", str(value)) or str(value).startswith("-"):
            raise ValueError(f"{name} {value!r} contains disallowed characters for a path")
        return str(value)
    raise AssertionError(f"unknown validation kind {kind!r}")


def build_launch_argv(entry_point_str, script_path, env, rank, world_size, local_rank):
    """Returns the subprocess argv to actually launch this job with.

    Two cases:
      - entry_point in WSL2_ENTRY_POINTS: this script needs NCCL, which
        only exists in this machine's WSL2 PyTorch build (native Windows
        PyTorch has no NCCL backend at all -- root-caused earlier this
        session, see 36b_..._wsl2.py's docstring). Launch via
        `wsl.exe -d <distro> -- bash -lc "source venv && torchrun ..."`,
        NOT sys.executable, which would just run the script under native
        Windows Python and hit the same missing-NCCL wall again.
        Uses torchrun (not a bare python invocation) because that's what
        actually works for rendezvous inside WSL2's PyTorch build --
        confirmed in this session, see the script's own LAUNCH section.
        torchrun sets RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT
        itself from its own flags, so this passes them as torchrun flags
        rather than env vars (the script also reads them from env as a
        fallback, so this is consistent either way).
      - anything else: unchanged, sys.executable direct launch (this is
        the path coordinator/test_noop_job.py uses, and is intentionally
        simple since that job has no GPU/NCCL requirement at all).
    """
    if entry_point_str not in WSL2_ENTRY_POINTS:
        return [sys.executable, str(script_path)]

    wsl_repo_root = _windows_path_to_wsl(REPO_ROOT)
    wsl_script_rel = entry_point_str  # already relative, forward-slashed; entry_point_str
                                       # itself is validated against WSL2_ENTRY_POINTS by
                                       # the caller before this function is ever reached

    # SECURITY: every value below is interpolated into a `bash -lc "..."`
    # string that actually runs inside WSL2 -- see _validate_wsl_launch_value's
    # docstring. All must pass validation before inner_cmd is built; any
    # ValueError here propagates to the /dispatch handler, which rejects
    # the request with HTTP 403 rather than launching anything.
    master_addr = _validate_wsl_launch_value("MASTER_ADDR", env.get("MASTER_ADDR", "127.0.0.1"), "host")
    master_port = _validate_wsl_launch_value("MASTER_PORT", env.get("MASTER_PORT", "29500"), "port")
    rank = _validate_wsl_launch_value("rank", rank, "int")
    world_size = _validate_wsl_launch_value("world_size", world_size, "int")

    env_prefix = ""
    per_gpu_batch = env.get("PER_GPU_BATCH")
    if per_gpu_batch:
        env_prefix += f"PER_GPU_BATCH={_validate_wsl_launch_value('PER_GPU_BATCH', per_gpu_batch, 'int')} "
    data_root = env.get("DATA_ROOT")
    if data_root:
        env_prefix += f"DATA_ROOT={_validate_wsl_launch_value('DATA_ROOT', data_root, 'path')} "

    inner_cmd = (
        f"source {WSL2_VENV}/bin/activate && "
        f"cd {wsl_repo_root} && "
        f"{env_prefix}torchrun --nnodes={world_size} --node_rank={rank} "
        f"--nproc_per_node={WSL2_NPROC_PER_NODE} "
        f"--master_addr={master_addr} --master_port={master_port} "
        f"{wsl_script_rel}"
    )
    wsl_exe = shutil.which("wsl") or r"C:\Windows\System32\wsl.exe"
    return [wsl_exe, "-d", WSL2_DISTRO, "--", "bash", "-lc", inner_cmd]


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
        self.log_file = None

    def snapshot(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is not None and self.exit_code is None:
                self.exit_code = self.proc.returncode
                self.finished_at = time.time()
                if self.log_file is not None:
                    self.log_file.close()
                    self.log_file = None

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
            # SECURITY/DEBUGGABILITY: previously stdout/stderr went to
            # DEVNULL -- a real cross-machine training job failed after
            # 112 minutes with no way to see WHY (no traceback, no NCCL
            # error text, nothing) since the actual Python/torchrun output
            # was discarded. Redirect to a per-job log file instead so a
            # future failure is diagnosable via `/exec cat
            # coordinator/job_output.log` rather than pure guesswork.
            # Overwritten each dispatch (not appended) -- only the most
            # recent job's output matters for debugging the most recent
            # failure; unbounded append growth on a long-running worker
            # serves no purpose here.
            log_path = Path(__file__).resolve().parent / "job_output.log"
            try:
                self.log_file = open(log_path, "w", encoding="utf-8", errors="replace")
                self.proc = subprocess.Popen(
                    argv, cwd=str(REPO_ROOT), env=env,
                    stdout=self.log_file, stderr=subprocess.STDOUT,
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
            "health": health_check(),
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

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def _handle_exec(self):
        """POST /exec {"command": "..."} -- runs an arbitrary shell command
        on this worker and returns its output. See the module docstring's
        SECURITY NOTE: this is deliberate, unrestricted remote code
        execution, built at the user's explicit request with the risk
        acknowledged. Every call is logged to worker_exec.log regardless of
        outcome, BEFORE the response is sent, so there's a local record
        even if the response never reaches the coordinator."""
        try:
            payload = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return
        command = payload.get("command")
        if not command or not isinstance(command, str):
            self._send_json(400, {"error": "command (string) is required"})
            return

        if not EXEC_LOCK.acquire(blocking=False):
            self._send_json(409, {"error": "another /exec command is already running on this worker"})
            return
        try:
            start = time.time()
            try:
                proc = subprocess.run(
                    command, shell=True, cwd=str(REPO_ROOT),
                    capture_output=True, text=True, timeout=EXEC_TIMEOUT_SEC,
                )
                duration = time.time() - start
                log_exec(command, proc.returncode, duration, len(proc.stdout), len(proc.stderr))
                self._send_json(200, {
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout[-20000:],  # cap response size, full output stays in the log's byte counts
                    "stderr": proc.stderr[-20000:],
                    "duration_sec": round(duration, 1),
                })
            except subprocess.TimeoutExpired:
                duration = time.time() - start
                log_exec(command, "TIMEOUT", duration, 0, 0)
                self._send_json(408, {"error": f"command exceeded {EXEC_TIMEOUT_SEC}s timeout and was killed"})
        finally:
            EXEC_LOCK.release()

    def _handle_dataset_pull(self):
        """POST /dataset-pull {"url": "http://<main-ip>:<port>/",
        "dest": "data/yolo_detect_v4"} -- recursively fetches every file
        under a directory served by Python's http.server (see
        coordinator.py's cmd_send_dataset) into this worker's own dest
        path. dest is relative to REPO_ROOT, validated to stay inside it
        (same containment logic as entry_point validation).

        Implemented as a pure-Python recursive fetch (urllib + a tiny HTML
        directory-listing parser), NOT a shelled-out `wget` -- an earlier
        version called subprocess.run(["wget", ...]) directly, which
        worked when this daemon runs inside WSL2 (real wget binary) but
        silently crashed with an uncaught FileNotFoundError when running
        as native Windows Python (where "wget" is only a PowerShell
        alias for Invoke-WebRequest, not a real executable subprocess.run
        can find) -- the client just saw the connection drop with no
        useful error. Found by testing this for real, not just review.
        A pure-Python implementation works identically on both."""
        try:
            payload = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return
        url = payload.get("url")
        dest = payload.get("dest")
        if not url or not dest:
            self._send_json(400, {"error": "url and dest are required"})
            return

        dest_path = Path(os.path.realpath(REPO_ROOT / dest))
        repo_root_real = Path(os.path.realpath(REPO_ROOT))
        try:
            dest_path.relative_to(repo_root_real)
        except ValueError:
            self._send_json(403, {"error": f"dest {dest!r} resolves outside the repo root — refused"})
            return

        dest_path.mkdir(parents=True, exist_ok=True)
        start = time.time()
        try:
            files_written, bytes_written = _recursive_http_fetch(url, dest_path)
            duration = time.time() - start
            log_exec(f"dataset-pull {url} -> {dest_path}", 0, duration, bytes_written, 0)
            self._send_json(200, {
                "dest": str(dest_path),
                "files_written": files_written,
                "bytes_written": bytes_written,
                "duration_sec": round(duration, 1),
            })
        except Exception as exc:
            duration = time.time() - start
            log_exec(f"dataset-pull {url} -> {dest_path}", "ERROR", duration, 0, 0)
            self._send_json(500, {"error": f"dataset pull failed: {exc}"})

    def do_POST(self):
        if self.path not in ("/dispatch", "/exec", "/dataset-pull"):
            self._send_json(404, {"error": "not found"})
            return
        if not self._check_auth():
            self._send_json(401, {"error": "missing or invalid Authorization bearer token"})
            return

        if self.path == "/exec":
            self._handle_exec()
            return
        if self.path == "/dataset-pull":
            self._handle_dataset_pull()
            return

        try:
            payload = self._read_json_body()
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

        try:
            argv = build_launch_argv(entry_point, script_path, env, rank, world_size,
                                      payload.get("local_rank", 0))
        except ValueError as exc:
            # SECURITY: a dispatch value failed WSL2 launch-string validation
            # (see _validate_wsl_launch_value) -- refuse rather than build a
            # shell command from an unvalidated value.
            self._send_json(403, {"error": str(exc)})
            return
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
