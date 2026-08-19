"""
Coordinator control API -- turns coordinator.py's existing CLI actions
(status/dispatch/exec/health/send-dataset) into a small local HTTP API,
so dashboard/index.html can be a real web app with buttons instead of a
read-only view of a file someone has to remember to regenerate with
`coordinator.py watch`.

This is a thin wrapper: every endpoint calls the SAME functions
coordinator.py's CLI commands already call (poll_worker, dispatch_job,
exec_on_worker, request_dataset_pull, build_dashboard_status) -- no new
worker-facing behavior, no new security surface on the WORKER side.
worker_daemon.py's own auth/allowlist/validation are unchanged and still
apply to every request this API makes to a worker.

WHAT IS NEW: an HTTP-reachable control point on the COORDINATOR side.
Anyone who can reach this API AND has its bearer token can do everything
coordinator.py's CLI can do, including /exec on any worker (genuine
unrestricted remote code execution -- see worker_daemon.py's docstring).
This is a different token from any WORKER's dispatch_token.secret --
compromising this one is equivalent to compromising the coordinator
machine itself. Binds to 127.0.0.1 by default for exactly this reason;
only bind a real interface deliberately, same rule as worker_daemon.py.

Usage:
    python -c "import secrets; print(secrets.token_hex(32))" > coordinator/control_api_token.secret
    python coordinator/control_api.py --port 8790 --token-file coordinator/control_api_token.secret

Endpoints (all require `Authorization: Bearer <token>` except health-of-this-process):
    GET  /api/status                         -- poll all workers, return dashboard-shaped JSON
    POST /api/dispatch   {job_spec: "name"}   -- dispatch a job-specs/<name>.yaml to all workers
    POST /api/exec       {worker, command}    -- run a shell command on one worker ("*" = all)
    POST /api/health                          -- poll health signals, return problem list
    POST /api/send-dataset {worker, dataset_path, advertise_host}
                                               -- push a dataset directory to a worker
"""
import argparse
import hmac
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coordinator as coord  # noqa: E402  (path insert must come first)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 8790  # distinct from worker_daemon.py's 8770 and send-dataset's 29505
DEFAULT_WORKERS_PATH = REPO_ROOT / "coordinator" / "workers.yaml"
DEFAULT_TOKEN_PATH = Path(__file__).resolve().parent / "control_api_token.secret"


def _job_spec_path(name):
    """Resolve a job spec NAME (not an arbitrary path) to job-specs/<name>.yaml,
    refusing anything that isn't a bare filename -- same containment
    reasoning as worker_daemon.py's entry_point validation: the request
    body supplies data (a name), not a path the caller gets to control the
    shape of. Raises ValueError on anything suspicious."""
    if not name or not isinstance(name, str) or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"job_spec {name!r} must be a bare filename with no path separators")
    candidate = (REPO_ROOT / "job-specs" / f"{name}.yaml").resolve()
    job_specs_root = (REPO_ROOT / "job-specs").resolve()
    if job_specs_root not in candidate.parents:
        raise ValueError(f"job_spec {name!r} resolves outside job-specs/ -- refused")
    if not candidate.is_file():
        raise ValueError(f"job-specs/{name}.yaml does not exist")
    return candidate


class Handler(BaseHTTPRequestHandler):
    server_version = "ControlAPI/1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[control_api] " + (fmt % args) + "\n")

    def _send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")  # dashboard served from a different port/origin
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self):
        expected = self.server.api_token
        got = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not got.startswith(prefix):
            return False
        return hmac.compare_digest(got[len(prefix):], expected)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_OPTIONS(self):
        # CORS preflight -- dashboard fetch() calls arrive from a different
        # origin (dashboard served on one port, this API on another).
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/status":
            self._handle_status()
            return
        self._send_json(404, {"error": "not found"})

    def _handle_status(self):
        workers = coord.load_workers(str(self.server.workers_path))
        statuses = [coord.poll_worker(w) for w in workers]
        dashboard_status = coord.build_dashboard_status(workers, statuses)
        Path(self.server.status_out).write_text(json.dumps(dashboard_status, indent=2))
        self._send_json(200, dashboard_status)

    def do_POST(self):
        if self.path not in ("/api/dispatch", "/api/exec", "/api/health", "/api/send-dataset"):
            self._send_json(404, {"error": "not found"})
            return
        if not self._check_auth():
            self._send_json(401, {"error": "missing or invalid Authorization bearer token"})
            return
        try:
            payload = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        handler = {
            "/api/dispatch": self._handle_dispatch,
            "/api/exec": self._handle_exec,
            "/api/health": self._handle_health,
            "/api/send-dataset": self._handle_send_dataset,
        }[self.path]
        handler(payload)

    def _handle_dispatch(self, payload):
        job_spec_name = payload.get("job_spec")
        try:
            spec_path = _job_spec_path(job_spec_name)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        workers = coord.load_workers(str(self.server.workers_path))
        job_spec_yaml_text = spec_path.read_text()
        import yaml
        spec = yaml.safe_load(job_spec_yaml_text)

        world_size = len(workers)
        master = workers[0]
        master_addr = master["host"]
        master_port = spec.get("env", {}).get("MASTER_PORT", "29500")

        results = []
        for rank, worker in enumerate(workers):
            status_code, resp = coord.dispatch_job(
                worker, job_spec_yaml_text, spec, rank, world_size, master_addr, master_port,
            )
            results.append({"worker": worker["name"], "status_code": status_code, "response": resp})
        ok = all(r["status_code"] == 202 for r in results)
        self._send_json(200 if ok else 502, {"ok": ok, "results": results})

    def _handle_exec(self, payload):
        worker_name = payload.get("worker")
        command = payload.get("command")
        if not command or not isinstance(command, str):
            self._send_json(400, {"error": "command (string) is required"})
            return

        workers = coord.load_workers(str(self.server.workers_path))
        targets = workers if worker_name == "*" else [w for w in workers if w["name"] == worker_name]
        if not targets:
            self._send_json(404, {"error": f"no worker named {worker_name!r}"})
            return

        results = []
        for worker in targets:
            status_code, resp = coord.exec_on_worker(worker, command)
            results.append({"worker": worker["name"], "status_code": status_code, "response": resp})
        ok = all(r["status_code"] == 200 for r in results)
        self._send_json(200 if ok else 502, {"ok": ok, "results": results})

    def _handle_health(self, payload):
        workers = coord.load_workers(str(self.server.workers_path))
        problems = []
        details = []
        for worker in workers:
            status = coord.poll_worker(worker)
            name = worker["name"]
            entry = {"worker": name}
            if status.get("unreachable"):
                problems.append(f"{name}: UNREACHABLE ({status.get('error')})")
                entry["unreachable"] = True
            else:
                health = status.get("health", {})
                entry.update({
                    "torch_version": health.get("torch_version"),
                    "disk_free_gb": health.get("disk_free_gb"),
                    "tailscale_tailnet": health.get("tailscale_tailnet"),
                    "tailscale_peer_count": health.get("tailscale_peer_count"),
                    "job_status": status.get("job", {}).get("status"),
                })
                if not health.get("tailscale_connected"):
                    problems.append(f"{name}: Tailscale not connected")
                disk_free = health.get("disk_free_gb")
                if disk_free is not None and disk_free < 5:
                    problems.append(f"{name}: LOW DISK ({disk_free}GB free)")
            details.append(entry)
        self._send_json(200, {"problems": problems, "workers": details})

    def _handle_send_dataset(self, payload):
        worker_name = payload.get("worker")
        dataset_path = payload.get("dataset_path")
        advertise_host = payload.get("advertise_host")
        if not dataset_path or not advertise_host:
            self._send_json(400, {"error": "dataset_path and advertise_host are required"})
            return

        workers = coord.load_workers(str(self.server.workers_path))
        targets = workers if worker_name == "*" else [w for w in workers if w["name"] == worker_name]
        if not targets:
            self._send_json(404, {"error": f"no worker named {worker_name!r}"})
            return

        serve_dir = (REPO_ROOT / dataset_path).resolve()
        repo_root_real = REPO_ROOT.resolve()
        if repo_root_real not in serve_dir.parents and serve_dir != repo_root_real:
            self._send_json(403, {"error": f"dataset_path {dataset_path!r} resolves outside the repo root"})
            return
        if not serve_dir.is_dir():
            self._send_json(400, {"error": f"{serve_dir} is not a directory"})
            return

        import http.server as _http_server
        import socketserver as _socketserver

        bind_port = int(payload.get("bind_port", 29505))

        class _FileHandler(_http_server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=str(serve_dir), **kw)
            def log_message(self, *a):
                pass

        httpd = _socketserver.TCPServer(("0.0.0.0", bind_port), _FileHandler)
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        url = f"http://{advertise_host}:{bind_port}/"
        results = []
        try:
            for worker in targets:
                status_code, resp = coord.request_dataset_pull(worker, url, dataset_path)
                results.append({"worker": worker["name"], "status_code": status_code, "response": resp})
        finally:
            httpd.shutdown()
        ok = all(r["status_code"] == 200 for r in results)
        self._send_json(200 if ok else 502, {"ok": ok, "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1",
                         help="bind address; defaults to loopback-only, same reasoning as "
                              "worker_daemon.py -- only bind a real interface deliberately.")
    parser.add_argument("--token-file", default=str(DEFAULT_TOKEN_PATH))
    parser.add_argument("--workers", default=str(DEFAULT_WORKERS_PATH))
    parser.add_argument("--status-out", default=str(coord.DEFAULT_STATUS_OUT))
    args = parser.parse_args()

    token_path = Path(args.token_file)
    if not token_path.is_file():
        print(f"[control_api] FATAL: token file {token_path} not found. Generate one:\n"
              f'  python -c "import secrets; print(secrets.token_hex(32))" > {token_path}',
              file=sys.stderr)
        sys.exit(1)
    api_token = token_path.read_text(encoding="utf-8").strip()
    if not api_token:
        print(f"[control_api] FATAL: token file {token_path} is empty.", file=sys.stderr)
        sys.exit(1)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.api_token = api_token
    server.workers_path = Path(args.workers)
    server.status_out = Path(args.status_out)
    print(f"[control_api] listening on {args.host}:{args.port}, workers={args.workers}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
