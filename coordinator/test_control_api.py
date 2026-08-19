"""
End-to-end test for control_api.py -- the HTTP wrapper around
coordinator.py's CLI actions that dashboard/index.html's "web app" actions
panel talks to.

Starts a real worker_daemon.py AND a real control_api.py as subprocesses
on loopback, then drives control_api.py exactly like the dashboard's own
JS does (plain fetch() calls, same endpoints/payload shapes) and asserts:

  GET  /api/status          -> real machine/job data, matches
                                coordinator.py's own build_dashboard_status
  POST /api/dispatch        -> dispatches job-specs/test-noop.yaml, worker
                                actually runs it, reports "done"
  POST /api/exec            -> runs a real command, returns real output
  POST /api/health          -> reports real health signals
  auth: /api/dispatch, /api/exec, /api/health all reject a missing/bad
        bearer token with 401; /api/status is unauthenticated (read-only,
        matches worker_daemon.py's own /status)
  /api/dispatch job_spec path-traversal / nonexistent-name rejection

Run with: python coordinator/test_control_api.py
Exits 0 and prints "CONTROL API TEST PASSED" on success, non-zero otherwise.
"""
import json
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COORDINATOR_DIR = REPO_ROOT / "coordinator"
WORKER_PORT = 8783  # distinct from test_e2e.py's 8781 and the real daemon's 8770
API_PORT = 8791     # distinct from control_api.py's own default 8790


def fail(msg):
    print(f"CONTROL API TEST FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def http_json(method, url, token=None, body=None, timeout=15):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except json.JSONDecodeError:
            return exc.code, {"error": "non-JSON error response"}


def main():
    token_dir = Path(tempfile.mkdtemp(prefix="mgpu-test-control-api-"))
    worker_token_path = token_dir / "worker.secret"
    worker_token_path.write_text(secrets.token_hex(32), encoding="utf-8")
    api_token_path = token_dir / "api.secret"
    api_token = secrets.token_hex(32)
    api_token_path.write_text(api_token, encoding="utf-8")

    # A throwaway workers.yaml pointing at the loopback worker this test starts.
    workers_yaml_path = token_dir / "workers.yaml"
    workers_yaml_path.write_text(
        "workers:\n"
        "  - name: test-worker\n"
        "    host: 127.0.0.1\n"
        f"    port: {WORKER_PORT}\n"
        f"    token_file: {worker_token_path}\n",
        encoding="utf-8",
    )

    print(f"starting worker_daemon.py on 127.0.0.1:{WORKER_PORT} ...")
    worker_proc = subprocess.Popen(
        [sys.executable, str(COORDINATOR_DIR / "worker_daemon.py"),
         "--port", str(WORKER_PORT), "--host", "127.0.0.1", "--token-file", str(worker_token_path)],
        cwd=str(REPO_ROOT),
    )
    print(f"starting control_api.py on 127.0.0.1:{API_PORT} ...")
    api_proc = subprocess.Popen(
        [sys.executable, str(COORDINATOR_DIR / "control_api.py"),
         "--port", str(API_PORT), "--host", "127.0.0.1", "--token-file", str(api_token_path),
         "--workers", str(workers_yaml_path),
         "--status-out", str(token_dir / "status.json")],
        cwd=str(REPO_ROOT),
    )
    base = f"http://127.0.0.1:{API_PORT}"

    try:
        for _ in range(50):
            try:
                code, _ = http_json("GET", f"{base}/api/status", timeout=2)
                if code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            fail("control_api.py never became reachable on /api/status")
        print("OK: control API reachable")

        # /api/status: unauthenticated GET works (read-only, matches worker_daemon.py's own /status).
        code, data = http_json("GET", f"{base}/api/status")
        if code != 200:
            fail(f"/api/status did not return 200, got {code}: {data}")
        if data["cluster"]["machines_total"] != 1 or data["machines"][0]["name"] != "test-worker":
            fail(f"/api/status did not report the expected worker: {data}")
        print("OK: /api/status returns real dashboard-shaped data")

        # /api/dispatch without auth -> 401.
        code, data = http_json("POST", f"{base}/api/dispatch", token=None, body={"job_spec": "test-noop"})
        if code != 401:
            fail(f"expected 401 for unauthenticated /api/dispatch, got {code}: {data}")
        print("OK: unauthenticated /api/dispatch correctly rejected with 401")

        # /api/dispatch with a bad token -> 401.
        code, data = http_json("POST", f"{base}/api/dispatch", token="wrong-token", body={"job_spec": "test-noop"})
        if code != 401:
            fail(f"expected 401 for a bad token on /api/dispatch, got {code}: {data}")
        print("OK: bad-token /api/dispatch correctly rejected with 401")

        # /api/dispatch: path-traversal job_spec name rejected.
        code, data = http_json("POST", f"{base}/api/dispatch", token=api_token,
                                body={"job_spec": "../../../etc/passwd"})
        if code != 400:
            fail(f"expected 400 for a path-escaping job_spec name, got {code}: {data}")
        print("OK: path-escaping job_spec name correctly rejected with 400")

        # /api/dispatch: nonexistent job_spec name rejected.
        code, data = http_json("POST", f"{base}/api/dispatch", token=api_token,
                                body={"job_spec": "does-not-exist-anywhere"})
        if code != 400:
            fail(f"expected 400 for a nonexistent job_spec, got {code}: {data}")
        print("OK: nonexistent job_spec name correctly rejected with 400")

        # /api/dispatch: real dispatch of the safe no-op job spec.
        code, data = http_json("POST", f"{base}/api/dispatch", token=api_token, body={"job_spec": "test-noop"})
        if code != 200 or not data.get("ok"):
            fail(f"expected a successful dispatch, got {code}: {data}")
        if data["results"][0]["status_code"] != 202:
            fail(f"expected worker to accept dispatch with 202, got: {data}")
        print(f"OK: /api/dispatch accepted: {data['results'][0]['response']}")

        # Poll /api/status until the job reports done.
        for _ in range(50):
            _, data = http_json("GET", f"{base}/api/status")
            job = data["jobs"][0] if data["jobs"] else {}
            if job.get("status") == "done":
                break
            time.sleep(0.2)
        else:
            fail(f"dispatched job never reached 'done' via /api/status, last seen: {data}")
        print("OK: dispatched job reached 'done' status via /api/status polling")

        # /api/exec: unauthenticated -> 401.
        code, data = http_json("POST", f"{base}/api/exec", token=None,
                                body={"worker": "test-worker", "command": "echo should-not-run"})
        if code != 401:
            fail(f"expected 401 for unauthenticated /api/exec, got {code}: {data}")
        print("OK: unauthenticated /api/exec correctly rejected with 401")

        # /api/exec: real command execution.
        code, data = http_json("POST", f"{base}/api/exec", token=api_token,
                                body={"worker": "test-worker", "command": "echo control-api-exec-marker"})
        if code != 200 or not data.get("ok"):
            fail(f"expected a successful exec, got {code}: {data}")
        stdout = data["results"][0]["response"].get("stdout", "")
        if "control-api-exec-marker" not in stdout:
            fail(f"exec output missing expected marker: {data}")
        print("OK: /api/exec ran a real command, stdout contained the expected marker")

        # /api/exec: unknown worker name -> 404.
        code, data = http_json("POST", f"{base}/api/exec", token=api_token,
                                body={"worker": "no-such-worker", "command": "echo hi"})
        if code != 404:
            fail(f"expected 404 for an unknown worker name, got {code}: {data}")
        print("OK: unknown worker name on /api/exec correctly rejected with 404")

        # /api/health: unauthenticated -> 401; authenticated -> real health data.
        code, data = http_json("POST", f"{base}/api/health", token=None, body={})
        if code != 401:
            fail(f"expected 401 for unauthenticated /api/health, got {code}: {data}")
        code, data = http_json("POST", f"{base}/api/health", token=api_token, body={})
        if code != 200 or "workers" not in data:
            fail(f"expected real health data, got {code}: {data}")
        if data["workers"][0]["worker"] != "test-worker":
            fail(f"health data missing expected worker: {data}")
        print("OK: /api/health returns real per-worker health data")

        print("\nCONTROL API TEST PASSED")
    finally:
        for proc in (api_proc, worker_proc):
            proc.terminate()
        for proc in (api_proc, worker_proc):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
