"""
Local loopback end-to-end test for Layer 2, since no second laptop is
available: starts worker_daemon.py as a subprocess on 127.0.0.1, dispatches
the trivial coordinator/test_noop_job.yaml (NOT the real multi-hour YOLO DDP
training job -- that would collide with training already running on this
GPU), and asserts the full loop works:

  coordinator polls worker /status (idle)
  -> coordinator dispatches job via /dispatch
  -> worker validates entry_point against the repo, launches subprocess
  -> coordinator polls /status again and sees "running", then "done"
  -> coordinator writes status.json in the dashboard-shaped format

Run with: python coordinator/test_e2e.py
Exits 0 and prints "E2E TEST PASSED" on success, non-zero otherwise.
"""
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COORDINATOR_DIR = REPO_ROOT / "coordinator"
PORT = 8781  # distinct from the daemon's own default, to avoid clashing with a real daemon someone left running
STATUS_OUT = COORDINATOR_DIR / "test_status.json"


def fail(msg):
    print(f"E2E TEST FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def _which(exe):
    return shutil.which(exe)


def wd_module():
    """worker_daemon module, already imported into sys.modules by the time
    the WSL2-dispatch check runs (main() imports coordinator, which is
    separate, so this does its own small import here)."""
    import importlib
    return importlib.import_module("worker_daemon")


def main():
    sys.path.insert(0, str(COORDINATOR_DIR))
    import coordinator as coord  # coordinator/coordinator.py

    # Worker daemon now requires a token file (added after a security review
    # flagged /dispatch as unauthenticated) -- generate a throwaway one for
    # this test run so the test still exercises the real auth path rather
    # than bypassing it.
    token_dir = Path(tempfile.mkdtemp(prefix="mgpu-test-token-"))
    token_path = token_dir / "test.secret"
    token_path.write_text(secrets.token_hex(32), encoding="utf-8")

    worker = {"name": "loopback-test-worker", "host": "127.0.0.1", "port": PORT,
              "_token": token_path.read_text(encoding="utf-8").strip()}

    print(f"starting worker_daemon.py on 127.0.0.1:{PORT} ...")
    daemon = subprocess.Popen(
        [sys.executable, str(COORDINATOR_DIR / "worker_daemon.py"),
         "--port", str(PORT), "--host", "127.0.0.1", "--token-file", str(token_path)],
        cwd=str(REPO_ROOT),
    )
    try:
        # Wait for the daemon to come up.
        for _ in range(50):
            status = coord.poll_worker(worker)
            if not status.get("unreachable"):
                break
            time.sleep(0.2)
        else:
            fail("worker daemon never became reachable on /status")

        if status.get("job", {}).get("status") != "idle":
            fail(f"expected idle status before dispatch, got: {status}")
        print(f"OK: worker reachable, initial job status = idle (hostname={status.get('hostname')})")

        # Dispatch the trivial test job.
        spec_path = COORDINATOR_DIR / "test_noop_job.yaml"
        job_spec_yaml_text = spec_path.read_text()
        import yaml
        spec = yaml.safe_load(job_spec_yaml_text)
        status_code, resp = coord.dispatch_job(
            worker, job_spec_yaml_text, spec,
            rank=0, world_size=1, master_addr="127.0.0.1", master_port="29601",
        )
        if status_code != 202:
            fail(f"dispatch did not return 202, got {status_code}: {resp}")
        print(f"OK: dispatch accepted, pid={resp.get('pid')}")

        # Poll until the job finishes (test_noop_job.py sleeps ~2s).
        seen_running = False
        for _ in range(50):
            status = coord.poll_worker(worker)
            job_status = status.get("job", {}).get("status")
            if job_status == "running":
                seen_running = True
            if job_status == "done":
                break
            time.sleep(0.2)
        else:
            fail(f"job never reached 'done' status, last seen: {status}")

        if not seen_running:
            print("NOTE: never observed 'running' status directly (job may have finished between polls) -- not fatal")
        job = status["job"]
        if job.get("exit_code") != 0:
            fail(f"job did not exit 0: {job}")
        print(f"OK: job completed, exit_code=0, rank={job.get('rank')}, world_size={job.get('world_size')}")

        # Re-dispatch should be rejected while... actually job is done now,
        # so instead verify the dashboard-shaped status.json write works.
        statuses = [coord.poll_worker(worker)]
        dashboard_status = coord.build_dashboard_status([worker], statuses)
        STATUS_OUT.write_text(__import__("json").dumps(dashboard_status, indent=2))
        assert dashboard_status["cluster"]["machines_online"] == 1
        assert dashboard_status["jobs"][0]["name"] == "test-noop"
        assert dashboard_status["jobs"][0]["status"] == "done"
        print(f"OK: dashboard-shaped status.json written to {STATUS_OUT}")

        # WSL2 dispatch path: proves build_launch_argv's wsl.exe -> bash ->
        # torchrun branch (worker_daemon.py) actually launches and NCCL
        # process-group init/destroy succeeds, single-node (world_size=1),
        # dispatched through the real HTTP /dispatch endpoint -- not just
        # a unit test of the argv string. Run AFTER the dashboard-status
        # assertion above (not before) since this worker only tracks one
        # job at a time and would otherwise overwrite the "test-noop"
        # job name that assertion checks for. Skips (not fails) if this
        # machine has no WSL2/Ubuntu-24.04/yolo_ddp_env, since a fresh
        # clone or CI runner won't have that set up.
        wsl_available = subprocess.run(
            ["wsl.exe", "-d", wd_module().WSL2_DISTRO, "--", "bash", "-lc",
             f"test -d {wd_module().WSL2_VENV}"],
            capture_output=True, timeout=15,
        ).returncode == 0 if _which("wsl.exe") or _which("wsl") else False

        if wsl_available:
            wsl_spec_yaml = "name: wsl2-smoke\nentry_point: coordinator/test_noop_job_wsl2.py\n"
            status_code, resp = coord.dispatch_job(
                worker, wsl_spec_yaml, yaml.safe_load(wsl_spec_yaml),
                rank=0, world_size=1, master_addr="127.0.0.1", master_port="29599",
            )
            if status_code != 202:
                fail(f"WSL2 dispatch did not return 202, got {status_code}: {resp}")
            for _ in range(90):
                status = coord.poll_worker(worker)
                job_status = status.get("job", {}).get("status")
                if job_status == "done":
                    break
                time.sleep(1)
            else:
                fail(f"WSL2 job never reached 'done' status, last seen: {status}")
            if status["job"].get("exit_code") != 0:
                fail(f"WSL2 job did not exit 0: {status['job']}")
            print("OK: WSL2-dispatched job (real torchrun + NCCL init/destroy) completed cleanly")
        else:
            print("SKIP: WSL2 dispatch check (no WSL2/Ubuntu-24.04/yolo_ddp_env on this machine)")

        # Regression test for a real command-injection vulnerability found
        # by a security review: build_launch_argv (worker_daemon.py)
        # interpolated MASTER_ADDR/MASTER_PORT/PER_GPU_BATCH/DATA_ROOT
        # directly into a `bash -lc "..."` string run inside WSL2, with no
        # validation -- a malicious MASTER_ADDR like
        # "127.0.0.1; touch /tmp/pwned" would have injected an arbitrary
        # second shell command. Fixed with _validate_wsl_launch_value
        # (host/port/int/path format checks per field). This test dispatches
        # exactly such a payload against the entry_point in
        # WSL2_ENTRY_POINTS and asserts it's rejected with HTTP 403 BEFORE
        # wsl.exe is ever invoked -- runs unconditionally (no WSL2 install
        # required), since validation happens before the WSL2 launch step.
        injection_spec_yaml = "name: inject-attempt\nentry_point: coordinator/test_noop_job_wsl2.py\n"
        code, resp4 = coord.dispatch_job(
            worker, injection_spec_yaml, yaml.safe_load(injection_spec_yaml),
            rank=0, world_size=1, master_addr="127.0.0.1; touch /tmp/pwned_e2e_marker",
            master_port="29500",
        )
        if code != 403:
            fail(f"expected 403 for a command-injecting MASTER_ADDR, got {code}: {resp4}")
        print("OK: command-injecting MASTER_ADDR in WSL2 dispatch correctly rejected with HTTP 403")

        # Bonus: reject-on-bad-entry_point check (security constraint) --
        # rejected both because it's a path escape AND because it's not in
        # worker_daemon.py's ALLOWED_ENTRY_POINTS allowlist.
        bad_spec_yaml = "name: bad\nversion: 1\nentry_point: ../outside_repo.py\n"
        bad_spec = yaml.safe_load(bad_spec_yaml)
        code, resp2 = coord.dispatch_job(worker, bad_spec_yaml, bad_spec, rank=0, world_size=1,
                                          master_addr="127.0.0.1", master_port="29601")
        if code != 403:
            fail(f"expected 403 for a path-escaping entry_point, got {code}: {resp2}")
        print("OK: path-escaping entry_point correctly rejected with HTTP 403")

        # Bonus: reject-unauthenticated-dispatch check (security constraint
        # added after review -- /dispatch must require a valid bearer token).
        import json
        import urllib.request
        import urllib.error
        bad_url = f"http://127.0.0.1:{PORT}/dispatch"
        bad_req = urllib.request.Request(
            bad_url, data=json.dumps({
                "job_spec_yaml": job_spec_yaml_text, "rank": 0, "world_size": 1,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},  # deliberately no Authorization header
            method="POST",
        )
        try:
            urllib.request.urlopen(bad_req, timeout=5)
            fail("expected 401 for an unauthenticated /dispatch request, got success")
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                fail(f"expected 401 for unauthenticated /dispatch, got {exc.code}")
            print("OK: unauthenticated dispatch correctly rejected with HTTP 401")

        # /exec: run a real command on the worker and check its output.
        status_code, resp = coord.exec_on_worker(worker, "echo exec-test-marker")
        if status_code != 200:
            fail(f"exec did not return 200, got {status_code}: {resp}")
        if "exec-test-marker" not in resp.get("stdout", ""):
            fail(f"exec output missing expected marker: {resp}")
        print(f"OK: /exec ran a real command, stdout contained the expected marker")

        # /exec: unauthenticated request rejected (same pattern as /dispatch).
        exec_url = f"http://127.0.0.1:{PORT}/exec"
        bad_exec_req = urllib.request.Request(
            exec_url, data=json.dumps({"command": "echo should-not-run"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            urllib.request.urlopen(bad_exec_req, timeout=5)
            fail("expected 401 for an unauthenticated /exec request, got success")
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                fail(f"expected 401 for unauthenticated /exec, got {exc.code}")
            print("OK: unauthenticated /exec correctly rejected with HTTP 401")

        # /dataset-pull: serve a small real directory and fetch it via the
        # worker, into a scratch destination, then verify content landed.
        import shutil as _shutil
        import http.server as _http_server
        import socketserver as _socketserver
        import threading as _threading

        src_dir = Path(tempfile.mkdtemp(prefix="mgpu-test-src-"))
        (src_dir / "sub").mkdir()
        (src_dir / "a.txt").write_text("alpha")
        (src_dir / "sub" / "b.txt").write_text("beta")

        class _Handler(_http_server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=str(src_dir), **kw)
            def log_message(self, *a):
                pass

        file_server = _socketserver.TCPServer(("127.0.0.1", 0), _Handler)
        file_server_port = file_server.server_address[1]
        file_server_thread = _threading.Thread(target=file_server.serve_forever, daemon=True)
        file_server_thread.start()
        try:
            dest_rel = "coordinator/_test_dataset_pull_scratch"
            dest_abs = REPO_ROOT / dest_rel
            _shutil.rmtree(dest_abs, ignore_errors=True)
            status_code, resp = coord.request_dataset_pull(
                worker, f"http://127.0.0.1:{file_server_port}/", dest_rel,
            )
            if status_code != 200:
                fail(f"dataset-pull did not return 200, got {status_code}: {resp}")
            if resp.get("files_written") != 2:
                fail(f"expected 2 files written, got: {resp}")
            if not (dest_abs / "a.txt").read_text() == "alpha" or not (dest_abs / "sub" / "b.txt").read_text() == "beta":
                fail(f"dataset-pull wrote wrong content into {dest_abs}")
            print(f"OK: /dataset-pull fetched a real 2-file directory with correct content")

            # path traversal on dest must be refused.
            code, resp2 = coord.request_dataset_pull(
                worker, f"http://127.0.0.1:{file_server_port}/", "../outside_repo_scratch",
            )
            if code != 403:
                fail(f"expected 403 for a path-escaping dataset-pull dest, got {code}: {resp2}")
            print("OK: path-escaping dataset-pull dest correctly rejected with HTTP 403")
        finally:
            file_server.shutdown()
            _shutil.rmtree(src_dir, ignore_errors=True)
            _shutil.rmtree(dest_abs, ignore_errors=True)

        # Regression test for a real path-traversal / arbitrary-file-write
        # vulnerability found by a security review (fixed in _safe_join,
        # worker_daemon.py): a MALICIOUS server's directory-listing HTML
        # (not the dest argument, which was already validated) could
        # contain a percent-encoded traversal href like
        # "..%2f..%2f..%2fpwned.txt" that decodes to "../../../pwned.txt"
        # AFTER the old filename-extraction logic already split on "/",
        # letting it escape the destination directory. This test serves
        # exactly that malicious listing and asserts nothing gets written
        # outside the intended destination.
        class _MaliciousHandler(_http_server.BaseHTTPRequestHandler):
            def do_GET(self):
                html = (b'<html><body><ul>'
                        b'<li><a href="..%2f..%2f..%2fpwned_e2e_test.txt">x</a></li>'
                        b'</ul></body></html>')
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            def log_message(self, *a):
                pass

        malicious_server = _socketserver.TCPServer(("127.0.0.1", 0), _MaliciousHandler)
        malicious_port = malicious_server.server_address[1]
        malicious_thread = _threading.Thread(target=malicious_server.serve_forever, daemon=True)
        malicious_thread.start()
        traversal_dest_rel = "coordinator/_test_traversal_scratch"
        traversal_dest_abs = REPO_ROOT / traversal_dest_rel
        outside_marker = REPO_ROOT.parent / "pwned_e2e_test.txt"
        _shutil.rmtree(traversal_dest_abs, ignore_errors=True)
        outside_marker.unlink(missing_ok=True)
        try:
            code, resp3 = coord.request_dataset_pull(
                worker, f"http://127.0.0.1:{malicious_port}/", traversal_dest_rel,
            )
            if outside_marker.exists():
                fail(f"SECURITY REGRESSION: malicious href wrote a file outside dest at {outside_marker}")
            if code == 200:
                fail(f"expected a non-200 (blocked) response for a malicious traversal href, got 200: {resp3}")
            print(f"OK: malicious traversal href in directory listing correctly blocked "
                  f"(HTTP {code}), no file written outside destination")
        finally:
            malicious_server.shutdown()
            _shutil.rmtree(traversal_dest_abs, ignore_errors=True)
            outside_marker.unlink(missing_ok=True)

        print("\nE2E TEST PASSED")
    finally:
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()


if __name__ == "__main__":
    main()
