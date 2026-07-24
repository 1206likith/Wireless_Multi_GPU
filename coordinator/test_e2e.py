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

        print("\nE2E TEST PASSED")
    finally:
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()


if __name__ == "__main__":
    main()
