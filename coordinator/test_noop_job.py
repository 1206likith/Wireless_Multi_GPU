"""
Trivial no-op "job" used ONLY by coordinator/test_e2e.py to prove the
coordinator -> worker -> subprocess -> status-report loop works, without
running the real (multi-hour) YOLO DDP training script and without needing
a second physical machine.

Reads RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT the same way the real
36b_train_yolo_v4_ddp_wsl2.py script does (so dispatch/env-injection is
exercised identically), sleeps briefly, then exits 0.
"""
import os
import sys
import time

rank = os.environ.get("RANK", "?")
world_size = os.environ.get("WORLD_SIZE", "?")
master_addr = os.environ.get("MASTER_ADDR", "?")
master_port = os.environ.get("MASTER_PORT", "?")

print(f"[test_noop_job] rank={rank} world_size={world_size} "
      f"master={master_addr}:{master_port}", flush=True)
time.sleep(2)
print("[test_noop_job] done", flush=True)
sys.exit(0)
