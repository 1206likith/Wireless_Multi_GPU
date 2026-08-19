"""
WSL2-launched no-op job used ONLY to prove worker_daemon.py's
build_launch_argv WSL2 path (wsl.exe -> bash -> torchrun) actually works
end-to-end -- real NCCL process-group init/destroy, dispatched through the
real HTTP /dispatch endpoint -- without running the real (multi-hour) YOLO
DDP training script.

Uses torchrun's own single-node rendezvous (env://) since torchrun sets
RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT itself when launched
with --nnodes/--node_rank/--master_addr/--master_port, same as the real
training script.
"""
import os
import sys
import time

import torch
import torch.distributed as dist

rank = os.environ.get("RANK", "?")
world_size = os.environ.get("WORLD_SIZE", "?")
master_addr = os.environ.get("MASTER_ADDR", "?")
master_port = os.environ.get("MASTER_PORT", "?")

print(f"[test_noop_job_wsl2] rank={rank} world_size={world_size} "
      f"master={master_addr}:{master_port} "
      f"cuda={torch.cuda.is_available()} nccl={dist.is_nccl_available()}", flush=True)

dist.init_process_group(backend="nccl", init_method="env://")
print("[test_noop_job_wsl2] NCCL process group initialized", flush=True)
time.sleep(1)
dist.destroy_process_group()
print("[test_noop_job_wsl2] done", flush=True)
sys.exit(0)
