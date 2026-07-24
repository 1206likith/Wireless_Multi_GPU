"""
WSL2 + NCCL version of 36_train_yolo_v4_ddp_multinode.py.

WHY THIS SCRIPT EXISTS SEPARATELY FROM 36_train_yolo_v4_ddp_multinode.py:
that script targets native Windows PyTorch (gloo backend, direct
init_process_group bypassing torchrun's broken rendezvous) because at the
time it was written, no NCCL backend was available at all on this machine.
Since then, this session validated a working WSL2 (Ubuntu 24.04) + NCCL
setup end-to-end: installed PyTorch 2.11.0+cu128 in a WSL2 venv
(~/yolo_ddp_env), confirmed GPU passthrough (nvidia-smi inside WSL2 sees
the RTX 5060), confirmed torch.distributed.is_nccl_available() == True,
and reran this project's exact DDP forward/backward/optimizer-step repro
inside WSL2 with backend="nccl" -- it completed cleanly with NO segfault
(the native-Windows gloo version segfaulted on .backward() in isolated
testing, root-caused to PyTorch's own C++ DDP Reducer internals, not
fixable from Python). A 21-batch real-dataset test also passed cleanly at
~4 batches/sec.

This script is otherwise IDENTICAL in structure/logic to
36_train_yolo_v4_ddp_multinode.py (same model-construction fix, same
"call model(batch) through DDP's own __call__" gradient-flow fix) --
only the distributed backend changed (nccl instead of gloo) and torchrun
CAN be used normally here (WSL2's PyTorch build has proper libuv/NCCL
support, so the native-Windows torchrun blockers don't apply).

REQUIREMENTS BEFORE RUNNING (see DDP_MULTINODE_SETUP.md for the full
second-laptop checklist -- this section assumes that setup is done):
  - WSL2 (Ubuntu 24.04 or similar) installed and GPU-passthrough-verified
    on BOTH machines (check: `wsl -d Ubuntu-24.04 -- nvidia-smi` shows the
    GPU on each).
  - Matching PyTorch/CUDA/ultralytics versions in each machine's WSL2 venv
    (this machine: torch 2.11.0+cu128, ultralytics 8.4.104, venv at
    ~/yolo_ddp_env inside WSL2).
  - data/yolo_detect_v4/ accessible from WSL2 on both machines at the
    SAME path relative to this script (this machine accesses it via the
    /mnt/l/... Windows-drive mount; the second laptop's drive letter/path
    may differ -- adjust DATA_YAML's resolution if so, see inline note).
  - Both WSL2 instances able to reach each other over the LAN. WSL2's
    default NAT networking is NOT directly reachable from another
    machine -- this needs Windows 11 "mirrored" networking mode (set
    `networkingMode=mirrored` in %USERPROFILE%\.wslconfig on BOTH
    machines, then `wsl --shutdown` + restart WSL2) OR a port-forwarding
    workaround. NOT YET VERIFIED as of this session -- this is the next
    unconfirmed piece of the setup, flagged in DDP_MULTINODE_SETUP.md.

LAUNCH (run inside WSL2 on BOTH machines, via torchrun this time --
WSL2's PyTorch has working rendezvous, unlike native Windows):
  This machine (rank 0, the "master"):
    source ~/yolo_ddp_env/bin/activate
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
      --master_addr=<THIS_MACHINE_LAN_IP> --master_port=29500 \
      36b_train_yolo_v4_ddp_wsl2.py

  Second laptop (rank 1), inside its own WSL2:
    source ~/yolo_ddp_env/bin/activate
    PER_GPU_BATCH=12 torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
      --master_addr=<THIS_MACHINE_LAN_IP> --master_port=29500 \
      36b_train_yolo_v4_ddp_wsl2.py

Note: torchrun sets RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT
automatically as env vars when launched this way -- unlike the native-
Windows script, this one does NOT need them set by hand first (though it
still reads them the same way, for consistency and so it degrades
gracefully to a manual-env-var launch if torchrun itself has issues on a
given machine).
"""
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import DEFAULT_CFG

# This machine's WSL2 accesses the Windows L: drive via /mnt/l/. If the
# second laptop's data lives at a different mount point (different drive
# letter, or copied into WSL2's native filesystem instead of mounted),
# override via the DATA_ROOT env var rather than editing this path.
DATA_ROOT = Path(os.environ.get(
    "DATA_ROOT",
    "/mnt/l/Likith/Coding_Projects/IDP_D_J/full-project/prototype",
))
DATA_YAML = DATA_ROOT / "data" / "yolo_detect_v4" / "data.yaml"
RUNS_DIR = DATA_ROOT / "runs" / "yolo_detect_v4_ddp_wsl2"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 150
IMGSZ = 640
PER_GPU_BATCH = int(os.environ.get("PER_GPU_BATCH", "8"))


def setup_distributed():
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    dist.init_process_group(
        backend="nccl",  # confirmed available in this session's WSL2 PyTorch build
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def build_dataloader(data_cfg, split, rank, world_size, batch):
    dataset = YOLODataset(
        img_path=str(DATA_ROOT / "data" / "yolo_detect_v4" / data_cfg[split]),
        data=data_cfg,
        imgsz=IMGSZ,
        augment=(split == "train"),
        rect=False,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=(split == "train")
    )
    loader = DataLoader(
        dataset,
        batch_size=batch,
        sampler=sampler,
        num_workers=0,
        collate_fn=YOLODataset.collate_fn,
        pin_memory=True,
    )
    return loader, sampler


def main():
    rank, world_size, local_rank = setup_distributed()
    is_master = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    if is_master:
        print(f"[ddp-wsl2] world_size={world_size}, this rank={rank}, device={device}", flush=True)

    with open(DATA_YAML) as f:
        data_cfg = yaml.safe_load(f)
    data_cfg["nc"] = len(data_cfg["names"])
    n_classes = data_cfg["nc"]

    base_model = DetectionModel("yolov8s.yaml", ch=3, nc=n_classes, verbose=is_master)
    pretrained, _ = load_checkpoint(str(DATA_ROOT / "yolov8s.pt"))
    base_model.load(pretrained)
    base_model = base_model.to(device)
    base_model.args = DEFAULT_CFG
    model = DDP(base_model, device_ids=[local_rank])

    train_loader, train_sampler = build_dataloader(data_cfg, "train", rank, world_size, PER_GPU_BATCH)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.937, weight_decay=0.0005)

    best_loss = float("inf")
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch["img"] = batch["img"].to(device, non_blocking=True).float() / 255.0
            for k in ("cls", "bboxes", "batch_idx"):
                batch[k] = batch[k].to(device, non_blocking=True)
            optimizer.zero_grad()
            loss, loss_items = model(batch)  # must go through DDP.__call__, see 36_...py docstring
            loss.sum().backward()
            optimizer.step()
            running_loss += loss.sum().item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        elapsed = time.time() - epoch_start

        if is_master:
            print(f"[ddp-wsl2] epoch {epoch+1}/{EPOCHS} avg_loss={avg_loss:.4f} time={elapsed:.1f}s", flush=True)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.module.state_dict(), RUNS_DIR / "best_ddp.pt")
            torch.save(model.module.state_dict(), RUNS_DIR / "last_ddp.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
