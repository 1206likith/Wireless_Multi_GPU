# Multi-laptop DDP training setup checklist

This is the setup checklist for connecting a second (or third) laptop's GPU
to this one for distributed YOLOv8 training. It assumes you've already
read the reasoning in `36b_train_yolo_v4_ddp_wsl2.py`'s docstring for *why*
this uses WSL2 + NCCL rather than native Windows (short version: native
Windows PyTorch has no NCCL backend at all, and the gloo-backend
workaround segfaults on `.backward()` — root-caused this session,
not fixable from Python).

## Phase 0 — this machine (DONE, 2026-07-23/24)

- WSL2 (Ubuntu 24.04) installed, GPU passthrough verified
  (`wsl -d Ubuntu-24.04 -- nvidia-smi` shows the RTX 5060).
- PyTorch 2.11.0+cu128 installed in a venv at `~/yolo_ddp_env` inside
  WSL2, with NCCL confirmed available
  (`torch.distributed.is_nccl_available()` → `True`).
- Full DDP forward/backward/optimizer-step cycle tested in isolation —
  no segfault (unlike the native-Windows gloo attempt).
- A 21-batch real-dataset training loop test passed cleanly at
  ~4 batches/sec.
- **Known ongoing issue**: WSL2's background VM process (`vmmemWSL`) grows
  over time and competes with native Windows training jobs for system
  RAM — this caused 3 crashed segmentation-training attempts in one
  session. `wsl --shutdown` (via PowerShell, not bash — bash-wrapped
  calls were less reliable) frees it, but it restarts on its own after a
  while. **Don't run WSL2-dependent work and RAM-heavy native training at
  the same time on this machine** — alternate between them, shutting WSL2
  down before returning to native training.

## Phase 1 — second (and further) laptop(s): what YOU need to do

I (the assistant) have no access to the second laptop — everything below
needs to be run there directly, either by you or by pointing an agent
session at that machine if one becomes available.

### 1. Install WSL2 with the same distro
```powershell
wsl --install -d Ubuntu-24.04
```
Reboot if prompted. Verify GPU passthrough:
```bash
wsl -d Ubuntu-24.04 -- nvidia-smi
```
This should show that laptop's GPU (e.g. the RTX 5070). If it doesn't,
GPU passthrough isn't working and needs troubleshooting (usually an
outdated NVIDIA driver — needs 525.60.13+ on Windows for CUDA-in-WSL2
support) before anything else here will work.

### 2. Set up Python + PyTorch inside that WSL2 instance
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv
python3 -m venv ~/yolo_ddp_env
source ~/yolo_ddp_env/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy pyyaml pillow ultralytics
```
Verify NCCL is available (this is the whole point of using WSL2 instead
of native Windows):
```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.distributed.is_nccl_available())"
```
Both should print `True`. If CUDA isn't available here, GPU passthrough
from step 1 didn't work — fix that first.

### 3. Get the dataset onto that machine
The training script (`36b_train_yolo_v4_ddp_wsl2.py`) expects
`data/yolo_detect_v4/` to be reachable from WSL2. Two options:
- **Copy the whole `full-project/prototype/data/yolo_detect_v4/` folder**
  (about 1GB) to the same relative path on the second laptop, and set the
  `DATA_ROOT` env var to wherever it ends up (see script's docstring).
- **Or**, if the second laptop can see this laptop's files over the
  network (e.g. a shared drive), point `DATA_ROOT` at that shared path
  instead — untested, but should work the same way this laptop's own
  `/mnt/l/...` Windows-drive mount does.

### 4. Networking — THE PART NOT YET VERIFIED THIS SESSION

WSL2's default networking mode is NAT — each WSL2 instance gets a private
IP that's only reachable from its own Windows host, not from another
machine on the LAN. This will block the rendezvous step (both machines
need to reach the same `MASTER_ADDR:MASTER_PORT`).

**The fix to try first**: Windows 11 22H2+ supports "mirrored" networking
mode for WSL2, which makes the WSL2 instance share the host's network
identity (so it's reachable at the same IP as the Windows machine itself).
On **both** machines, create/edit `%USERPROFILE%\.wslconfig`:
```ini
[wsl2]
networkingMode=mirrored
```
Then on both machines:
```powershell
wsl --shutdown
```
and restart WSL2. Verify with `ipconfig` (Windows side) and `ip addr`
(inside WSL2) that they now report matching/compatible LAN-reachable
addresses.

**If mirrored mode isn't available or doesn't work**: the fallback is
port forwarding from the Windows host into WSL2 (`netsh interface portproxy`
on Windows), which is more fragile to set up correctly — try mirrored mode
first.

**Verify connectivity before attempting training**: from the second
laptop's WSL2, ping this laptop's LAN IP; then try a simple TCP connection
test on the intended port (29500) before trusting a full DDP rendezvous
to work.

### 5. Launch the actual training

Once networking is verified, follow the LAUNCH instructions in
`36b_train_yolo_v4_ddp_wsl2.py`'s docstring — run the script inside WSL2
on both machines via `torchrun`, one as `--node_rank=0` (this machine,
the master) and the other as `--node_rank=1`.

**Recommended first step before a real training run**: do a short smoke
test (a handful of batches, like the 21-batch test this session ran
locally) with `world_size=2` across both real machines, to confirm the
cross-machine rendezvous and gradient sync actually work, before
committing to a multi-hour training run.

## Status as of 2026-07-24

- Phase 0: done and verified.
- Phase 1 steps 1-3: not started (need second laptop access/details).
- Phase 1 step 4 (networking): not verified even in principle — this is
  the biggest unknown remaining and should be tackled first once the
  second laptop is reachable, since everything after it depends on it
  working.
