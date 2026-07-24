# Multi-GPU DDP Training Setup

One-step environment setup for joining a second (or further) GPU laptop to
this project's distributed YOLOv8 training, over WSL2 + NCCL (see
`DDP_MULTINODE_SETUP.md` for the full reasoning and manual steps).

## Quick start (on the second laptop)

**If WSL2 isn't installed yet**, open PowerShell **as Administrator** on
Windows and run:
```powershell
wsl --install -d Ubuntu-24.04
```
Reboot when prompted, then open the new "Ubuntu" app from the Start menu.

**Inside that Ubuntu shell**, run:
```bash
curl -fsSL https://raw.githubusercontent.com/1206likith/Wireless_Multi_GPU/main/install.sh | bash
```
(Replace `1206likith/Wireless_Multi_GPU` with wherever this repo actually lives
— see the URL in your browser or `git remote -v` in this repo.)

This automatically:
- Confirms you're in WSL2 and GPU passthrough works
- Installs Python/pip/venv if missing
- Installs PyTorch (CUDA 12.8) + ultralytics into a venv at `~/yolo_ddp_env`
- Verifies NCCL is available (the whole point of using WSL2 over native
  Windows for this — see `DDP_MULTINODE_SETUP.md`)
- Clones this repo's training script into `~/yolo_ddp_project`

It prints a clear summary at the end: what succeeded automatically, and
what still needs manual action (there are two things it genuinely can't
automate — see below).

## What still needs manual steps (can't be automated by install.sh)

1. **The dataset.** `data/yolo_detect_v4/` (~1GB) needs to physically get
   onto the second laptop — ask whoever's running the main laptop for
   the current transfer method (shared drive, direct file copy, etc).
   Set the `DATA_ROOT` environment variable to point at wherever it ends
   up if it's not at the same path as the main laptop.
2. **Networking.** WSL2 defaults to a private NAT network not reachable
   from another machine on the LAN. On **both** laptops, create/edit
   `%USERPROFILE%\.wslconfig` (Windows side):
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```
   Then (from Windows PowerShell) `wsl --shutdown` and reopen Ubuntu on
   both machines. This is the one piece not yet verified as working —
   test it before trusting a full training run to it.

Once both of those are done, see `DDP_MULTINODE_SETUP.md`'s launch
instructions (or `scripts/36b_train_yolo_v4_ddp_wsl2.py`'s own docstring)
for the exact `torchrun` commands to run on each machine.

## Repo contents

- `install.sh` — the one-step setup script described above.
- `scripts/36b_train_yolo_v4_ddp_wsl2.py` — the actual multi-node DDP
  training script (run identically on every participating machine, each
  with a different `--node_rank`).
- `DDP_MULTINODE_SETUP.md` — full setup checklist and background on why
  this uses WSL2 + NCCL instead of native Windows PyTorch (short version:
  native Windows has no NCCL backend at all, and the workaround segfaults
  — root-caused in an earlier session, not fixable from Python).
