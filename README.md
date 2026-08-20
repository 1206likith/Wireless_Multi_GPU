# Multi-GPU DDP Training Setup

A coordinator + worker system for joining 2+ Windows/WSL2 GPU laptops into
one distributed YOLOv8 training cluster over Tailscale + NCCL. One script
(`setup-worker.ps1`) takes a new laptop from nothing installed to a
running, auto-starting worker — see "One-shot worker setup" below.

## One-shot worker setup (on each new laptop)

On a **completely fresh** Windows machine (nothing installed yet), open
PowerShell **as Administrator** and run:
```powershell
git clone https://github.com/1206likith/Wireless_Multi_GPU.git C:\multigpu-setup
cd C:\multigpu-setup
powershell -ExecutionPolicy Bypass -File setup-worker.ps1
```
(Replace the repo URL with wherever this repo actually lives — see
`git remote -v` in this checkout, or ask the project owner.)

`setup-worker.ps1` chains together everything this project previously
required as separate manual steps:
1. Installs WSL2 + Ubuntu-24.04 if not already present.
2. Installs Tailscale (with signature verification) and sets up WSL2
   mirrored networking + the required firewall rule.
3. Installs PyTorch/NCCL/ultralytics inside WSL2 (delegates to
   `install.sh`, same as before).
4. Generates this worker's dispatch token.
5. Registers `worker_daemon.py` as a Windows Scheduled Task that starts
   automatically at login and restarts itself if it crashes.
6. Verifies the daemon actually answers over Tailscale before finishing.

**Three things genuinely cannot be one click**, and the script tells you
exactly what to do for each rather than silently failing:
- **A reboot**, if WSL2 wasn't already installed — this is a real Windows
  kernel-component requirement, not something a script can skip. The
  script detects this, tells you to reboot and re-run the *exact same
  command* (it's idempotent — already-done steps are skipped on rerun),
  and exits cleanly.
- **Tailscale's first login** — an interactive OAuth browser flow
  (Google/Microsoft/GitHub/email). The script opens it for you; you
  click through it once.
- **Administrator elevation itself** — Windows requires this for WSL2,
  firewall rules, and the Scheduled Task; there's no way around the UAC
  prompt, by design.

At the end, the script prints the exact `workers.yaml` entry to add on
the coordinator machine, and reminds you to copy the generated token
file over out-of-band (never over this project's own HTTP channel).

## What still needs manual steps after setup-worker.ps1

1. **The dataset.** `data/yolo_detect_v4/` (~1GB) needs to physically get
   onto the new machine. Use the coordinator's `send-dataset` command
   (or the dashboard's future dataset-push action) from the main laptop
   once this worker is reachable — see `coordinator/README.md`.
2. **Adding the worker to `workers.yaml`** on the coordinator machine —
   the script prints the exact entry to add; this repo's `workers.yaml`
   isn't shared automatically across machines (each machine has its own
   checkout).

Once both of those are done, `coordinator.py dispatch
job-specs/yolo-detect-v4-ddp.yaml` (or the dashboard's Dispatch button)
runs the real training job across every configured worker.

## Repo contents

- `setup-worker.ps1` — the one-shot Windows-side setup script described
  above; the actual entry point for adding a new machine to the cluster.
- `install-host-tailscale.ps1` — Tailscale + WSL2 mirrored-networking
  setup, called by `setup-worker.ps1` (also runnable standalone).
- `install.sh` — WSL2-side Python/PyTorch/NCCL setup, called by
  `setup-worker.ps1` via `wsl -d Ubuntu-24.04 -- bash install.sh` (also
  runnable standalone from inside WSL2, unchanged from before).
- `coordinator/` — the coordinator, worker daemon, and control API; see
  `coordinator/README.md` for how to dispatch jobs and use the web
  dashboard.
- `scripts/36b_train_yolo_v4_ddp_wsl2.py` — the actual multi-node DDP
  training script.
- `DDP_MULTINODE_SETUP.md` — background on why this uses WSL2 + NCCL
  instead of native Windows PyTorch (native Windows has no NCCL backend
  at all, and the workaround segfaults — root-caused in an earlier
  session, not fixable from Python).
