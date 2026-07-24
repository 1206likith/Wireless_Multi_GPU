#!/usr/bin/env bash
# One-step setup for a second (or third) GPU laptop to join the multi-GPU
# YOLOv8 training project. Run this INSIDE WSL2 Ubuntu on Windows (NOT in
# native Windows PowerShell/CMD), after WSL2 itself is installed.
#
# Usage (from inside WSL2 Ubuntu):
#   curl -fsSL https://raw.githubusercontent.com/<GITHUB_USER>/<REPO>/main/install.sh | bash
# or, after cloning the repo:
#   bash install.sh
#
# WHY THIS DOESN'T ALSO INSTALL WSL2 ITSELF:
# WSL2 installation (`wsl --install`) happens on the WINDOWS side, requires
# admin rights, and typically needs a REBOOT before Ubuntu can be entered
# for the first time -- that can't be automated from inside a bash script
# running in WSL2 (chicken-and-egg: this script assumes WSL2/Ubuntu already
# works enough to run bash at all). If WSL2 isn't installed yet, run this
# in Windows PowerShell (as Administrator) FIRST, reboot, then come back
# and run this script inside the resulting Ubuntu shell:
#   wsl --install -d Ubuntu-24.04
#
# WHAT THIS SCRIPT DOES:
#   1. Verifies it's actually running inside WSL2 (refuses to run on
#      native Linux or plain Windows, to avoid confusing failures).
#   2. Verifies GPU passthrough works (nvidia-smi sees a GPU).
#   3. Installs python3-pip/venv if missing, creates ~/yolo_ddp_env.
#   4. Installs PyTorch (CUDA 12.8 build) + ultralytics + deps into that venv.
#   5. Verifies NCCL is available in that PyTorch build (the whole point
#      of using WSL2 instead of native Windows -- see this repo's
#      DDP_MULTINODE_SETUP.md for why).
#   6. Clones/pulls this repo's training scripts into ~/yolo_ddp_project.
#   7. Prints a clear final status: what's ready, what still needs manual
#      action (dataset transfer, networking mode), and the exact next
#      command to run.
#
# This script is idempotent -- safe to re-run if an earlier step failed;
# already-satisfied steps are skipped.

set -uo pipefail  # NOT -e: we want to detect failures ourselves and report
                  # them clearly rather than dying on the first error with
                  # a raw bash traceback the user can't act on.

VENV_DIR="$HOME/yolo_ddp_env"
PROJECT_DIR="$HOME/yolo_ddp_project"
REPO_URL="__REPO_URL_PLACEHOLDER__"  # filled in by publish step, see README

STATUS_OK=()
STATUS_WARN=()
STATUS_FAIL=()

ok()   { STATUS_OK+=("$1");   echo "[OK]   $1"; }
warn() { STATUS_WARN+=("$1"); echo "[WARN] $1"; }
fail() { STATUS_FAIL+=("$1"); echo "[FAIL] $1"; }

echo "=== Multi-GPU DDP training setup ==="
echo ""

# --- Step 1: confirm we're in WSL2 ---
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    fail "This doesn't look like WSL2 (no 'microsoft' in /proc/version)."
    echo "Run this script INSIDE a WSL2 Ubuntu shell, not native Linux or Windows."
    echo "If WSL2 isn't installed yet: open PowerShell AS ADMINISTRATOR on Windows and run:"
    echo "  wsl --install -d Ubuntu-24.04"
    echo "then reboot, open the new Ubuntu app, and re-run this script inside it."
    exit 1
fi
ok "Running inside WSL2."

# --- Step 2: GPU passthrough ---
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU passthrough working: $GPU_NAME"
else
    fail "nvidia-smi not found or failed -- GPU passthrough is not working."
    echo "This usually means the Windows NVIDIA driver needs updating (525.60.13+"
    echo "required for CUDA-in-WSL2) or WSL2 itself needs a restart:"
    echo "  (from Windows PowerShell) wsl --shutdown"
    echo "then reopen Ubuntu and re-run this script."
    exit 1
fi

# --- Step 3: Python/pip/venv ---
if ! command -v pip3 >/dev/null 2>&1; then
    echo "Installing python3-pip / python3-venv (needs sudo)..."
    sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip python3-venv
fi
if command -v pip3 >/dev/null 2>&1; then
    ok "pip3 available."
else
    fail "Could not install pip3. Check sudo access and internet connectivity."
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "Venv ready at $VENV_DIR."

# --- Step 4: PyTorch + deps ---
if ! python3 -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch (CUDA 12.8 build) -- this downloads several GB,"
    echo "may take 10-20+ minutes depending on connection speed..."
    pip install --upgrade pip -q
    pip install torch --index-url https://download.pytorch.org/whl/cu128
else
    ok "PyTorch already installed."
fi

if ! python3 -c "import ultralytics" 2>/dev/null; then
    echo "Installing numpy/scipy/pyyaml/pillow/ultralytics..."
    pip install numpy scipy pyyaml pillow ultralytics -q
else
    ok "ultralytics already installed."
fi

# --- Step 5: verify NCCL ---
NCCL_CHECK=$(python3 -c "
import torch
print('CUDA_OK' if torch.cuda.is_available() else 'CUDA_FAIL')
print('NCCL_OK' if torch.distributed.is_nccl_available() else 'NCCL_FAIL')
" 2>/dev/null)

if echo "$NCCL_CHECK" | grep -q "CUDA_OK"; then
    ok "torch.cuda.is_available() == True"
else
    fail "torch.cuda.is_available() == False -- GPU passthrough to PyTorch is broken."
fi

if echo "$NCCL_CHECK" | grep -q "NCCL_OK"; then
    ok "NCCL backend available (this is the whole point of using WSL2)."
else
    fail "NCCL not available in this PyTorch build -- something is wrong with the install."
fi

# --- Step 6: project code ---
if [ "$REPO_URL" = "__REPO_URL_PLACEHOLDER__" ]; then
    warn "REPO_URL placeholder not filled in -- skipping project clone step."
    warn "Manually clone the training-scripts repo into $PROJECT_DIR."
elif [ -d "$PROJECT_DIR/.git" ]; then
    echo "Updating existing clone at $PROJECT_DIR..."
    git -C "$PROJECT_DIR" pull --ff-only && ok "Project code updated." || warn "git pull failed -- check for local changes or conflicts."
else
    echo "Cloning project code into $PROJECT_DIR..."
    git clone "$REPO_URL" "$PROJECT_DIR" && ok "Project code cloned." || fail "git clone failed -- check REPO_URL and network access."
fi

# --- Step 7: final report ---
echo ""
echo "=== SETUP SUMMARY ==="
echo "OK:   ${#STATUS_OK[@]} checks passed"
echo "WARN: ${#STATUS_WARN[@]} items need manual attention"
echo "FAIL: ${#STATUS_FAIL[@]} items are broken"
echo ""

if [ ${#STATUS_FAIL[@]} -gt 0 ]; then
    echo "NOT READY -- fix the FAIL items above before training."
    exit 1
fi

echo "STILL NEEDED (not automatable from this script):"
echo "  1. Dataset transfer: copy data/yolo_detect_v4/ (~1GB) from the main"
echo "     laptop to this machine. Ask the project owner for the current"
echo "     transfer method (shared drive, direct copy, etc)."
echo "  2. Networking: WSL2 defaults to NAT, not reachable from another"
echo "     machine. On BOTH laptops, add to %USERPROFILE%\\.wslconfig:"
echo "       [wsl2]"
echo "       networkingMode=mirrored"
echo "     then run (from Windows PowerShell): wsl --shutdown"
echo "     and reopen Ubuntu on both machines."
echo "  3. See DDP_MULTINODE_SETUP.md in the cloned project for the exact"
echo "     launch commands once 1 and 2 are done."
echo ""
echo "Everything this script CAN automate is done. Environment is ready"
echo "for training once the dataset is in place and networking is verified."
