# Setting up your laptop for the Multi-GPU cluster

Hi dheeraj — this walks you through connecting your laptop's GPU to
Likith's training cluster. It's mostly automated, but there are three
points where Windows genuinely needs you to click something by hand —
those are called out clearly below so you know they're expected, not
something going wrong.

**Total time**: usually 20-40 minutes, plus a possible reboot in the
middle if WSL2 isn't already installed on your machine.

---

## Before you start

- You need to be logged into an account on this laptop that can install
  software (an "Administrator" account — most personal laptops are set
  up this way by default).
- You'll need a Google, Microsoft, GitHub, or email account to log into
  Tailscale (a free networking tool this uses) — whichever you find
  easiest.
- Keep this laptop plugged into power and connected to the internet for
  the whole process — one step downloads a few gigabytes.

---

## Step 1 — Install Git (if you don't already have it)

Open the **Start menu**, type `powershell`, and press Enter to open a
normal PowerShell window. Type:

```powershell
git --version
```

If it prints a version number, skip to Step 2.

If it says `git is not recognized`, download and install Git from
**https://git-scm.com/download/win** — just click Next through the
installer with the default options. Once it finishes, close this
PowerShell window (you'll open a new one in Step 2).

## Step 2 — Run the setup script, as Administrator

This is important: the script needs Administrator permission because it
installs system-level components (WSL2, a firewall rule). Windows will
ask you to confirm this — that's expected, not a warning sign.

1. Plug in the USB drive Likith gave you.
2. Open the USB drive in File Explorer, go into the
   `MultiGPU-Setup-DheerajLaptop` folder.
3. Right-click `dheeraj-bootstrap.ps1` and choose **"Run with
   PowerShell"**.
   - If Windows shows a blue "Windows protected your PC" screen, click
     **"More info"** then **"Run anyway"** — this happens for any
     script downloaded/copied from outside the Microsoft Store, it
     doesn't mean anything is wrong.
   - If Windows asks **"Do you want to allow this app to make changes
     to your device?"** (a UAC prompt), click **Yes**. This is the
     Administrator permission the script needs.

The script will now:
- Download a copy of the project from GitHub to `C:\multigpu-setup`.
- Hand off automatically to the real setup process.

## Step 3 — Watch for the three points that need you

The script does almost everything by itself, but there are three exact
moments where it will stop and wait for you:

### A. If it says a REBOOT is required

If your laptop doesn't already have WSL2 installed, you'll see a
message like:

```
[WARN] A REBOOT IS LIKELY REQUIRED...
NEXT STEP: reboot this machine now, then re-run this EXACT command...
```

**Just restart your laptop normally.** After it comes back on, repeat
Step 2 exactly the same way (run `dheeraj-bootstrap.ps1` again as
Administrator). It will skip the parts already done and continue from
where it left off — you won't lose progress.

### B. The Tailscale login

At some point a browser window will pop open asking you to log into
Tailscale. This is the one thing that genuinely can't be automated —
pick whichever account type is easiest (Google/Microsoft/GitHub/email)
and log in. Once you see "You are connected!" or similar in the
browser, go back to the PowerShell window and it should continue
automatically. If it doesn't within a minute or two, just re-run the
same command from Step 2 again.

### C. A few "this may take a while" waits

Installing WSL2 and downloading PyTorch (a few GB) both take a few
minutes with no visible progress at times — this is normal, just let it
run. Don't close the window unless it's been stuck with zero disk/network
activity for more than ~15 minutes.

## Step 4 — Confirm it worked

When everything finishes, you'll see a summary block that looks like:

```
=== SUMMARY ===
OK: 6   WARN: 0   FAIL: 0

Add this to coordinator/workers.yaml on the COORDINATOR machine:
  - name: YOUR-COMPUTER-NAME
    host: 100.x.y.z
    port: 8770
    token_file: coordinator/tokens/YOUR-COMPUTER-NAME.secret
```

**Take a screenshot or copy this block** and send it to Likith — this
is exactly what's needed on the other end to add your laptop to the
cluster.

## Step 5 — Send Likith one file

The setup created a small text file (your machine's security token) at:

```
C:\multigpu-setup\coordinator\tokens\YOUR-COMPUTER-NAME.secret
```

Send this file to Likith **directly** — over the same USB drive, a
private message, or similar. **Don't post it publicly or paste its
contents into a group chat** — it works like a password for your
laptop's training daemon. Likith needs the actual file, not just a
screenshot of it.

## What happens after that

Once Likith adds your laptop to his machine's list, your laptop's GPU
becomes part of the shared cluster. The worker program on your machine
now **starts automatically every time you log in** — you don't need to
open PowerShell or run anything again for normal use.

## Troubleshooting

- **"I ran it and nothing seems to have happened / it closed
  immediately"** — right-click PowerShell in the Start menu and choose
  "Run as Administrator" first, then navigate to the USB drive
  (`cd D:\MultiGPU-Setup-DheerajLaptop`, adjusting the drive letter to
  match your USB) and run
  `powershell -ExecutionPolicy Bypass -File dheeraj-bootstrap.ps1`
  manually — this shows any error message instead of the window closing.
- **"It says my dataset/files are missing"** — that's expected at this
  stage; the actual training data gets sent to your laptop separately,
  after your machine is confirmed connected. Don't worry about this yet.
- **Anything else that looks like a real error** (a red `[FAIL]` line
  with an error message, not one of the three expected pauses above) —
  screenshot it and send it to Likith rather than trying to fix it
  yourself; some of these need a small code change on his end.

## Why all this is needed (if you're curious)

Your GPU currently only runs on native Windows, which is missing a
piece (called NCCL) needed to combine two GPUs into one distributed
training job. The fix is running the actual training inside a Linux
environment on your machine (called WSL2), which Windows can run
side-by-side with everything else — that's most of what this script is
installing. Tailscale is just how your laptop and Likith's laptop find
and talk to each other securely over the internet without any complex
network configuration.
