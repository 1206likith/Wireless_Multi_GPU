# One-shot worker setup: takes a Windows laptop from "nothing installed"
# to "worker_daemon.py running and auto-starting on login", chaining
# together everything this project previously required as separate manual
# steps: WSL2, Tailscale + mirrored networking, PyTorch/NCCL inside WSL2,
# a dispatch token, and a Windows Scheduled Task so the daemon survives
# reboots/logins without anyone opening a terminal again.
#
# WHAT THIS CANNOT MAKE ONE CLICK (documented, not silently glossed over):
#   1. WSL2 kernel-component installation, if not already present, requires
#      a REAL WINDOWS REBOOT before Ubuntu can be entered for the first
#      time. This is a Windows/Hyper-V constraint, not something a script
#      can work around -- this script detects that case, tells you
#      exactly what to do (reboot, then re-run this exact script), and
#      exits cleanly rather than pretending to continue on a half-working
#      WSL2 install.
#   2. Tailscale's first-time login is an interactive OAuth browser flow
#      (Google/Microsoft/GitHub/email) -- cannot be scripted without
#      capturing credentials, which this project will never do. This
#      script opens the login for you and waits, but you still click
#      through it once.
#   3. Administrator elevation is required (WSL2, firewall rules, and the
#      Scheduled Task all need it) -- Windows itself requires an explicit
#      UAC consent for this, which is intentional and not bypassable.
#
# Everything else -- WSL2 mirrored networking, the Hyper-V firewall rule,
# the WSL2-side Python venv/PyTorch/NCCL install, generating this worker's
# dispatch token, and making worker_daemon.py auto-start -- is fully
# automated by this script.
#
# USAGE (run as Administrator, from this repo's root on the NEW machine):
#   powershell -ExecutionPolicy Bypass -File setup-worker.ps1
#
# Idempotent -- safe to re-run after a reboot or a partial failure;
# already-completed steps are detected and skipped.

param(
    [string]$WorkerName = $env:COMPUTERNAME,
    [int]$WorkerPort = 8770,
    [string]$WslDistro = "Ubuntu-24.04"
)

$ErrorActionPreference = "Continue"
$RepoRoot = $PSScriptRoot
$okCount = 0
$warnCount = 0
$failCount = 0

function Write-Ok($msg)   { Write-Host "[OK]   $msg" -ForegroundColor Green;  $script:okCount++ }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow; $script:warnCount++ }
function Write-Fail($msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red;    $script:failCount++ }
function Write-Step($msg) { Write-Host ""; Write-Host "--- $msg ---" -ForegroundColor Cyan }

Write-Host "=== One-shot worker setup: $WorkerName ===" -ForegroundColor Cyan
Write-Host "Repo root: $RepoRoot"
Write-Host ""

# --- Step 0: confirm Administrator ---
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Fail "Not running as Administrator. Right-click PowerShell and choose 'Run as Administrator', then re-run:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit 1
}
Write-Ok "Running as Administrator."

# --- Step 1: WSL2 itself ---
Write-Step "Step 1/6: WSL2"
# `wsl --status` exits 0 once the WSL2 platform is installed, even if no
# distro is registered yet -- distinct from `wsl -l -v` which lists
# registered distros. Check both: platform present, AND our target distro
# registered, since a machine could have WSL2 with a different distro only.
$wslPlatformPresent = $false
try {
    wsl --status 2>&1 | Out-Null
    $wslPlatformPresent = ($LASTEXITCODE -eq 0)
} catch {
    $wslPlatformPresent = $false
}

$distroRegistered = $false
if ($wslPlatformPresent) {
    $distroList = (wsl -l -q 2>&1) -join "`n"
    # wsl -l -q prints UTF-16-ish output through some consoles with embedded
    # nulls/odd spacing -- normalize before matching so a real match isn't
    # missed due to encoding artifacts, not just literal string comparison.
    $normalized = ($distroList -replace "`0", "") -replace "\s+", ""
    # NOTE: the inner (...) around the -replace expression is required, not
    # cosmetic -- without it, PowerShell's static-method-call parser reads
    # "-replace" as a second positional argument to Escape() rather than
    # part of the expression producing the first argument, and throws
    # "Cannot find an overload for 'Escape' and the argument count: 2" at
    # runtime (confirmed by reproducing this exact error on Windows
    # PowerShell 5.1 with the un-parenthesized form, then confirming the
    # parenthesized form works correctly -- found via a real run on
    # dheeraj's laptop, not caught by [scriptblock]::Create syntax-checking
    # alone, since that only validates parseability, not this kind of
    # argument-binding ambiguity).
    $distroRegistered = $normalized -match [regex]::Escape(($WslDistro -replace "\s+", ""))
}

if ($wslPlatformPresent -and $distroRegistered) {
    Write-Ok "WSL2 + $WslDistro already installed."
} else {
    Write-Host "Installing WSL2 + $WslDistro (this can take several minutes)..."
    wsl --install -d $WslDistro
    $installExit = $LASTEXITCODE

    # Don't trust $installExit alone to decide reboot-vs-ready: a real
    # session found `wsl --install` can fail with a transient network error
    # (WININET_E_NAME_NOT_RESOLVED fetching the distro list) while STILL
    # partially registering the distro, so a later rerun hits
    # ERROR_ALREADY_EXISTS even though the distro is registered but was
    # never actually confirmed usable -- and separately, the distro CAN
    # already be fully usable (e.g. registered by a previous run, or by
    # the user directly) even when this specific `wsl --install` call
    # reports a nonzero/error exit code, because "already exists" isn't a
    # real failure. Ground truth is only "can I actually run a command
    # inside it right now" -- check that directly instead of trusting exit
    # codes or reasoning about install-command semantics.
    $distroActuallyUsable = $false
    try {
        wsl -d $WslDistro -- true 2>&1 | Out-Null
        $distroActuallyUsable = ($LASTEXITCODE -eq 0)
    } catch {
        $distroActuallyUsable = $false
    }

    if ($distroActuallyUsable) {
        Write-Ok "WSL2 + $WslDistro is installed and usable (verified with a real command, not just the install command's exit code)."
    } else {
        # A fresh WSL2 kernel-component install almost always requires a
        # real reboot before `wsl -d <distro>` will work at all -- this is
        # a Windows constraint (Hyper-V/Virtual Machine Platform optional
        # components need a restart to activate), not something this
        # script can skip. Only tell the user to reboot once the distro is
        # CONFIRMED not actually usable yet, not just because the install
        # command itself reported a nonzero exit code.
        Write-Host ""
        Write-Warn "WSL2 install command finished (exit code $installExit), and $WslDistro is not usable yet. A REBOOT IS LIKELY REQUIRED before it can actually be entered for the first time -- this is a Windows requirement, not something this script can bypass."
        Write-Host ""
        Write-Host "NEXT STEP: reboot this machine now, then re-run this EXACT command (as Administrator):" -ForegroundColor Yellow
        Write-Host "  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "This script is idempotent -- re-running it will detect WSL2 is now ready and continue from Step 2 automatically." -ForegroundColor Yellow
        exit 2
    }
}

# --- Step 2: Tailscale + mirrored networking (delegates to the existing,
#     already-reviewed script rather than duplicating its logic) ---
Write-Step "Step 2/6: Tailscale + WSL2 mirrored networking"
$tailscaleScript = Join-Path $RepoRoot "install-host-tailscale.ps1"
if (-not (Test-Path $tailscaleScript)) {
    Write-Fail "install-host-tailscale.ps1 not found at $tailscaleScript -- is this script being run from the repo root?"
    exit 1
}
& $tailscaleScript
if ($LASTEXITCODE -ne 0) {
    Write-Fail "install-host-tailscale.ps1 reported a failure -- fix that before continuing (see its output above)."
    exit 1
}

# Confirm Tailscale is actually logged in before continuing -- a WARN from
# the sub-script (not logged in yet) is a hard blocker for everything after
# this point (no Tailscale IP to bind the daemon to, no mesh to be part of).
$tailscaleExe = "C:\Program Files\Tailscale\tailscale.exe"
if (-not (Test-Path $tailscaleExe)) {
    $cmd = Get-Command tailscale -ErrorAction SilentlyContinue
    if ($cmd) { $tailscaleExe = $cmd.Source }
}
$tsStatus = & $tailscaleExe status 2>&1
if ($LASTEXITCODE -ne 0 -or $tsStatus -match "Logged out" -or $tsStatus -match "NeedsLogin") {
    Write-Host ""
    Write-Warn "Tailscale is installed but not logged in yet. Opening the login flow now -- complete it in your browser (this is the one step that genuinely cannot be automated)."
    & $tailscaleExe up
    Write-Host ""
    Write-Host "After logging in, re-run this exact command to continue:" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -ForegroundColor Yellow
    exit 2
}
$tailscaleIp = (& $tailscaleExe ip -4 2>&1).Trim()
Write-Ok "Tailscale connected, IP: $tailscaleIp"

# --- Step 3: WSL2-side Python/PyTorch/NCCL setup (delegates to install.sh) ---
Write-Step "Step 3/6: PyTorch + NCCL inside WSL2 (this is the slow step -- multi-GB download)"
# wslpath -a needs forward slashes in its input -- confirmed by testing:
# passing the raw Windows path (backslashes) silently mangles the
# separators instead of translating them (e.g. "L:\Likith\..." becomes
# "L:LikithCoding_Projects..." with backslashes just dropped, not
# converted), producing a broken path with no error. Converting to
# forward slashes first fixes it.
$repoRootForwardSlash = $RepoRoot -replace '\\', '/'
$wslRepoPathClean = (wsl -d $WslDistro -- wslpath -a "$repoRootForwardSlash" 2>&1) -join ""
Write-Host "Repo as seen from WSL2: $wslRepoPathClean"

wsl -d $WslDistro -- bash -lc "cd '$wslRepoPathClean' && bash install.sh"
if ($LASTEXITCODE -ne 0) {
    Write-Fail "install.sh reported a failure inside WSL2 -- see its output above. Fix and re-run this script (idempotent, will retry this step)."
    exit 1
}
Write-Ok "PyTorch/NCCL/venv setup complete inside WSL2."

# --- Step 4: dispatch token ---
Write-Step "Step 4/6: dispatch token"
$tokensDir = Join-Path $RepoRoot "coordinator\tokens"
if (-not (Test-Path $tokensDir)) { New-Item -ItemType Directory -Path $tokensDir | Out-Null }
$tokenPath = Join-Path $tokensDir "$WorkerName.secret"
if (Test-Path $tokenPath) {
    Write-Ok "Dispatch token already exists at $tokenPath (not overwriting -- delete it manually first if you want a fresh one)."
} else {
    $bytes = New-Object byte[] 32
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $token = -join ($bytes | ForEach-Object { $_.ToString("x2") })
    Set-Content -Path $tokenPath -Value $token -NoNewline -Encoding ascii
    Write-Ok "Generated dispatch token at $tokenPath."
}

# --- Step 5: worker_daemon.py auto-start wrapper + Scheduled Task ---
Write-Step "Step 5/6: auto-start on login"
# The daemon needs the machine's Tailscale IP, which is stable once
# registered but wasn't known at Scheduled-Task-authoring time on a truly
# fresh machine -- resolve it at LAUNCH time inside the wrapper script,
# not bake a possibly-stale IP into the task definition itself.
$startScriptPath = Join-Path $RepoRoot "coordinator\_start_worker_daemon.ps1"
$startScriptContent = @"
# Auto-generated by setup-worker.ps1 -- do not edit by hand, re-run
# setup-worker.ps1 to regenerate if paths/ports change.
`$tailscaleExe = "C:\Program Files\Tailscale\tailscale.exe"
if (-not (Test-Path `$tailscaleExe)) { `$tailscaleExe = "tailscale" }
`$ip = (& `$tailscaleExe ip -4 2>`$null | Select-Object -First 1).Trim()
if (-not `$ip) {
    Write-Host "[_start_worker_daemon] Tailscale IP not available yet, falling back to 127.0.0.1 (loopback-only, won't be reachable from other machines until Tailscale connects)."
    `$ip = "127.0.0.1"
}
Set-Location "$RepoRoot"
& python "$RepoRoot\coordinator\worker_daemon.py" --port $WorkerPort --host `$ip --token-file "$tokenPath"
"@
Set-Content -Path $startScriptPath -Value $startScriptContent -Encoding utf8
Write-Ok "Wrote auto-start wrapper: $startScriptPath"

$taskName = "MultiGPU-WorkerDaemon"
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "Removing existing '$taskName' scheduled task to re-register with current settings..."
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$startScriptPath`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0)  # 0 = no time limit, this is a long-running daemon

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "Auto-starts multigpu-setup's worker_daemon.py at login. Registered by setup-worker.ps1." | Out-Null
Write-Ok "Registered Scheduled Task '$taskName' (starts at login, restarts up to 3x on failure)."

# Start it now too -- don't make the user log off/on just to see it work.
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 3
$taskInfo = Get-ScheduledTaskInfo -TaskName $taskName
Write-Ok "Started now. Last run result: $($taskInfo.LastTaskResult) (0 = still running/launched OK)."

# --- Step 6: verify the daemon actually answers ---
Write-Step "Step 6/6: verifying worker_daemon.py is actually reachable"
Start-Sleep -Seconds 2
try {
    $resp = Invoke-WebRequest -Uri "http://$($tailscaleIp):$WorkerPort/status" -UseBasicParsing -TimeoutSec 5
    if ($resp.StatusCode -eq 200) {
        Write-Ok "worker_daemon.py answered /status over Tailscale ($tailscaleIp`:$WorkerPort)."
    } else {
        Write-Warn "worker_daemon.py responded with HTTP $($resp.StatusCode), expected 200."
    }
} catch {
    Write-Warn "Could not reach worker_daemon.py yet at $tailscaleIp`:$WorkerPort -- it may still be starting. Check with: Get-ScheduledTaskInfo -TaskName $taskName"
}

# --- Final summary ---
Write-Host ""
Write-Host "=== SUMMARY ===" -ForegroundColor Cyan
Write-Host "OK: $okCount   WARN: $warnCount   FAIL: $failCount"
Write-Host ""
if ($failCount -gt 0) {
    Write-Host "NOT READY -- fix the FAIL items above and re-run this script." -ForegroundColor Red
    exit 1
}
Write-Host "Add this to coordinator/workers.yaml on the COORDINATOR machine:" -ForegroundColor Green
Write-Host "  - name: $WorkerName" -ForegroundColor Green
Write-Host "    host: $tailscaleIp" -ForegroundColor Green
Write-Host "    port: $WorkerPort" -ForegroundColor Green
Write-Host "    token_file: coordinator/tokens/$WorkerName.secret" -ForegroundColor Green
Write-Host ""
Write-Host "Then copy coordinator/tokens/$WorkerName.secret from THIS machine to the coordinator machine (out-of-band, e.g. a direct file copy -- never over this project's own HTTP channel)." -ForegroundColor Green
Write-Host ""
Write-Host "worker_daemon.py will now auto-start on every login to this machine (Scheduled Task '$taskName')." -ForegroundColor Green
