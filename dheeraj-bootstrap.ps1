# Bootstrap script for dheeraj's laptop -- clones the multigpu-setup repo
# fresh from GitHub (so you get the current, up-to-date code, not a
# possibly-stale USB copy) and then runs the real one-shot setup script.
#
# WHY THIS CLONES INSTEAD OF JUST COPYING FILES FROM THE USB: the repo is
# public on GitHub and actively being updated -- cloning gets you the
# latest version every time you run this, and gives you a real git
# checkout you (or Likith) can `git pull` from later without needing the
# USB again. The USB is just how this bootstrap script + the manual
# physically get to your machine the first time.
#
# USAGE: right-click this file -> "Run with PowerShell", or open
# PowerShell AS ADMINISTRATOR and run:
#   powershell -ExecutionPolicy Bypass -File dheeraj-bootstrap.ps1
#
# See USER_MANUAL.md on this USB drive for the full walkthrough with
# screenshots-in-words and troubleshooting -- read that first if anything
# here is unclear.

param(
    [string]$InstallPath = "C:\multigpu-setup"
)

$ErrorActionPreference = "Continue"
$RepoUrl = "https://github.com/1206likith/Wireless_Multi_GPU.git"

Write-Host "=== Multi-GPU cluster: dheeraj's laptop bootstrap ===" -ForegroundColor Cyan
Write-Host ""

# --- confirm Administrator (setup-worker.ps1 needs this; check early with
#     a clear message rather than letting it fail confusingly later) ---
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[FAIL] Not running as Administrator." -ForegroundColor Red
    Write-Host ""
    Write-Host "Right-click PowerShell in the Start menu, choose 'Run as Administrator', then run:" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] Running as Administrator." -ForegroundColor Green

# --- confirm git is available ---
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
    Write-Host "[FAIL] git is not installed on this machine." -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Git for Windows first: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "(default install options are fine), then re-run this script." -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] git found: $($gitCmd.Source)" -ForegroundColor Green

# --- clone or update the repo ---
if (Test-Path "$InstallPath\.git") {
    Write-Host "[OK] $InstallPath already has a git checkout -- pulling latest instead of cloning fresh." -ForegroundColor Green
    Push-Location $InstallPath
    git pull origin master
    Pop-Location
} elseif (Test-Path $InstallPath) {
    Write-Host "[FAIL] $InstallPath already exists but isn't a git checkout." -ForegroundColor Red
    Write-Host "Delete or rename it, or pass a different path: -InstallPath 'C:\some\other\path'" -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "Cloning $RepoUrl to $InstallPath ..."
    git clone $RepoUrl $InstallPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] git clone failed -- check your internet connection and try again." -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Cloned to $InstallPath" -ForegroundColor Green
}

# --- hand off to the real setup script ---
Write-Host ""
Write-Host "=== Handing off to setup-worker.ps1 ===" -ForegroundColor Cyan
Write-Host "This is the real setup -- it installs WSL2, Tailscale, PyTorch/NCCL," -ForegroundColor Cyan
Write-Host "generates your machine's security token, and makes the worker" -ForegroundColor Cyan
Write-Host "daemon start automatically every time you log in." -ForegroundColor Cyan
Write-Host ""
Set-Location $InstallPath
& "$InstallPath\setup-worker.ps1"

Write-Host ""
Write-Host "If setup-worker.ps1 told you to REBOOT, do that now, then run this" -ForegroundColor Yellow
Write-Host "exact same bootstrap command again (it will skip the clone step and" -ForegroundColor Yellow
Write-Host "go straight back into setup-worker.ps1, which picks up where it left off)." -ForegroundColor Yellow
