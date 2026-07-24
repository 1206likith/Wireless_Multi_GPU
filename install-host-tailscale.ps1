# Layer 1 (auto-discovery networking) setup script -- run on the WINDOWS
# HOST side (PowerShell), NOT inside WSL2.
#
# WHY TAILSCALE GOES ON THE HOST, NOT INSIDE WSL2 (verified via research
# before writing this script -- do not "simplify" this by moving Tailscale
# into WSL2 without re-reading):
# Tailscale's own official documentation explicitly recommends AGAINST
# running it inside WSL2: WSL2's default network interface has an MTU of
# 1280 bytes, too small to also carry a WireGuard-encapsulated Tailscale
# tunnel -- running Tailscale on both the host and inside WSL2
# simultaneously breaks ("Tailscale packets not fitting in Tailscale
# packets", per Tailscale's own KB page and a Tailscale engineer's GitHub
# comments, tailscale/tailscale#4833). The correct architecture is
# Tailscale on the Windows host only, combined with WSL2 "mirrored"
# networking mode so a Tailscale connection reaching the host also reaches
# whatever's listening inside WSL2.
#
# WHAT THIS SCRIPT DOES:
#   1. Checks if Tailscale is installed; if not, downloads and installs it
#      (requires the user to complete an interactive login the first time
#      -- Tailscale's auth flow opens a browser, this can't be fully
#      unattended for a first-time setup).
#   2. Verifies/sets up WSL2 mirrored networking mode in .wslconfig.
#   3. Applies the required Hyper-V firewall inbound-allow rule (needs
#      Administrator -- this script must be run elevated).
#   4. Restarts WSL2 to apply networking changes.
#   5. Reports the machine's Tailscale IP/hostname (what the OTHER laptop
#      will use to reach this one) and confirms mirrored mode took effect.
#
# MUST BE RUN AS ADMINISTRATOR (right-click PowerShell -> "Run as
# Administrator") -- the Hyper-V firewall step and .wslconfig changes both
# require elevated privileges.

$ErrorActionPreference = "Continue"
$okCount = 0
$warnCount = 0
$failCount = 0

function Write-Ok($msg)   { Write-Host "[OK]   $msg" -ForegroundColor Green;  $script:okCount++ }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow; $script:warnCount++ }
function Write-Fail($msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red;    $script:failCount++ }

Write-Host "=== Layer 1: Auto-discovery networking setup (Windows host) ===" -ForegroundColor Cyan
Write-Host ""

# --- Step 0: confirm running as Administrator ---
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Fail "Not running as Administrator. Right-click PowerShell and choose 'Run as Administrator', then re-run this script."
    exit 1
}
Write-Ok "Running as Administrator."

# --- Step 1: Tailscale install check ---
$tailscaleInstalled = Get-Command tailscale -ErrorAction SilentlyContinue
if (-not $tailscaleInstalled) {
    Write-Host "Tailscale not found. Downloading installer..."
    $installerPath = "$env:TEMP\tailscale-setup.exe"
    try {
        Invoke-WebRequest -Uri "https://pkgs.tailscale.com/stable/tailscale-setup-latest.exe" -OutFile $installerPath -UseBasicParsing
        Write-Host "Running installer (this may open a UAC prompt)..."
        Start-Process -FilePath $installerPath -ArgumentList "/quiet" -Wait
        Write-Ok "Tailscale installed."
    } catch {
        Write-Fail "Failed to download/install Tailscale: $_"
        Write-Host "Manual fallback: download from https://tailscale.com/download/windows and install manually, then re-run this script."
        exit 1
    }
} else {
    Write-Ok "Tailscale already installed."
}

# --- Step 2: Tailscale login/connect ---
$tsStatus = & tailscale status 2>&1
if ($LASTEXITCODE -ne 0 -or $tsStatus -match "Logged out") {
    Write-Warn "Not logged into Tailscale yet. Run 'tailscale up' manually and complete the browser login -- this can't be automated (needs your Tailscale/Google/GitHub account)."
    Write-Host "  After running 'tailscale up' and logging in, re-run this script to continue setup."
} else {
    Write-Ok "Tailscale is connected."
    $tsIP = (& tailscale ip -4 2>&1)
    Write-Host "  This machine's Tailscale IP: $tsIP"
    Write-Host "  (The other laptop will use this IP to reach this machine.)"
}

# --- Step 3: WSL2 mirrored networking mode ---
$wslConfigPath = "$env:USERPROFILE\.wslconfig"
$mirroredAlreadySet = $false
if (Test-Path $wslConfigPath) {
    $content = Get-Content $wslConfigPath -Raw
    if ($content -match "networkingMode\s*=\s*mirrored") {
        $mirroredAlreadySet = $true
    }
}

if ($mirroredAlreadySet) {
    Write-Ok ".wslconfig already has networkingMode=mirrored."
} else {
    Write-Host "Adding networkingMode=mirrored to $wslConfigPath..."
    if (Test-Path $wslConfigPath) {
        $existing = Get-Content $wslConfigPath -Raw
        if ($existing -match "\[wsl2\]") {
            # Insert the setting into the existing [wsl2] section rather than
            # blindly overwriting the file -- preserves any other settings
            # (e.g. vmIdleTimeout) already present.
            $updated = $existing -replace "(\[wsl2\])", "`$1`nnetworkingMode=mirrored"
            Set-Content -Path $wslConfigPath -Value $updated -Encoding utf8
        } else {
            Add-Content -Path $wslConfigPath -Value "`n[wsl2]`nnetworkingMode=mirrored" -Encoding utf8
        }
    } else {
        Set-Content -Path $wslConfigPath -Value "[wsl2]`nnetworkingMode=mirrored" -Encoding utf8
    }
    Write-Ok ".wslconfig updated."
}

# --- Step 4: Hyper-V firewall rule (required for mirrored mode inbound) ---
try {
    Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow -ErrorAction Stop
    Write-Ok "Hyper-V firewall inbound rule applied."
} catch {
    Write-Fail "Could not apply Hyper-V firewall rule: $_"
}

# --- Step 5: restart WSL2 to apply changes ---
Write-Host "Restarting WSL2 to apply networking changes..."
wsl --shutdown
Start-Sleep -Seconds 3
Write-Ok "WSL2 restarted."

# --- Final summary ---
Write-Host ""
Write-Host "=== SUMMARY ===" -ForegroundColor Cyan
Write-Host "OK: $okCount   WARN: $warnCount   FAIL: $failCount"
Write-Host ""
if ($failCount -gt 0) {
    Write-Host "NOT READY -- fix the FAIL items above." -ForegroundColor Red
    exit 1
}
if ($warnCount -gt 0) {
    Write-Host "PARTIALLY READY -- address the WARN items (usually: log into Tailscale), then re-run this script." -ForegroundColor Yellow
} else {
    Write-Host "Networking layer ready. Verify from inside WSL2 with:" -ForegroundColor Green
    Write-Host "  wsl -d Ubuntu-24.04 -- hostname -I"
    Write-Host "It should show this machine's real LAN IP (mirrored mode), not a separate 172.x private WSL2 IP."
}
