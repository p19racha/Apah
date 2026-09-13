# Apah PowerShell One-Line Installer for Windows
# Usage: irm https://p19racha.github.io/Apah/install.ps1 | iex

$ErrorActionPreference = "Stop"

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "                 APAH INFERENCE RUNTIME INSTALLER                    " -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# 1. Detect Operating System and Python Version
Write-Host "[INFO] Checking Python 3 environment..." -ForegroundColor Green

$pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
}

if (-not $pythonCmd) {
    Write-Host "[ERROR] Python 3 is required but was not found in system PATH." -ForegroundColor Red
    Write-Host "[ERROR] Please install Python 3.10+ from https://www.python.org/ or via 'winget install Python.Python.3.11'." -ForegroundColor Red
    exit 1
}

$pythonVersionRaw = & $pythonCmd.Source -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
Write-Host "[INFO] Detected Python Version: $pythonVersionRaw" -ForegroundColor Green

$pythonMajor = [int](& $pythonCmd.Source -c "import sys; print(sys.version_info[0])")
$pythonMinor = [int](& $pythonCmd.Source -c "import sys; print(sys.version_info[1])")

if ($pythonMajor -lt 3 -or ($pythonMajor -eq 3 -and $pythonMinor -lt 10)) {
    Write-Host "[ERROR] Apah requires Python 3.10 or higher. Found $pythonVersionRaw." -ForegroundColor Red
    exit 1
}

# 2. Check NVIDIA GPU Acceleration
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    $gpuName = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
    Write-Host "[INFO] Detected NVIDIA Hardware Acceleration: $gpuName" -ForegroundColor Green
} else {
    Write-Host "[WARNING] nvidia-smi not detected in PATH. Running in CPU/Remote management mode." -ForegroundColor Yellow
}

# 3. Determine Installation Target Path
$defaultInstallDir = "$env:LOCALAPPDATA\Apah"
if ($env:APAH_INSTALL_DIR) {
    $installDir = $env:APAH_INSTALL_DIR
} else {
    $installDir = $defaultInstallDir
}

$venvDir = Join-Path $installDir "env"
$scriptsDir = Join-Path $venvDir "Scripts"
$apahExe = Join-Path $scriptsDir "apah.exe"

Write-Host "[INFO] Target installation directory: $installDir" -ForegroundColor Green

if (-not (Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
}

# 4. Create Virtual Environment and Install Apah
if (-not (Test-Path $venvDir)) {
    Write-Host "[INFO] Creating isolated Virtual Environment in $venvDir..." -ForegroundColor Green
    & $pythonCmd.Source -m venv $venvDir
}

$pipExe = Join-Path $scriptsDir "pip.exe"
Write-Host "[INFO] Upgrading pip and installing Apah package from GitHub..." -ForegroundColor Green

& $pipExe install --quiet --upgrade pip setuptools wheel
& $pipExe install --quiet --upgrade git+https://github.com/p19racha/Apah.git

if (-not (Test-Path $apahExe)) {
    Write-Host "[ERROR] Apah installation failed. Executable not found at $apahExe." -ForegroundColor Red
    exit 1
}

# 5. Add Scripts directory to User PATH environment variable if needed
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -notlike "*$scriptsDir*") {
    Write-Host "[INFO] Adding $scriptsDir to User PATH environment variable..." -ForegroundColor Green
    $newPath = "$userPath;$scriptsDir"
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    $env:PATH = "$env:PATH;$scriptsDir"
}

# 6. Success Output
Write-Host "`n[SUCCESS] Apah LLM Inference Runtime installed successfully!" -ForegroundColor Green
Write-Host "Executable location: $apahExe" -ForegroundColor Gray

Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "  1. Launch Apah Server:     apah serve" -ForegroundColor Green
Write-Host "  2. Pull a Model:           apah pull Qwen/Qwen2.5-0.5B-Instruct" -ForegroundColor Green
Write-Host "  3. Run Interactive Chat:   apah run Qwen/Qwen2.5-0.5B-Instruct" -ForegroundColor Green
Write-Host ""
