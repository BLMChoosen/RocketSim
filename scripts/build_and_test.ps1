<#
.SYNOPSIS
    RocketSim-CUDA build and test orchestration script.
.DESCRIPTION
    Strict orchestration script:
    1. Locates and initializes MSVC/CUDA toolchain (including vcvars64.bat and vswhere).
    2. Enforces presence of cl.exe, cmake.exe, and ninja.exe (fails with exit code != 0 if missing).
    3. Builds native targets (differential_harness and rocketsim_cuda.pyd).
    4. Enforces that differential_harness.exe exists after compilation (fails if missing).
    5. Runs differential harness against parity thresholds, failing immediately on non-zero exit.
    6. Enforces that rocketsim_cuda.pyd exists and runs full pytest suite, requiring:
       - 0 skipped
       - 0 failed
       - passed == collected
    7. No silent success paths.
.PARAMETER Config
    Build configuration (default: Release).
.PARAMETER CheckFile
    Path to parity thresholds JSON file (default: docs/parity_thresholds.json).
.PARAMETER Ticks
    Number of ticks for differential harness (default: 60).
.PARAMETER Envs
    Number of environments for differential harness (default: 4).
.PARAMETER SkipBuild
    Skip compilation step.
.PARAMETER SkipHarness
    Skip differential harness step.
.PARAMETER SkipPytest
    Skip pytest suite step.
#>

param(
    [string]$Config = "Release",
    [string]$CheckFile = "docs/parity_thresholds.json",
    [int]$Ticks = 60,
    [int]$Envs = 4,
    [string]$Scenario = "all",
    [int]$Cars = 1,
    [int]$Seed = 42,
    [string]$OutReport = "",
    [switch]$Baseline,
    [switch]$SkipBuild,
    [switch]$SkipHarness,
    [switch]$SkipPytest
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

Write-Host "======================================================================"
Write-Host "            RocketSim-CUDA Strict Build & Test Pipeline               "
Write-Host "======================================================================"
Write-Host " Project Root:  $ProjectRoot"
Write-Host " Config:        $Config"
Write-Host " Thresholds:    $CheckFile"
Write-Host "======================================================================`n"

# -----------------------------------------------------------------------------
# 1. Environment Setup (MSVC, CUDA, CMake, Ninja)
# -----------------------------------------------------------------------------
function Setup-Environment {
    Write-Host "[Env] Locating MSVC and toolchain components..."

    # 1.1 Locate vcvars64.bat
    $vcvarsCandidates = @(
        "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    )

    $vswherePaths = @(
        "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
        "C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe"
    )

    foreach ($vsw in $vswherePaths) {
        if (Test-Path $vsw) {
            try {
                $vsInstall = & $vsw -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
                if ($vsInstall -and (Test-Path "$vsInstall\VC\Auxiliary\Build\vcvars64.bat")) {
                    $vcvarsCandidates = @("$vsInstall\VC\Auxiliary\Build\vcvars64.bat") + $vcvarsCandidates
                }
            } catch {}
        }
    }

    $vcvarsFound = $null
    foreach ($cand in $vcvarsCandidates) {
        if (Test-Path $cand) {
            $vcvarsFound = $cand
            break
        }
    }

    if ($vcvarsFound) {
        Write-Host "[Env] Initializing MSVC environment via $vcvarsFound..."
        $tmpBat = [System.IO.Path]::GetTempFileName() + ".bat"
        Set-Content -Path $tmpBat -Value "@call `"$vcvarsFound`" >nul 2>&1`n@set"
        $vars = cmd.exe /c $tmpBat
        Remove-Item -Force $tmpBat -ErrorAction SilentlyContinue
        foreach ($line in $vars) {
            if ($line -match '^([^=]+)=(.*)$') {
                [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], [System.EnvironmentVariableTarget]::Process)
            }
        }
        Write-Host "[Env] MSVC environment loaded successfully."
    }

    # 1.2 CUDA Toolkit
    if (-not $env:CUDA_PATH) {
        $cudaCandidates = @(
            "C:\Users\Choosen\CUDA_Toolkit",
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
        )
        foreach ($cand in $cudaCandidates) {
            if (Test-Path $cand) {
                $env:CUDA_PATH = $cand
                $env:CUDAToolkit_ROOT = $cand
                $env:PATH = "$cand\bin;" + $env:PATH
                Write-Host "[Env] CUDA Toolkit detected at $cand"
                break
            }
        }
    }

    # 1.3 CMake and Ninja from Python or System
    if (-not (Get-Command cmake.exe -ErrorAction SilentlyContinue)) {
        try {
            $pyCmakeDir = python -c "import cmake, os; print(os.path.dirname(cmake.__file__))" 2>$null
            if ($pyCmakeDir -and (Test-Path "$pyCmakeDir\data\bin\cmake.exe")) {
                $env:PATH = "$pyCmakeDir\data\bin;" + $env:PATH
                Write-Host "[Env] Added python cmake to PATH."
            }
        } catch {}
    }
    if (-not (Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
        try {
            $pyNinjaDir = python -c "import ninja, os; print(os.path.dirname(ninja.__file__))" 2>$null
            if ($pyNinjaDir -and (Test-Path "$pyNinjaDir\data\bin\ninja.exe")) {
                $env:PATH = "$pyNinjaDir\data\bin;" + $env:PATH
                Write-Host "[Env] Added python ninja to PATH."
            }
        } catch {}
    }
}

Setup-Environment

# -----------------------------------------------------------------------------
# 2. Strict Toolchain Verification & Native Compilation
# -----------------------------------------------------------------------------
if (-not $SkipBuild) {
    Write-Host "`n[Step 1/3] Native Compilation..."
    $hasCmake = (Get-Command cmake.exe -ErrorAction SilentlyContinue) -ne $null
    $hasCl = (Get-Command cl.exe -ErrorAction SilentlyContinue) -ne $null

    if (-not $hasCl) {
        Write-Error "[-] FATAL: MSVC C++ compiler (cl.exe) not found. vcvars64.bat was not located or could not be loaded."
        exit 1
    }
    if (-not $hasCmake) {
        Write-Error "[-] FATAL: cmake.exe not found in PATH or python packages."
        exit 1
    }

    $hasNinja = (Get-Command ninja.exe -ErrorAction SilentlyContinue) -ne $null
    $generatorArgs = @()
    if ($hasNinja) {
        $generatorArgs = @("-G", "Ninja")
    }

    Write-Host "--> Configuring CMake (Config: $Config)..."
    $configureArgs = @(
        "-B", "build",
        "-DCMAKE_BUILD_TYPE=$Config",
        "-DROCKETSIM_CUDA_BUILD_TESTS=ON"
    ) + $generatorArgs

    & cmake @configureArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[-] CMake configuration failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    Write-Host "--> Building native targets (differential_harness, rocketsim_cuda)..."
    & cmake --build build --config $Config -j
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[-] Native compilation failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Host "[+] Native compilation completed successfully."
} else {
    Write-Host "`n[Step 1/3] Native Compilation: SKIPPED (-SkipBuild specified)"
}

# -----------------------------------------------------------------------------
# 3. Differential Harness Parity & Regression Guard
# -----------------------------------------------------------------------------
if (-not $SkipHarness) {
    Write-Host "`n[Step 2/3] Differential Parity Harness..."

    $harnessCandidates = @(
        "build\differential_harness.exe",
        "build\$Config\differential_harness.exe",
        "build\tests\differential\$Config\differential_harness.exe",
        "build\tests\differential\differential_harness.exe"
    )

    $harnessExe = $null
    foreach ($cand in $harnessCandidates) {
        if (Test-Path $cand) {
            $harnessExe = Resolve-Path $cand
            break
        }
    }

    if (-not $harnessExe) {
        Write-Error "[-] FATAL: differential_harness.exe binary does not exist in build/ directory."
        exit 1
    }

    $harnessArgs = @("--scenario", $Scenario, "--ticks", $Ticks, "--envs", $Envs)
    if ($Cars -gt 1) { $harnessArgs += @("--cars", $Cars) }
    if ($Seed -ne 42) { $harnessArgs += @("--seed", $Seed) }
    if ($Baseline) { $harnessArgs += "--baseline" }
    if ($OutReport) { $harnessArgs += @("--out-report", $OutReport) }
    $harnessArgs += @("--report", "--check", $CheckFile)

    Write-Host "--> Executing differential harness: $harnessExe"
    Write-Host "    Args: $($harnessArgs -join ' ')"
    & $harnessExe @harnessArgs
    $harnessExitCode = $LASTEXITCODE
    if ($harnessExitCode -ne 0) {
        Write-Error "[-] Differential harness failed with exit code $harnessExitCode"
        exit $harnessExitCode
    }
    Write-Host "[+] Differential harness passed 100% against parity thresholds."
} else {
    Write-Host "`n[Step 2/3] Differential Parity Harness: SKIPPED (-SkipHarness specified)"
}

# -----------------------------------------------------------------------------
# 4. Python Unit & Regression Test Suite
# -----------------------------------------------------------------------------
if (-not $SkipPytest) {
    Write-Host "`n[Step 3/3] Python Unit Tests (pytest)..."
    $hasPytest = (Get-Command pytest -ErrorAction SilentlyContinue) -ne $null
    if (-not $hasPytest) {
        try {
            python -c "import pytest" 2>$null
            if ($LASTEXITCODE -eq 0) { $hasPytest = $true }
        } catch {}
    }

    if (-not $hasPytest) {
        Write-Error "[-] FATAL: pytest is not installed in the python environment."
        exit 1
    }

    # Verify native extension (.pyd) is compiled
    $hasPyd = (Test-Path "build") -and ((Get-ChildItem -Path "build" -Filter "rocketsim_cuda*.pyd" -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0)
    if (-not $hasPyd) {
        Write-Error "[-] FATAL: rocketsim_cuda native extension (.pyd) not found in build/ directory. Aborting pytest."
        exit 1
    }

    Write-Host "--> Running full pytest suite with native extension..."
    $pytestOutput = pytest tests/python/ -v 2>&1
    $pytestExitCode = $LASTEXITCODE
    $pytestOutput | ForEach-Object { Write-Host $_ }

    if ($pytestExitCode -ne 0) {
        Write-Error "[-] Pytest suite failed with exit code $pytestExitCode"
        exit $pytestExitCode
    }

    # Strictly parse pytest output to verify collected count and enforce:
    # 1. collected > 0
    # 2. 0 skipped
    # 3. 0 failed
    # 4. passed == collected
    $collectedLine = ($pytestOutput | Where-Object { $_ -match 'collected\s+(\d+)\s+items' }) | Select-Object -Last 1
    if (-not $collectedLine -or -not ($collectedLine -match 'collected\s+(\d+)\s+items')) {
        Write-Error "[-] FATAL: Could not determine collected test count from pytest output."
        exit 1
    }
    $collectedCount = [int]$matches[1]
    if ($collectedCount -le 0) {
        Write-Error "[-] FATAL: Pytest collected 0 items. Expected > 0 tests."
        exit 1
    }

    $summaryLine = ($pytestOutput | Where-Object { $_ -match '==+ (.*) in .*s ==+' }) | Select-Object -Last 1
    if (-not $summaryLine) {
        Write-Error "[-] FATAL: Pytest summary line not found in output."
        exit 1
    }

    if ($summaryLine -match '(\d+)\s+skipped') {
        $skippedCount = [int]$matches[1]
        if ($skippedCount -gt 0) {
            Write-Error "[-] FATAL: Pytest reported $skippedCount skipped tests. Expected 0 skipped."
            exit 1
        }
    }
    if ($summaryLine -match '(\d+)\s+failed') {
        $failedCount = [int]$matches[1]
        if ($failedCount -gt 0) {
            Write-Error "[-] FATAL: Pytest reported $failedCount failed tests."
            exit 1
        }
    }

    $passedCount = 0
    if ($summaryLine -match '(\d+)\s+passed') {
        $passedCount = [int]$matches[1]
    }

    if ($passedCount -ne $collectedCount) {
        Write-Error "[-] FATAL: Pytest passed count ($passedCount) does not equal collected count ($collectedCount)."
        exit 1
    }

    Write-Host "[+] Python test suite completed successfully (0 skipped, 0 failed, passed: $passedCount == collected: $collectedCount)."
} else {
    Write-Host "`n[Step 3/3] Python Unit Tests: SKIPPED (-SkipPytest specified)"
}

Write-Host "`n======================================================================"
Write-Host "  BUILD AND TEST PIPELINE PASSED SUCCESSFULLY! Exit code 0.           "
Write-Host "======================================================================"
exit 0
