param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$HarnessArgs
)

$cudaCandidates = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
    "C:\Users\Choosen\CUDA_Toolkit"
)
foreach ($cand in $cudaCandidates) {
    if (Test-Path $cand) {
        $env:CUDA_PATH = $cand
        $env:PATH = "$cand\bin;" + $env:PATH
        break
    }
}

$exe = "build\differential_harness.exe"
if (-not (Test-Path $exe)) {
    Write-Error "differential_harness.exe not found at $exe"
    exit 1
}

& $exe @HarnessArgs
exit $LASTEXITCODE
