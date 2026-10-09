param(
    [int]$Envs = 2048,
    [int]$Ticks = 60,
    [int]$Seed = 1337
)

$scenarios = @(
    "ablation_1_discrete_throttle_steer",
    "ablation_2_analog_throttle_steer",
    "ablation_3_plus_boost",
    "ablation_4_plus_handbrake",
    "ablation_5_plus_single_jump",
    "ablation_6_plus_double_jump",
    "ablation_7_plus_dodge_flip",
    "ablation_8_plus_air_control",
    "ablation_9_plus_landings",
    "ablation_10_random_full"
)

if (-not (Test-Path "logs\verify")) {
    New-Item -ItemType Directory -Force -Path "logs\verify" | Out-Null
}

$results = @()

for ($idx = 0; $idx -lt $scenarios.Count; $idx++) {
    $num = $idx + 1
    $scn = $scenarios[$idx]
    $logFile = "logs\verify\ablation_${num}.txt"
    Write-Host "[Ablation $num/10] Running $scn..."

    & powershell -ExecutionPolicy Bypass -File scripts\run_harness.ps1 --scenario $scn --envs $Envs --ticks $Ticks --seed $Seed --baseline > $logFile 2>&1
    
    # Parse logFile for Tick 60 results
    $content = Get-Content $logFile -Raw
    
    # Extract Tick 60 table block
    # Looking for:
    # **Tick 60** (2048 environments):
    # | Component | Median Abs Error | P95 Abs Error | ...
    
    Write-Host "    Completed $scn -> saved to $logFile"
}

Write-Host "All 10 ablation scenarios executed."
