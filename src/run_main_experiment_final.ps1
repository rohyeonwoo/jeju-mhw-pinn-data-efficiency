param(
    [string]$RootDir = $PSScriptRoot,
    [string]$EnvName = "mhw_win",
    [string]$Gpus = "0",
    [string]$Seeds = "42,7,123",
    [string]$Leads = "1,3,7,14",
    [int]$History = 30,
    [string]$FinalPhysics = "P0",
    [int]$BatchSize = 8,
    [int]$ConvLSTMEpochs = 40,
    [int]$ConvLSTMPatience = 8,
    [string]$HiddenChannels = "32,32",
    [double]$ConvLSTMLr = 0.001,
    [int]$HiLAMEpochs = 80,
    [int]$HiLAMPatience = 12,
    [double]$HiLAMlr = 0.0002,
    [double]$HiLAMInitialLr = 0.00002,
    [double]$HiLAMWeightDecay = 0.1,
    [int]$HiLAMHiddenDim = 256,
    [int]$HiLAMTemporalHiddenDim = 256,
    [int]$HiLAMProcessorLayers = 3,
    [double]$LambdaPhysics = 0.1,
    [double]$LambdaRres = 10.0,
    [double]$LambdaWeTie = 0.0,
    [double]$KDiff = 100.0,
    [string]$Device = "cuda",
    [string]$HiLAMDevice = "auto",
    [string]$DataPath = "",
    [string]$OutputRoot = "",
    [bool]$RunSummary = $true,
    [bool]$ReuseAblationResults = $true,
    [string]$AblationRoot = "",
    [switch]$PreflightOnly,
    [switch]$SmokeOnly,
    [switch]$Force,
    [switch]$SkipConvLSTM,
    [switch]$SkipConvLSTMPINN,
    [switch]$SkipHiLAM,
    [switch]$SkipPIHiLAM,
    [switch]$PauseOnExit
)

# Final main experiment.
# FINAL_PHYSICS defaults to P0. Residual feature-map input is always disabled.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Split-List([string]$Text) {
    return @($Text -replace ",", " " -split "\s+" | Where-Object { $_.Trim().Length -gt 0 })
}

function Physics-Mode-For-HiLAM([string]$Preset) {
    switch ($Preset.ToUpperInvariant()) {
        "P0" { return "p0_original" }
        "P1" { return "p1_vertical" }
        default { throw "JejuHiLAM PINN runner currently supports P0/P1 only; requested $Preset" }
    }
}

function Invoke-PythonLogged([object[]]$Arguments, [string]$LogPath, [string]$FailureMessage) {
    $logDir = Split-Path -Parent $LogPath
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    Push-Location $RootDir
    try {
        conda run -n $EnvName --no-capture-output python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
        if ($LASTEXITCODE -ne 0) {
            throw $FailureMessage
        }
    }
    finally {
        Pop-Location
    }
}

function Test-MetricDone([string]$Path) {
    return ((Test-Path -LiteralPath $Path) -and (-not $Force))
}

$RootDir = [System.IO.Path]::GetFullPath($RootDir)
if ([string]::IsNullOrWhiteSpace($DataPath)) {
    $DataPath = "D:\MHW\data\merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d_budget2d.nc"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = "D:\MHW\outputs\main_experiment_final"
}
if ([string]::IsNullOrWhiteSpace($AblationRoot)) {
    $AblationRoot = "D:\MHW\outputs\physics_ablation_final"
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$AblationRoot = [System.IO.Path]::GetFullPath($AblationRoot)
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$FinalPhysics = $FinalPhysics.ToUpperInvariant()
if ($FinalPhysics -ne "P0") {
    throw "This final main experiment is locked to FINAL_PHYSICS=P0. Requested: $FinalPhysics"
}

$SeedList = @(Split-List $Seeds | ForEach-Object { [int]$_ })
$LeadList = @(Split-List $Leads | ForEach-Object { [int]$_ })
$DynamicVars = "sst_mean,sst_anom,u_ml,v_ml,mlotst,ssr,str,slhf,sshf,mslp,tcc,Td_glorys,wind10,thetao_z0m_anom,thetao_z16m_anom,thetao_z34m_anom,thetao_z78m_anom,thetao_z110m_anom,thetao_z156m_anom"
$env:CUDA_VISIBLE_DEVICES = $Gpus
$ActiveOutputRoot = $OutputRoot
$RunLeadList = $LeadList
$RunSeedList = $SeedList
if ($SmokeOnly) {
    $ActiveOutputRoot = Join-Path $OutputRoot "_smoke"
    New-Item -ItemType Directory -Force -Path $ActiveOutputRoot | Out-Null
    $RunLeadList = @($LeadList[0])
    $RunSeedList = @($SeedList[0])
}

Write-Host "=== MAIN EXPERIMENT FINAL ===" -ForegroundColor Green
Write-Host "RootDir        : $RootDir"
Write-Host "DataPath       : $DataPath"
Write-Host "OutputRoot     : $OutputRoot"
if ($SmokeOnly) { Write-Host "SmokeOutput    : $ActiveOutputRoot" }
Write-Host "FINAL_PHYSICS  : $FinalPhysics"
Write-Host "Leads          : $($LeadList -join ', ')"
Write-Host "Seeds          : $($SeedList -join ', ')"
Write-Host "Residual fmap  : OFF"
Write-Host "Reuse ablation : $ReuseAblationResults (root: $AblationRoot)"

$GitCommit = $null
try {
    $GitCommit = (git -C $RootDir rev-parse HEAD 2>$null)
} catch {
    $GitCommit = $null
}
if ([string]::IsNullOrWhiteSpace($GitCommit)) { $GitCommit = "unknown (git rev-parse failed)" }
$RunTimestampUtc = (Get-Date).ToUniversalTime().ToString("o")
$InvokedCommand = "$($MyInvocation.MyCommand.Path) " + (
    ($PSBoundParameters.GetEnumerator() | ForEach-Object {
        if ($_.Value -is [switch]) { "-$($_.Key):`$$($_.Value.IsPresent)" }
        else { "-$($_.Key) `"$($_.Value)`"" }
    }) -join " "
)

$preflightLog = Join-Path $OutputRoot "preflight_main_experiment_final.log"
$preflightJson = Join-Path $OutputRoot "preflight_main_experiment_final.json"
$preflightArgs = @(
    "preflight_main_experiment_final.py",
    "--data-path", $DataPath,
    "--history", $History,
    "--leads", ($LeadList -join ","),
    "--seeds", ($SeedList -join ","),
    "--final-physics", $FinalPhysics,
    "--expected-convlstm-channels", 22,
    "--expected-hilam-temporal-channels", 19,
    "--lambda-physics", $LambdaPhysics,
    "--lambda-rres", $LambdaRres,
    "--k-diff", $KDiff,
    "--hm-min", 5.0,
    "--forcing-mode", "perfect",
    "--output-json", $preflightJson
)
Invoke-PythonLogged $preflightArgs $preflightLog "preflight failed"

$manifest = [ordered]@{
    experiment_set = "main_experiment_final"
    final_physics = $FinalPhysics
    final_physics_reason = "P1 degraded too strongly at lead14 in final P0-P3 ablation; P0 is selected for main experiments."
    output_root = $OutputRoot
    data_path = $DataPath
    target = "sst_mean"
    split = "chronological 70/15/15"
    history_days = $History
    leads = $LeadList
    seeds = $SeedList
    dynamic_vars = $DynamicVars
    convlstm_static_vars = "ocean_mask_static,lat2d,lon2d"
    residual_feature_map = "off"
    reuse_old_results = $ReuseAblationResults
    convlstm_input_channels = 22
    hilam_temporal_input_channels = 19
    lambda_physics = $LambdaPhysics
    lambda_rres = $LambdaRres
    zero_rres_physics = $true
    rhs_normalization = "train-fixed"
    physics_time_mode = "daily_rollout"
    git_commit = $GitCommit
    run_timestamp_utc = $RunTimestampUtc
    invoked_command = $InvokedCommand
    ablation_root = $AblationRoot
    reused_runs = @()
}
$manifestPath = Join-Path $OutputRoot "experiment_manifest_final.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-Host "saved manifest: $manifestPath"

if ($PreflightOnly) {
    Write-Host "PreflightOnly requested; no training launched." -ForegroundColor Yellow
    if ($PauseOnExit) { Read-Host "Press Enter to close/return" | Out-Null }
    return
}

$physicsMode = Physics-Mode-For-HiLAM $FinalPhysics
$script:ReusedRuns = @()

function Try-ReuseAblationRun([string]$Family, [int]$Lead, [int]$Seed, [string]$OutDir, [string]$MetricsPath) {
    # Only ConvLSTM-PINN/P0/lead7-14 overlaps with the completed physics_ablation_final
    # runs (that experiment never trained ConvLSTM baseline or any HiLAM variant).
    if (-not $ReuseAblationResults) { return $false }
    if ($SmokeOnly) { return $false }
    if ($Family -ne "convlstm_pinn") { return $false }
    if ($FinalPhysics -ne "P0") { return $false }
    if ($Lead -notin @(7, 14)) { return $false }

    $srcDir = Join-Path (Join-Path (Join-Path $AblationRoot "seed$Seed") "lead$Lead") "p0_original"
    $srcMetricsPath = Join-Path $srcDir ("hybrid_mt_pinn_lead_{0:D2}_history_{1:D2}_metrics.json" -f $Lead, $History)
    if (-not (Test-Path -LiteralPath $srcMetricsPath)) { return $false }

    $src = Get-Content -Raw -LiteralPath $srcMetricsPath | ConvertFrom-Json
    $expectedDynamicVars = @($DynamicVars -split ",")
    $srcDynamicVars = @($src.dynamic_vars)
    # Order-sensitive: variable order maps to fixed input channel indices.
    $dynamicVarsMatch = ($expectedDynamicVars -join ",") -eq ($srcDynamicVars -join ",")

    $checks = [ordered]@{
        model_name               = ($src.model_name -eq "hybrid_mt_pinn")
        physics_preset           = ($src.physics_preset -eq "P0")
        physics_components       = ($src.physics_components -eq "original")
        disable_residual_map     = ([bool]$src.disable_residual_map -eq $true)
        residual_feature_map_off = ([bool]$src.residual_feature_map_enabled -eq $false)
        history_days             = ([int]$src.history_days -eq $History)
        lead_days                = ([int]$src.lead_days -eq $Lead)
        dynamic_vars             = $dynamicVarsMatch
        lambda_physics           = ([double]$src.lambda_physics -eq [double]$LambdaPhysics)
        lambda_rres              = ([double]$src.lambda_rres -eq [double]$LambdaRres)
        k_diff                   = ([double]$src.k_diff -eq [double]$KDiff)
        hm_min                   = ([double]$src.hm_min -eq 5.0)
        zero_rres_physics        = ([bool]$src.zero_rres_physics -eq $true)
        physics_normalize        = ($src.physics_normalize -eq "rhs")
        physics_time_mode        = ($src.physics_time_mode -eq "daily_rollout")
        forcing_mode             = ($src.forcing_mode -eq "perfect")
        forecast_mode            = ($src.forecast_mode -eq "free")
        entrainment_dt           = ([double]$src.entrainment_dt -eq 1.0)
        we_nonneg                = ([bool]$src.we_nonneg -eq $true)
        we_softplus_shift        = ([double]$src.we_softplus_shift -eq 1.0)
        physics_clip             = ([double]$src.physics_clip -eq 10.0)
        physics_huber_beta       = ([double]$src.physics_huber_beta -eq 0.25)
        rres_penalty_normalize   = ($src.rres_penalty_normalize -eq "rhs")
        selection_metric         = ($src.selection_metric -eq "rmse")
        model_input_channels     = ([int]$src.model_input_channels -eq 22)
    }
    $failed = $checks.GetEnumerator() | Where-Object { -not $_.Value } | ForEach-Object { $_.Key }
    if ($failed) {
        Write-Warning ("Ablation reuse skipped for {0} lead={1} seed={2} (mismatch: {3}); training from scratch." -f $Family, $Lead, $Seed, ($failed -join ", "))
        return $false
    }

    Write-Host "REUSE from ablation: $srcMetricsPath" -ForegroundColor Magenta
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    Get-ChildItem -LiteralPath $srcDir -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $OutDir $_.Name) -Force
    }

    $destMetrics = Get-Content -Raw -LiteralPath $srcMetricsPath | ConvertFrom-Json
    if ($destMetrics.model_path) { $destMetrics.model_path = Join-Path $OutDir (Split-Path -Leaf $destMetrics.model_path) }
    if ($destMetrics.test_fields_npz) { $destMetrics.test_fields_npz = Join-Path $OutDir (Split-Path -Leaf $destMetrics.test_fields_npz) }
    if ($destMetrics.epoch_log_csv) { $destMetrics.epoch_log_csv = Join-Path $OutDir (Split-Path -Leaf $destMetrics.epoch_log_csv) }
    $destMetrics | Add-Member -NotePropertyName "reused" -NotePropertyValue $true -Force
    $destMetrics | Add-Member -NotePropertyName "reused_from_ablation" -NotePropertyValue $srcMetricsPath -Force
    $destMetrics | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $MetricsPath -Encoding UTF8

    $script:ReusedRuns += [ordered]@{
        family = $Family
        lead = $Lead
        seed = $Seed
        source_metrics_path = $srcMetricsPath
        dest_metrics_path = $MetricsPath
    }
    return $true
}

function Invoke-ConvLSTMRun([string]$Family, [int]$Lead, [int]$Seed, [bool]$UsePhysics) {
    $modelName = if ($UsePhysics) { "hybrid_mt_pinn" } else { "hybrid_learned" }
    $familyRoot = Join-Path $ActiveOutputRoot $Family
    $outDir = Join-Path (Join-Path $familyRoot "lead$Lead") "seed$Seed"
    $metricsPath = Join-Path $outDir ("{0}_lead_{1:D2}_history_{2:D2}_metrics.json" -f $modelName, $Lead, $History)
    if (Test-MetricDone $metricsPath) {
        Write-Host "SKIP existing: $metricsPath" -ForegroundColor DarkYellow
        return
    }
    if (Try-ReuseAblationRun $Family $Lead $Seed $outDir $metricsPath) {
        return
    }
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $env:PYTHONHASHSEED = [string]$Seed
    $logPath = Join-Path $outDir "train_final.log"
    Write-Host "=== $Family lead=$Lead seed=$Seed ===" -ForegroundColor Cyan

    $runArgs = @(
        "train_hybrid_jeju_forecast.py",
        "--data-path", $DataPath,
        "--disable-residual-map",
        "--model", $modelName,
        "--dynamic-vars", $DynamicVars,
        "--history", $History,
        "--horizon", $Lead,
        "--seed", $Seed,
        "--batch-size", $BatchSize,
        "--epochs", $ConvLSTMEpochs,
        "--patience", $ConvLSTMPatience,
        "--lr", $ConvLSTMLr,
        "--hidden-channels", $HiddenChannels,
        "--num-workers", 0,
        "--selection-metric", "rmse",
        "--device", $Device,
        "--output-dir", $outDir
    )
    if ($UsePhysics) {
        $runArgs += @(
            "--physics-preset", $FinalPhysics,
            "--physics-time-mode", "daily_rollout",
            "--forcing-mode", "perfect",
            "--forecast-mode", "free",
            "--entrainment-dt", 1.0,
            "--we-nonneg",
            "--we-softplus-shift", 1.0,
            "--lambda-physics", $LambdaPhysics,
            "--lambda-rres", $LambdaRres,
            "--lambda-we-tie", $LambdaWeTie,
            "--rres-penalty-normalize", "rhs",
            "--adaptive-lambda-mode", "fixed",
            "--physics-loss", "mse",
            "--physics-normalize", "rhs",
            "--physics-warmup-epochs", 5,
            "--k-diff", $KDiff,
            "--zero-rres-physics",
            "--hm-min", 5.0
        )
    }
    if ($SmokeOnly) {
        $runArgs += @(
            "--epochs", 1,
            "--patience", 1,
            "--max-train-batches", 1,
            "--max-val-batches", 1,
            "--max-test-batches", 1
        )
    }
    Invoke-PythonLogged $runArgs $logPath "$Family failed: lead=$Lead seed=$Seed"
}

function Invoke-HiLAMRun([string]$Family, [int]$Lead, [int]$Seed, [bool]$UsePhysics) {
    $modelId = if ($UsePhysics) { "pi_jeju_hilam" } else { "jeju_hilam" }
    $familyRoot = Join-Path $ActiveOutputRoot $Family
    $runDir = Join-Path $familyRoot "lead$($Lead)_seed$Seed"
    $metricsPath = Join-Path $runDir ("{0}_lead_{1}_history_{2}_metrics.json" -f $modelId, $Lead, $History)
    if (Test-MetricDone $metricsPath) {
        Write-Host "SKIP existing: $metricsPath" -ForegroundColor DarkYellow
        return
    }
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $env:PYTHONHASHSEED = [string]$Seed
    $logPath = Join-Path $runDir "train_final.log"
    Write-Host "=== $Family lead=$Lead seed=$Seed ===" -ForegroundColor Cyan

    $runArgs = @(
        "train_jeju_hilam.py",
        "--data-path", $DataPath,
        "--output-root", $familyRoot,
        "--history", $History,
        "--lead", $Lead,
        "--seed", $Seed,
        "--epochs", $HiLAMEpochs,
        "--patience", $HiLAMPatience,
        "--batch-size", $BatchSize,
        "--num-workers", 0,
        "--hidden-dim", $HiLAMHiddenDim,
        "--temporal-hidden-dim", $HiLAMTemporalHiddenDim,
        "--processor-layers", $HiLAMProcessorLayers,
        "--lr", $HiLAMlr,
        "--initial-lr", $HiLAMInitialLr,
        "--weight-decay", $HiLAMWeightDecay,
        "--selection-metric", "val_rmse",
        "--device", $HiLAMDevice,
        "--prediction-mode", "direct",
        "--no-residual-feature"
    )
    if ($UsePhysics) {
        $runArgs += @(
            "--physics-mode", $physicsMode,
            "--lambda-physics", $LambdaPhysics,
            "--physics-warmup-epochs", 5,
            "--lambda-rres", $LambdaRres,
            "--rres-penalty-normalize", "rhs",
            "--entrainment-dt", 1.0,
            "--hm-min", 5.0,
            "--we-scale", 1.0e-5,
            "--rres-scale", 1.0e-4,
            "--we-softplus-shift", 1.0
        )
    }
    else {
        $runArgs += @("--physics-mode", "none")
    }
    if ($SmokeOnly) {
        $runArgs += @("--smoke-test", "--max-train-batches", 1, "--max-val-batches", 1)
    }
    Invoke-PythonLogged $runArgs $logPath "$Family failed: lead=$Lead seed=$Seed"
}

foreach ($lead in $RunLeadList) {
    foreach ($seed in $RunSeedList) {
        if (-not $SkipConvLSTM) { Invoke-ConvLSTMRun "convlstm" $lead $seed $false }
        if (-not $SkipConvLSTMPINN) { Invoke-ConvLSTMRun "convlstm_pinn" $lead $seed $true }
        if (-not $SkipHiLAM) { Invoke-HiLAMRun "jeju_hilam" $lead $seed $false }
        if (-not $SkipPIHiLAM) { Invoke-HiLAMRun "pi_jeju_hilam" $lead $seed $true }
    }
}

if (-not $SmokeOnly) {
    $manifestNow = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    $manifestNow | Add-Member -NotePropertyName "reused_runs" -NotePropertyValue $script:ReusedRuns -Force
    $manifestNow | Add-Member -NotePropertyName "reused_run_count" -NotePropertyValue $script:ReusedRuns.Count -Force
    $manifestNow | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Write-Host "Reused $($script:ReusedRuns.Count) run(s) from $AblationRoot" -ForegroundColor Magenta
}

if ($SmokeOnly) {
    Write-Host "SmokeOnly requested; stopping after short smoke runs." -ForegroundColor Yellow
    if ($PauseOnExit) { Read-Host "Press Enter to close/return" | Out-Null }
    return
}

if ($RunSummary) {
    $summaryLog = Join-Path $OutputRoot "summarize_main_experiment_final.log"
    $summaryArgs = @(
        "summarize_main_experiment_final.py",
        "--data-path", $DataPath,
        "--root", $OutputRoot,
        "--history", $History,
        "--leads", ($LeadList -join ","),
        "--seeds", ($SeedList -join ","),
        "--final-physics", $FinalPhysics
    )
    Invoke-PythonLogged $summaryArgs $summaryLog "summary failed"
}

Write-Host "Done: $OutputRoot" -ForegroundColor Green
Write-Host "FINAL_PHYSICS=$FinalPhysics; no P1/P2/P3 winner is selected in the main experiment." -ForegroundColor Green
if ($PauseOnExit) {
    Read-Host "Press Enter to close/return" | Out-Null
}
