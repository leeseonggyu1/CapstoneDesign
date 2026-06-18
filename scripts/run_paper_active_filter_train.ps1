param(
  [ValidateSet("llvip", "yeoju", "both")]
  [string]$Dataset = "yeoju",
  [ValidateSet("proposed10", "baseline10")]
  [string]$Source = "proposed10",
  [int]$ExtraEpochs = 30,
  [int]$BatchSize = 2,
  [double]$BaseLr = 0.00005,
  [double]$FilterLrMult = 50,
  [double]$GammaRange = 5,
  [double]$SharpnessMax = 8,
  [int]$SaveInterval = 5,
  [int]$Seed = 317,
  [int]$NumWorkers = 4,
  [bool]$ResetFilterEstimator = $true
)

$ErrorActionPreference = "Stop"

$Python = "C:\Users\owner\.conda\envs\CenterNet\python.exe"
$Project = Get-ChildItem -LiteralPath "D:\KDG" -Directory |
  Where-Object {
    (Test-Path -LiteralPath (Join-Path $_.FullName "src\main.py")) -and
    (Test-Path -LiteralPath (Join-Path $_.FullName "gamma_values_epoch_None.txt"))
  } |
  Select-Object -First 1 -ExpandProperty FullName

if (-not $Project) {
  throw "Could not find gamma-saving CenterNet project under D:\KDG"
}

$Src = Join-Path $Project "src"
$RunRoot = "D:\KDG\paper_strict_runs"
$LogRoot = Join-Path $RunRoot "_active_filter_logs"
$IlluminationCkpt = Join-Path $RunRoot "illumination_llvip\illumination_best.pth"
$ResetTag = if ($ResetFilterEstimator) { "reset" } else { "keep" }
$FilterTag = ("flr{0:g}" -f $FilterLrMult).Replace(".", "p")
$GammaTag = ("g{0:g}" -f $GammaRange).Replace(".", "p")
$SharpTag = ("s{0:g}" -f $SharpnessMax).Replace(".", "p")

if (-not (Test-Path -LiteralPath $Python)) {
  throw "Python not found: $Python"
}
if (-not (Test-Path -LiteralPath $IlluminationCkpt)) {
  throw "Illumination checkpoint not found: $IlluminationCkpt"
}
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Set-ProposedEnv($DatasetRoot, $NumClasses, $ValidIds, $ClassNames) {
  $env:KDG_COCO_DATA_DIR = $DatasetRoot
  $env:KDG_COCO_IMAGE_SUBDIR = ""
  $env:KDG_NUM_CLASSES = $NumClasses
  $env:KDG_VALID_IDS = $ValidIds
  $env:KDG_CLASS_NAMES = $ClassNames
  $env:KDG_GAMMA_WRAPPER = "paper_filter"
  $env:KDG_DATASET_CNN_DIP = "0"
  $env:KDG_EXP_DIR = $RunRoot
  $env:KDG_GAMMA_LOSS_WEIGHT = "0"
  $env:KDG_PAPER_GAMMA_RANGE = "$GammaRange"
  $env:KDG_PAPER_SHARPNESS_MAX = "$SharpnessMax"
  $env:KDG_PAPER_SHARPNESS_NIGHT_WEIGHT = "1"
  $env:KDG_ILLUMINATION_CKPT = $IlluminationCkpt
  $env:KDG_FREEZE_ILLUMINATION = "1"
  $env:KDG_FILTER_LR_MULT = "$FilterLrMult"
  $env:KDG_RESET_FILTER_ESTIMATOR = if ($ResetFilterEstimator) { "1" } else { "0" }
  Remove-Item Env:\KDG_FILTER_LR -ErrorAction SilentlyContinue
  $env:KMP_DUPLICATE_LIB_OK = "TRUE"
  $env:PYTHONWARNINGS = "ignore"
  $env:PYTHONUNBUFFERED = "1"
}

function Get-SourceCheckpoint($Name) {
  $SourceExp = if ($Source -eq "baseline10") {
    "paper_strict_${Name}_baseline_bs2_10ep_seed$Seed"
  } else {
    "paper_strict_${Name}_proposed_bs2_10ep_seed$Seed"
  }
  $Path = Join-Path $RunRoot "$SourceExp\model_last.pth"
  if (-not (Test-Path -LiteralPath $Path)) {
    throw "Missing source checkpoint: $Path"
  }
  return $Path
}

function Run-ActiveTrain($Name, $DatasetRoot, $NumClasses, $ValidIds, $ClassNames) {
  $LoadModel = Get-SourceCheckpoint $Name
  Set-ProposedEnv $DatasetRoot $NumClasses $ValidIds $ClassNames

  $TargetExp = "paper_active_${Name}_${Source}_extra${ExtraEpochs}_base${BaseLr}_${FilterTag}_${GammaTag}_${SharpTag}_${ResetTag}_seed$Seed"
  $TargetExp = $TargetExp.Replace(".", "p").Replace("-", "m")
  $SaveDir = Join-Path $RunRoot $TargetExp
  $Log = Join-Path $LogRoot "$TargetExp.txt"
  New-Item -ItemType Directory -Force -Path $SaveDir, (Join-Path $SaveDir "debug") | Out-Null

  "===== $TargetExp / $(Get-Date) =====" | Out-File -FilePath $Log -Encoding utf8
  "Dataset: $Name" | Out-File -FilePath $Log -Encoding utf8 -Append
  "DatasetRoot: $DatasetRoot" | Out-File -FilePath $Log -Encoding utf8 -Append
  "LoadModel: $LoadModel" | Out-File -FilePath $Log -Encoding utf8 -Append
  "IlluminationCkpt: $IlluminationCkpt" | Out-File -FilePath $Log -Encoding utf8 -Append
  "Config: extra_epochs=$ExtraEpochs, base_lr=$BaseLr, filter_lr=$($BaseLr * $FilterLrMult), gamma_range=$GammaRange, sharpness_max=$SharpnessMax, reset_filter_estimator=$ResetFilterEstimator" |
    Out-File -FilePath $Log -Encoding utf8 -Append

  Write-Host "Dataset:  $Name"
  Write-Host "Source:   $LoadModel"
  Write-Host "Output:   $SaveDir"
  Write-Host "Log:      $Log"
  Write-Host "LR:       base=$BaseLr, gamma/sharp=$($BaseLr * $FilterLrMult)"
  Write-Host "Filter:   gamma_range=$GammaRange, sharpness_max=$SharpnessMax, reset=$ResetFilterEstimator"

  $CommandArgs = @(
    "$Src\main.py",
    "--exp_id", $TargetExp,
    "--arch", "hourglass",
    "--gpus", "0",
    "--debug", "0",
    "--batch_size", "$BatchSize",
    "--num_epochs", "$ExtraEpochs",
    "--lr", "$BaseLr",
    "--val_intervals", "0",
    "--print_iter", "20",
    "--num_workers", "$NumWorkers",
    "--save_interval", "$SaveInterval",
    "--load_model", $LoadModel
  )

  Push-Location "D:\KDG"
  try {
    & $Python @CommandArgs 2>&1 | Tee-Object -FilePath $Log -Append
    if ($LASTEXITCODE -ne 0) {
      throw "$TargetExp failed with exit code $LASTEXITCODE"
    }
  }
  finally {
    Pop-Location
  }
}

if ($Dataset -eq "llvip" -or $Dataset -eq "both") {
  Run-ActiveTrain `
    "llvip" `
    "D:\KDG\paper_strict_data\llvip_7to3_seed317\coco_llvip_rgb" `
    "1" `
    "1" `
    "person"
}

if ($Dataset -eq "yeoju" -or $Dataset -eq "both") {
  Run-ActiveTrain `
    "yeoju" `
    "D:\KDG\paper_strict_data\yeoju_7to3_seed317\coco_llvip_rgb" `
    "2" `
    "1,2" `
    "person,car"
}
