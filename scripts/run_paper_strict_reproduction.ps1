param(
  [ValidateSet("llvip", "yeoju", "both")]
  [string]$Dataset = "both",
  [int]$Epochs = 10,
  [int]$BatchSize = 2,
  [double]$Lr = 0.0001,
  [int]$Seed = 317,
  [switch]$SkipIlluminationPretrain
)

$ErrorActionPreference = "Stop"

$Python = "C:\Users\owner\.conda\envs\CenterNet\python.exe"
$Project = Get-ChildItem -Path "D:\KDG" -Directory |
  Where-Object {
    $_.Name -like "CenterNet_Origin -*" -and
    (Test-Path -LiteralPath (Join-Path $_.FullName "src\main.py"))
  } |
  Select-Object -First 1 -ExpandProperty FullName

if (-not $Project) {
  throw "Could not find CenterNet project folder under D:\KDG"
}

$Src = Join-Path $Project "src"
$LoadModel = Join-Path $Project "models\ctdet_coco_hg.pth"
$RunRoot = "D:\KDG\paper_strict_runs"
$DataRoot = "D:\KDG\paper_strict_data"
$LogRoot = Join-Path $RunRoot "_logs"
$TrainLogRoot = Join-Path $RunRoot "_train_logs"
$EvalLogRoot = Join-Path $RunRoot "_eval_logs"
$SummaryCsv = Join-Path $RunRoot "paper_strict_summary.csv"
$MasterLog = Join-Path $LogRoot ("paper_strict_master_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$IlluminationDir = Join-Path $RunRoot "illumination_llvip"
$IlluminationCkpt = Join-Path $IlluminationDir "illumination_best.pth"

New-Item -ItemType Directory -Force -Path $RunRoot, $DataRoot, $LogRoot, $TrainLogRoot, $EvalLogRoot | Out-Null

function Write-Log($Message) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
  $line | Tee-Object -FilePath $MasterLog -Append
}

function Invoke-Logged($Title, $Exe, $CommandArgs, $StepLog) {
  Write-Log "START: $Title"
  "===== $Title / $(Get-Date) =====" | Out-File -FilePath $StepLog -Encoding utf8
  & $Exe @CommandArgs 2>&1 | Tee-Object -FilePath $StepLog -Append | Tee-Object -FilePath $MasterLog -Append
  $code = $LASTEXITCODE
  if ($code -ne 0) {
    throw "$Title failed with exit code $code"
  }
  Write-Log "DONE: $Title"
}

function Set-CommonEnv($DatasetRoot, $NumClasses, $ValidIds, $ClassNames, $Wrapper) {
  $env:KDG_COCO_DATA_DIR = $DatasetRoot
  $env:KDG_COCO_IMAGE_SUBDIR = ""
  $env:KDG_NUM_CLASSES = $NumClasses
  $env:KDG_VALID_IDS = $ValidIds
  $env:KDG_CLASS_NAMES = $ClassNames
  $env:KDG_GAMMA_WRAPPER = $Wrapper
  $env:KDG_DATASET_CNN_DIP = "0"
  $env:KDG_EXP_DIR = $RunRoot
  $env:KMP_DUPLICATE_LIB_OK = "TRUE"
  $env:PYTHONWARNINGS = "ignore"
  $env:PYTHONUNBUFFERED = "1"
}

function Set-ProposedEnv($IlluminationCkpt) {
  $env:KDG_GAMMA_LOSS_WEIGHT = "0"
  $env:KDG_PAPER_GAMMA_RANGE = "3"
  $env:KDG_PAPER_SHARPNESS_MAX = "5"
  $env:KDG_PAPER_SHARPNESS_NIGHT_WEIGHT = "1"
  $env:KDG_ILLUMINATION_CKPT = $IlluminationCkpt
  $env:KDG_FREEZE_ILLUMINATION = "1"
}

function Train-Model($Exp, $DatasetRoot, $NumClasses, $ValidIds, $ClassNames, $Wrapper) {
  Set-CommonEnv $DatasetRoot $NumClasses $ValidIds $ClassNames $Wrapper
  if ($Wrapper -eq "paper_filter") {
    Set-ProposedEnv $IlluminationCkpt
  }
  else {
    $env:KDG_GAMMA_LOSS_WEIGHT = ""
    $env:KDG_ILLUMINATION_CKPT = ""
  }

  $SaveDir = Join-Path $RunRoot $Exp
  New-Item -ItemType Directory -Force -Path $SaveDir, (Join-Path $SaveDir "debug") | Out-Null
  $StepLog = Join-Path $TrainLogRoot "$Exp.txt"
  $CommandArgs = @(
    "$Src\main.py",
    "--exp_id", $Exp,
    "--arch", "hourglass",
    "--gpus", "0",
    "--batch_size", "$BatchSize",
    "--num_epochs", "$Epochs",
    "--lr", "$Lr",
    "--val_intervals", "0",
    "--print_iter", "20",
    "--num_workers", "4",
    "--save_interval", "$Epochs",
    "--load_model", $LoadModel
  )
  Push-Location "D:\KDG"
  try {
    Invoke-Logged "train $Exp" $Python $CommandArgs $StepLog
  }
  finally {
    Pop-Location
  }
}

function Eval-Model($Exp, $DatasetRoot, $NumClasses, $ValidIds, $ClassNames, $Wrapper) {
  Set-CommonEnv $DatasetRoot $NumClasses $ValidIds $ClassNames $Wrapper
  if ($Wrapper -eq "paper_filter") {
    Set-ProposedEnv $IlluminationCkpt
  }

  $EvalExp = "${Exp}_eval"
  $Model = Join-Path $RunRoot "$Exp\model_last.pth"
  if (-not (Test-Path -LiteralPath $Model)) {
    throw "Missing trained model: $Model"
  }
  $StepLog = Join-Path $EvalLogRoot "$EvalExp.txt"
  $CommandArgs = @(
    "$Src\test.py",
    "--test",
    "--not_prefetch_test",
    "--gpus", "0",
    "--exp_id", $EvalExp,
    "--load_model", $Model
  )
  Push-Location "D:\KDG"
  try {
    Invoke-Logged "eval $EvalExp" $Python $CommandArgs $StepLog
  }
  finally {
    Pop-Location
  }
}

function Parse-Metric($Text, $Pattern) {
  $match = [regex]::Match($Text, $Pattern)
  if ($match.Success) { return $match.Groups[1].Value }
  return ""
}

function Append-Summary($DatasetName, $ModelName, $Exp) {
  $Log = Join-Path $EvalLogRoot "${Exp}_eval.txt"
  $Text = (Get-Content -LiteralPath $Log -Raw) -replace "`0", ""
  $ap = Parse-Metric $Text "IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)"
  $ap50 = Parse-Metric $Text "IoU=0\.50\s+\| area=\s+all \| maxDets=100 \] = ([0-9.]+)"
  $ap75 = Parse-Metric $Text "IoU=0\.75\s+\| area=\s+all \| maxDets=100 \] = ([0-9.]+)"
  $ar100 = Parse-Metric $Text "Average Recall\s+\(AR\) @\[ IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)"
  "$DatasetName,$ModelName,$Exp,$ap,$ap50,$ap75,$ar100" | Out-File -FilePath $SummaryCsv -Encoding utf8 -Append
}

if (-not (Test-Path -LiteralPath $Python)) {
  throw "Missing Python: $Python"
}
if (-not (Test-Path -LiteralPath $LoadModel)) {
  throw "Missing CenterNet COCO pretrained model: $LoadModel"
}

Write-Log "Paper strict reproduction started"
Write-Log "Dataset=$Dataset epochs=$Epochs batch=$BatchSize lr=$Lr seed=$Seed"
Write-Log "Project=$Project"

$llvipSource = "D:\KDG\llvip\coco_llvip_rgb"
$llvipStrict = Join-Path $DataRoot "llvip_7to3_seed$Seed\coco_llvip_rgb"
$yeojuSource = "D:\KDG\Yeoju_rain\coco_llvip_rgb"
$yeojuStrict = Join-Path $DataRoot "yeoju_7to3_seed$Seed\coco_llvip_rgb"

$splitTargets = @()
if ($Dataset -eq "llvip" -or $Dataset -eq "both") {
  $splitTargets += [pscustomobject]@{
    Name = "LLVIP"
    Source = $llvipSource
    Output = $llvipStrict
  }
}
if ($Dataset -eq "yeoju" -or $Dataset -eq "both") {
  $splitTargets += [pscustomobject]@{
    Name = "Yeoju"
    Source = $yeojuSource
    Output = $yeojuStrict
  }
}

foreach ($target in $splitTargets) {
  $name = $target.Name
  $source = $target.Source
  $output = $target.Output
  $StepLog = Join-Path $LogRoot "split_${name}_7to3_seed$Seed.txt"
  $CommandArgs = @(
    "D:\KDG\make_coco_7to3_split.py",
    "--source", $source,
    "--output", $output,
    "--seed", "$Seed",
    "--train-ratio", "0.7",
    "--mode", "hardlink"
  )
  Invoke-Logged "make 7:3 split $name" $Python $CommandArgs $StepLog
}

if (-not $SkipIlluminationPretrain) {
  $StepLog = Join-Path $LogRoot "pretrain_illumination_llvip.txt"
  $CommandArgs = @(
    "D:\KDG\pretrain_illumination_classifier.py",
    "--src", $Src,
    "--dataset-root", $llvipSource,
    "--labels", "D:\KDG\llvip_gamma_brightness\fixed_merged_predictions.txt",
    "--output-dir", $IlluminationDir,
    "--epochs", "10",
    "--batch-size", "64",
    "--lr", "0.001",
    "--image-size", "512",
    "--num-workers", "4",
    "--seed", "$Seed",
    "--train-ratio", "0.7"
  )
  Invoke-Logged "pretrain illumination classifier LLVIP" $Python $CommandArgs $StepLog
}
elseif (-not (Test-Path -LiteralPath $IlluminationCkpt)) {
  throw "SkipIlluminationPretrain was set, but missing $IlluminationCkpt"
}

"dataset,model,exp,AP,AP50,AP75,AR100" | Out-File -FilePath $SummaryCsv -Encoding utf8

if ($Dataset -eq "llvip" -or $Dataset -eq "both") {
  $base = "paper_strict_llvip_baseline_bs${BatchSize}_${Epochs}ep_seed$Seed"
  $prop = "paper_strict_llvip_proposed_bs${BatchSize}_${Epochs}ep_seed$Seed"
  Train-Model $base $llvipStrict "1" "1" "person" "none"
  Eval-Model $base $llvipStrict "1" "1" "person" "none"
  Append-Summary "LLVIP" "CenterNet" $base
  Train-Model $prop $llvipStrict "1" "1" "person" "paper_filter"
  Eval-Model $prop $llvipStrict "1" "1" "person" "paper_filter"
  Append-Summary "LLVIP" "Proposed CenterNet" $prop
}

if ($Dataset -eq "yeoju" -or $Dataset -eq "both") {
  $base = "paper_strict_yeoju_baseline_bs${BatchSize}_${Epochs}ep_seed$Seed"
  $prop = "paper_strict_yeoju_proposed_bs${BatchSize}_${Epochs}ep_seed$Seed"
  Train-Model $base $yeojuStrict "2" "1,2" "person,car" "none"
  Eval-Model $base $yeojuStrict "2" "1,2" "person,car" "none"
  Append-Summary "Yeoju/CCTV" "CenterNet" $base
  Train-Model $prop $yeojuStrict "2" "1,2" "person,car" "paper_filter"
  Eval-Model $prop $yeojuStrict "2" "1,2" "person,car" "paper_filter"
  Append-Summary "Yeoju/CCTV" "Proposed CenterNet" $prop
}

Write-Log "Paper strict reproduction finished"
Write-Log "Summary CSV: $SummaryCsv"
