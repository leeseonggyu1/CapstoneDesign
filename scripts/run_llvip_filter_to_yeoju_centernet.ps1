param(
  [int]$LLVIPFilterEpochs = 30,
  [int]$YeojuCenterNetEpochs = 30,
  [int]$BatchSize = 8,
  [int]$NumWorkers = 4,
  [double]$Lr = 0.000125,
  [double]$FilterLrMult = 5.0,
  [double]$GammaRange = 3.0,
  [double]$SharpnessMax = 5.0,
  [int]$Seed = 317,
  [int]$SaveInterval = 5,
  [switch]$SkipSplit,
  [switch]$SkipLLVIPFilterTrain,
  [switch]$SkipFilterImages,
  [switch]$SkipYeojuCenterNet,
  [switch]$RunRawBaseline,
  [string]$ExistingLLVIPFilterModel = ""
)

$ErrorActionPreference = "Stop"

$Python = "C:\Users\owner\.conda\envs\CenterNet\python.exe"
$Project = Get-ChildItem -Path "D:\KDG" -Directory |
  Where-Object {
    $_.Name -like "CenterNet_Origin -*" -and
    (Test-Path -LiteralPath (Join-Path $_.FullName "src\main.py"))
  } |
  Select-Object -First 1 -ExpandProperty FullName

if (-not $Project) { throw "Could not find CenterNet project folder under D:\KDG" }
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing Python: $Python" }

$Src = Join-Path $Project "src"
$CocoPretrained = Join-Path $Project "models\ctdet_coco_hg.pth"
$RunRoot = "D:\KDG\llvip_filter_to_yeoju_runs"
$DataRoot = "D:\KDG\paper_strict_data"
$LLVIPSource = "D:\KDG\llvip\coco_llvip_rgb"
$YeojuSource = "D:\KDG\Yeoju_rain\coco_llvip_rgb"
$LLVIPSplit = Join-Path $DataRoot "llvip_7to3_seed$Seed\coco_llvip_rgb"
$YeojuSplit = Join-Path $DataRoot "yeoju_7to3_seed$Seed\coco_llvip_rgb"
$IlluminationCkpt = "D:\KDG\paper_strict_runs\illumination_llvip\illumination_best.pth"

$LogRoot = Join-Path $RunRoot "_logs"
$TrainLogRoot = Join-Path $RunRoot "_train_logs"
$EvalLogRoot = Join-Path $RunRoot "_eval_logs"
$SummaryCsv = Join-Path $RunRoot "summary.csv"
$MasterLog = Join-Path $LogRoot ("master_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

$FilterTag = "llvip_filter_bs${BatchSize}_${LLVIPFilterEpochs}ep_flr${FilterLrMult}_seed$Seed"
$LLVIPFilterExp = $FilterTag
$FilteredRoot = Join-Path $RunRoot "${FilterTag}_yeoju_filtered_images\coco_llvip_rgb"
$YeojuFilteredExp = "yeoju_centernet_on_llvip_filter_bs${BatchSize}_${YeojuCenterNetEpochs}ep_seed$Seed"
$RawBaseExp = "yeoju_raw_centernet_bs${BatchSize}_${YeojuCenterNetEpochs}ep_seed$Seed"

New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot, $TrainLogRoot, $EvalLogRoot | Out-Null

function Write-Log($Message) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
  $line | Tee-Object -FilePath $MasterLog -Append
}

function Invoke-Step($Title, $Exe, $CommandArgs, $StepLog) {
  Write-Log "START: $Title"
  "===== $Title / $(Get-Date) =====" | Out-File -FilePath $StepLog -Encoding utf8
  & $Exe @CommandArgs 2>&1 | Tee-Object -FilePath $StepLog -Append | Tee-Object -FilePath $MasterLog -Append
  $code = $LASTEXITCODE
  if ($code -ne 0) { throw "$Title failed with exit code $code" }
  Write-Log "DONE: $Title"
}

function Set-CommonEnv($DatasetRoot, $Wrapper, $NumClasses, $ValidIds, $ClassNames) {
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

function Set-PaperFilterEnv() {
  $env:KDG_ILLUMINATION_CKPT = $IlluminationCkpt
  $env:KDG_FREEZE_ILLUMINATION = "1"
  $env:KDG_PAPER_GAMMA_RANGE = "$GammaRange"
  $env:KDG_PAPER_SHARPNESS_MAX = "$SharpnessMax"
  $env:KDG_PAPER_SHARPNESS_NIGHT_WEIGHT = "1"
  $env:KDG_GAMMA_LOSS_WEIGHT = "0"
  $env:KDG_FILTER_LR_MULT = "$FilterLrMult"
}

function Clear-PaperFilterEnv() {
  $env:KDG_ILLUMINATION_CKPT = ""
  $env:KDG_FREEZE_ILLUMINATION = ""
  $env:KDG_PAPER_GAMMA_RANGE = ""
  $env:KDG_PAPER_SHARPNESS_MAX = ""
  $env:KDG_PAPER_SHARPNESS_NIGHT_WEIGHT = ""
  $env:KDG_GAMMA_LOSS_WEIGHT = ""
  $env:KDG_FILTER_LR_MULT = ""
}

function Train-Model($Exp, $DatasetRoot, $Wrapper, $Epochs, $LoadModel, $NumClasses, $ValidIds, $ClassNames, $LogName) {
  Set-CommonEnv $DatasetRoot $Wrapper $NumClasses $ValidIds $ClassNames
  if ($Wrapper -eq "paper_filter") { Set-PaperFilterEnv } else { Clear-PaperFilterEnv }
  $args = @(
    "$Src\main.py",
    "--exp_id", $Exp,
    "--arch", "hourglass",
    "--gpus", "0",
    "--batch_size", "$BatchSize",
    "--num_epochs", "$Epochs",
    "--lr", "$Lr",
    "--val_intervals", "0",
    "--print_iter", "50",
    "--num_workers", "$NumWorkers",
    "--save_interval", "$SaveInterval",
    "--load_model", $LoadModel
  )
  Invoke-Step "train $Exp" $Python $args (Join-Path $TrainLogRoot $LogName)
}

function Eval-Model($Exp, $DatasetRoot, $Wrapper, $Model, $NumClasses, $ValidIds, $ClassNames, $LogName) {
  Set-CommonEnv $DatasetRoot $Wrapper $NumClasses $ValidIds $ClassNames
  if ($Wrapper -eq "paper_filter") { Set-PaperFilterEnv } else { Clear-PaperFilterEnv }
  $args = @(
    "$Src\test.py",
    "--test",
    "--not_prefetch_test",
    "--gpus", "0",
    "--exp_id", $Exp,
    "--load_model", $Model
  )
  Invoke-Step "eval $Exp" $Python $args (Join-Path $EvalLogRoot $LogName)
}

function Parse-Metric($Text, $Pattern) {
  $match = [regex]::Match($Text, $Pattern)
  if ($match.Success) { return $match.Groups[1].Value }
  return ""
}

function Append-Summary($Name, $Exp, $LogName) {
  $log = Join-Path $EvalLogRoot $LogName
  if (-not (Test-Path -LiteralPath $log)) { return }
  $text = (Get-Content -LiteralPath $log -Raw) -replace "`0", ""
  $ap = Parse-Metric $text "IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)"
  $ap50 = Parse-Metric $text "IoU=0\.50\s+\| area=\s+all \| maxDets=100 \] = ([0-9.]+)"
  $ap75 = Parse-Metric $text "IoU=0\.75\s+\| area=\s+all \| maxDets=100 \] = ([0-9.]+)"
  $ar100 = Parse-Metric $text "Average Recall\s+\(AR\) @\[ IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)"
  "$Name,$Exp,$ap,$ap50,$ap75,$ar100" | Out-File -FilePath $SummaryCsv -Encoding utf8 -Append
}

if (-not (Test-Path -LiteralPath $CocoPretrained)) { throw "Missing COCO pretrained model: $CocoPretrained" }
if (-not (Test-Path -LiteralPath $IlluminationCkpt)) { throw "Missing LLVIP illumination checkpoint: $IlluminationCkpt" }

Write-Log "LLVIP filter -> Yeoju CenterNet pipeline started"
Write-Log "Project=$Project"
Write-Log "LLVIP split=$LLVIPSplit"
Write-Log "Yeoju split=$YeojuSplit"
Write-Log "Illumination=$IlluminationCkpt"

if (-not $SkipSplit) {
  $args = @(
    "D:\KDG\make_coco_7to3_split.py",
    "--source", $LLVIPSource,
    "--output", $LLVIPSplit,
    "--seed", "$Seed",
    "--train-ratio", "0.7",
    "--mode", "hardlink"
  )
  Invoke-Step "make LLVIP 7:3 split" $Python $args (Join-Path $LogRoot "split_llvip.txt")

  $args = @(
    "D:\KDG\make_coco_7to3_split.py",
    "--source", $YeojuSource,
    "--output", $YeojuSplit,
    "--seed", "$Seed",
    "--train-ratio", "0.7",
    "--mode", "hardlink"
  )
  Invoke-Step "make Yeoju 7:3 split" $Python $args (Join-Path $LogRoot "split_yeoju.txt")
}

if (-not $SkipLLVIPFilterTrain) {
  Train-Model `
    $LLVIPFilterExp $LLVIPSplit "paper_filter" $LLVIPFilterEpochs $CocoPretrained `
    "1" "1" "person" "$LLVIPFilterExp.txt"
  $LLVIPFilterModel = Join-Path $RunRoot "$LLVIPFilterExp\model_last.pth"
  Eval-Model `
    "${LLVIPFilterExp}_eval" $LLVIPSplit "paper_filter" $LLVIPFilterModel `
    "1" "1" "person" "${LLVIPFilterExp}_eval.txt"
}
else {
  if ($ExistingLLVIPFilterModel -eq "") {
    $ExistingLLVIPFilterModel = Join-Path $RunRoot "$LLVIPFilterExp\model_last.pth"
  }
  if (-not (Test-Path -LiteralPath $ExistingLLVIPFilterModel)) {
    throw "Missing ExistingLLVIPFilterModel: $ExistingLLVIPFilterModel"
  }
  $LLVIPFilterModel = $ExistingLLVIPFilterModel
  Write-Log "Using existing LLVIP filter model: $LLVIPFilterModel"
}

if (-not $SkipFilterImages) {
  $args = @(
    "D:\KDG\generate_paper_filtered_images.py",
    "--src", $Src,
    "--input-root", $YeojuSplit,
    "--output-root", $FilteredRoot,
    "--checkpoint", $LLVIPFilterModel,
    "--illumination-ckpt", $IlluminationCkpt,
    "--num-classes", "1",
    "--gamma-range", "$GammaRange",
    "--sharpness-max", "$SharpnessMax"
  )
  Invoke-Step "generate Yeoju images with LLVIP-trained filter" $Python $args (Join-Path $LogRoot "generate_yeoju_filtered_by_llvip_filter.txt")
}

if ($RunRawBaseline) {
  Train-Model `
    $RawBaseExp $YeojuSplit "none" $YeojuCenterNetEpochs $CocoPretrained `
    "2" "1,2" "person,car" "$RawBaseExp.txt"
  $RawBaseModel = Join-Path $RunRoot "$RawBaseExp\model_last.pth"
  Eval-Model `
    "${RawBaseExp}_eval" $YeojuSplit "none" $RawBaseModel `
    "2" "1,2" "person,car" "${RawBaseExp}_eval.txt"
}

if (-not $SkipYeojuCenterNet) {
  Train-Model `
    $YeojuFilteredExp $FilteredRoot "none" $YeojuCenterNetEpochs $CocoPretrained `
    "2" "1,2" "person,car" "$YeojuFilteredExp.txt"
  $YeojuFilteredModel = Join-Path $RunRoot "$YeojuFilteredExp\model_last.pth"
  Eval-Model `
    "${YeojuFilteredExp}_eval" $FilteredRoot "none" $YeojuFilteredModel `
    "2" "1,2" "person,car" "${YeojuFilteredExp}_eval.txt"
}

"name,exp,AP,AP50,AP75,AR100" | Out-File -FilePath $SummaryCsv -Encoding utf8
Append-Summary "llvip_filter_eval" $LLVIPFilterExp "${LLVIPFilterExp}_eval.txt"
Append-Summary "raw_yeoju_baseline" $RawBaseExp "${RawBaseExp}_eval.txt"
Append-Summary "yeoju_centernet_on_llvip_filter" $YeojuFilteredExp "${YeojuFilteredExp}_eval.txt"

Write-Log "Finished"
Write-Log "Summary: $SummaryCsv"
Write-Host ""
Write-Host "Summary: $SummaryCsv"
Write-Host "Filtered Yeoju images: $FilteredRoot"
