param(
  [int]$EndToEndEpochs = 30,
  [int]$FilteredEpochs = 10,
  [int]$BatchSize = 8,
  [int]$NumWorkers = 4,
  [double]$Lr = 0.000125,
  [double]$FilterLrMult = 5.0,
  [double]$GammaRange = 3.0,
  [double]$SharpnessMax = 5.0,
  [int]$Seed = 317,
  [int]$SaveInterval = 5,
  [switch]$RunRawBaseline,
  [switch]$SkipSplit,
  [switch]$SkipEndToEnd,
  [switch]$SkipFilterImages,
  [switch]$SkipFilteredCenterNet
)

$ErrorActionPreference = "Continue"

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
$RunRoot = "D:\KDG\yeoju_llvipillum_pipeline_runs"
$DataRoot = "D:\KDG\paper_strict_data"
$YeojuSource = "D:\KDG\Yeoju_rain\coco_llvip_rgb"
$YeojuSplit = Join-Path $DataRoot "yeoju_7to3_seed$Seed\coco_llvip_rgb"
$IlluminationCkpt = "D:\KDG\paper_strict_runs\illumination_llvip\illumination_best.pth"
$LogRoot = Join-Path $RunRoot "_logs"
$EvalLogRoot = Join-Path $RunRoot "_eval_logs"
$TrainLogRoot = Join-Path $RunRoot "_train_logs"
$SummaryCsv = Join-Path $RunRoot "summary.csv"
$MasterLog = Join-Path $LogRoot ("master_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

$E2EExp = "yeoju_llvipillum_e2e_bs${BatchSize}_${EndToEndEpochs}ep_flr${FilterLrMult}_seed$Seed"
$FilteredRoot = Join-Path $RunRoot "${E2EExp}_filtered_images\coco_llvip_rgb"
$FilteredExp = "yeoju_llvipillum_filtered_centernet_bs${BatchSize}_${FilteredEpochs}ep_seed$Seed"
$RawBaseExp = "yeoju_raw_centernet_bs${BatchSize}_${FilteredEpochs}ep_seed$Seed"

New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot, $EvalLogRoot, $TrainLogRoot | Out-Null

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

function Set-CommonEnv($DatasetRoot, $Wrapper) {
  $env:KDG_COCO_DATA_DIR = $DatasetRoot
  $env:KDG_COCO_IMAGE_SUBDIR = ""
  $env:KDG_NUM_CLASSES = "2"
  $env:KDG_VALID_IDS = "1,2"
  $env:KDG_CLASS_NAMES = "person,car"
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

function Train-CenterNet($Exp, $DatasetRoot, $Wrapper, $Epochs, $LoadModel, $LogName) {
  Set-CommonEnv $DatasetRoot $Wrapper
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

function Eval-CenterNet($Exp, $DatasetRoot, $Wrapper, $Model, $LogName) {
  Set-CommonEnv $DatasetRoot $Wrapper
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

Write-Log "Yeoju pipeline with LLVIP illumination started"
Write-Log "Project=$Project"
Write-Log "Yeoju split=$YeojuSplit"
Write-Log "Illumination=$IlluminationCkpt"

if (-not $SkipSplit) {
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

if ($RunRawBaseline) {
  Train-CenterNet $RawBaseExp $YeojuSplit "none" $FilteredEpochs $CocoPretrained "$RawBaseExp.txt"
  $rawModel = Join-Path $RunRoot "$RawBaseExp\model_last.pth"
  Eval-CenterNet "${RawBaseExp}_eval" $YeojuSplit "none" $rawModel "${RawBaseExp}_eval.txt"
}

if (-not $SkipEndToEnd) {
  Train-CenterNet $E2EExp $YeojuSplit "paper_filter" $EndToEndEpochs $CocoPretrained "$E2EExp.txt"
  $e2eModel = Join-Path $RunRoot "$E2EExp\model_last.pth"
  Eval-CenterNet "${E2EExp}_eval" $YeojuSplit "paper_filter" $e2eModel "${E2EExp}_eval.txt"
}

$E2EModelForFiltering = Join-Path $RunRoot "$E2EExp\model_last.pth"
if (-not $SkipFilterImages) {
  $args = @(
    "D:\KDG\generate_paper_filtered_images.py",
    "--src", $Src,
    "--input-root", $YeojuSplit,
    "--output-root", $FilteredRoot,
    "--checkpoint", $E2EModelForFiltering,
    "--illumination-ckpt", $IlluminationCkpt,
    "--num-classes", "2",
    "--gamma-range", "$GammaRange",
    "--sharpness-max", "$SharpnessMax"
  )
  Invoke-Step "generate filtered images" $Python $args (Join-Path $LogRoot "generate_filtered_images.txt")
}

if (-not $SkipFilteredCenterNet) {
  Train-CenterNet $FilteredExp $FilteredRoot "none" $FilteredEpochs $CocoPretrained "$FilteredExp.txt"
  $filteredModel = Join-Path $RunRoot "$FilteredExp\model_last.pth"
  Eval-CenterNet "${FilteredExp}_eval" $FilteredRoot "none" $filteredModel "${FilteredExp}_eval.txt"
}

"name,exp,AP,AP50,AP75,AR100" | Out-File -FilePath $SummaryCsv -Encoding utf8
Append-Summary "raw_baseline" $RawBaseExp "${RawBaseExp}_eval.txt"
Append-Summary "e2e_paper_filter" $E2EExp "${E2EExp}_eval.txt"
Append-Summary "filtered_image_centernet" $FilteredExp "${FilteredExp}_eval.txt"

Write-Log "Finished"
Write-Log "Summary: $SummaryCsv"
Write-Host ""
Write-Host "Summary: $SummaryCsv"
Write-Host "Filtered images: $FilteredRoot"
