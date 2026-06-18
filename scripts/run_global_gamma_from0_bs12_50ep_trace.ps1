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
$RunRoot = "D:\KDG\global_gamma_trace_runs"
$LoadModel = Join-Path $Project "exp\ctdet\test\model_last.pth"
$Exp = "global_gamma_from0_bs12_50ep_trace"

if (-not (Test-Path -LiteralPath $Python)) {
  throw "Missing Python: $Python"
}
if (-not (Test-Path -LiteralPath (Join-Path $Src "main.py"))) {
  throw "Missing main.py: $Src"
}
if (-not (Test-Path -LiteralPath $LoadModel)) {
  throw "Missing baseline CenterNet model: $LoadModel"
}

$env:KDG_GAMMA_WRAPPER = "gamma_net"
$env:KDG_DATASET_CNN_DIP = "0"
$env:KDG_EXP_DIR = $RunRoot
$env:KDG_GAMMA_LOSS_WEIGHT = "0.01"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$SaveDir = Join-Path $RunRoot $Exp
$TrainLogDir = Join-Path $RunRoot "_train_logs"
$TrainLog = Join-Path $TrainLogDir "$Exp.txt"
New-Item -ItemType Directory -Force -Path $SaveDir, (Join-Path $SaveDir "debug"), $TrainLogDir | Out-Null

"===== $Exp / $(Get-Date) =====" | Out-File -FilePath $TrainLog -Encoding utf8
"Project: $Project" | Out-File -FilePath $TrainLog -Encoding utf8 -Append
"Start from baseline CenterNet model, train global GammaNet from epoch 0 to 50, batch=12, workers=8, checkpoint every 5 epochs" |
  Out-File -FilePath $TrainLog -Encoding utf8 -Append

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader |
    Out-File -FilePath $TrainLog -Encoding utf8 -Append
}

Write-Host "Project: $Project"
Write-Host "Output:  $SaveDir"
Write-Host "Log:     $TrainLog"
Write-Host "Start:   baseline CenterNet -> new global GammaNet training"

$PythonArgs = @(
  "$Src\main.py",
  "--exp_id", $Exp,
  "--arch", "hourglass",
  "--gpus", "0",
  "--batch_size", "12",
  "--num_epochs", "50",
  "--lr", "1.25e-4",
  "--val_intervals", "0",
  "--print_iter", "20",
  "--num_workers", "8",
  "--save_interval", "5",
  "--load_model", $LoadModel
)

Push-Location "D:\KDG"
try {
  & $Python @PythonArgs 2>&1 | Tee-Object -FilePath $TrainLog -Append
  $ExitCode = $LASTEXITCODE
}
finally {
  Pop-Location
}

exit $ExitCode
