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
$RunRoot = "D:\KDG\global_gamma_runs"
$LoadModel = Join-Path $Project "exp\ctdet\global_gamma_bs8_more64ep\model_last.pth"
$Exp = "global_gamma_bs8_64to120_lr5e5"

if (-not (Test-Path -LiteralPath $Python)) {
  throw "Missing Python: $Python"
}

if (-not (Test-Path -LiteralPath (Join-Path $Src "main.py"))) {
  throw "Missing main.py: $Src"
}

if (-not (Test-Path -LiteralPath $LoadModel)) {
  throw "Missing 64epoch global gamma model: $LoadModel"
}

$env:KDG_GAMMA_WRAPPER = "gamma_net"
$env:KDG_DATASET_CNN_DIP = "0"
$env:KDG_EXP_DIR = $RunRoot
$env:KDG_GAMMA_LOSS_WEIGHT = "0.01"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONUNBUFFERED = "1"

$SaveDir = Join-Path $RunRoot $Exp
$TrainLogDir = Join-Path $RunRoot "_train_logs"
$TrainLog = Join-Path $TrainLogDir "$Exp.txt"
New-Item -ItemType Directory -Force -Path $SaveDir, (Join-Path $SaveDir "debug"), $TrainLogDir | Out-Null

"===== $Exp / $(Get-Date) =====" | Out-File -FilePath $TrainLog -Encoding utf8
"Project: $Project" | Out-File -FilePath $TrainLog -Encoding utf8 -Append
"Continue global gamma: 64ep + 56ep = 120ep total, lr=5e-5, checkpoint every 10 epochs" |
  Out-File -FilePath $TrainLog -Encoding utf8 -Append

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader |
    Out-File -FilePath $TrainLog -Encoding utf8 -Append
}

Write-Host "Project: $Project"
Write-Host "Output:  $SaveDir"
Write-Host "Log:     $TrainLog"

$PythonArgs = @(
  "$Src\main.py",
  "--exp_id", $Exp,
  "--arch", "hourglass",
  "--gpus", "0",
  "--batch_size", "8",
  "--num_epochs", "56",
  "--lr", "5e-5",
  "--lr_step", "40,50",
  "--val_intervals", "0",
  "--print_iter", "20",
  "--num_workers", "4",
  "--save_interval", "10",
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
