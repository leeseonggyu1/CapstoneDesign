$ErrorActionPreference = "Stop"

$Python = "C:\Users\owner\.conda\envs\CenterNet\python.exe"
$Project = Get-ChildItem -Path "D:\KDG" -Directory |
  Where-Object {
    $_.Name -like "CenterNet_Origin -*" -and
    (Test-Path -LiteralPath (Join-Path $_.FullName "src\test.py"))
  } |
  Select-Object -First 1 -ExpandProperty FullName

if (-not $Project) {
  throw "Could not find CenterNet project folder under D:\KDG"
}

$Src = Join-Path $Project "src"
$RunRoot = "D:\KDG\global_gamma_trace_runs"
$TrainExp = "global_gamma_from0_bs12_50ep_trace"
$EvalLogDir = "D:\KDG\global_gamma_from0_trace_eval_logs"
New-Item -ItemType Directory -Force -Path $EvalLogDir | Out-Null

if (-not (Test-Path -LiteralPath $Python)) {
  throw "Missing Python: $Python"
}

$env:KDG_GAMMA_WRAPPER = "gamma_net"
$env:KDG_DATASET_CNN_DIP = "0"
$env:KDG_EXP_DIR = $RunRoot
$env:KDG_GAMMA_LOSS_WEIGHT = "0.01"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONUNBUFFERED = "1"

$Checkpoints = 5,10,15,20,25,30,35,40,45,50 | ForEach-Object { "model_$_.pth" }

Write-Host "Project: $Project"
Write-Host "Eval logs: $EvalLogDir"

foreach ($ckpt in $Checkpoints) {
  $LoadModel = Join-Path $RunRoot "$TrainExp\$ckpt"
  if (-not (Test-Path -LiteralPath $LoadModel)) {
    Write-Host "skip missing $LoadModel"
    continue
  }

  $ModelName = [System.IO.Path]::GetFileNameWithoutExtension($ckpt)
  $EvalExp = "eval_${TrainExp}_${ModelName}"
  $EvalLog = Join-Path $EvalLogDir "$EvalExp.txt"

  "===== $EvalExp / $(Get-Date) =====" | Out-File -FilePath $EvalLog -Encoding utf8

  $PythonArgs = @(
    "$Src\test.py",
    "--test",
    "--not_prefetch_test",
    "--gpus", "0",
    "--exp_id", $EvalExp,
    "--load_model", $LoadModel
  )

  Write-Host "Evaluating $ckpt"
  Push-Location "D:\KDG"
  try {
    & $Python @PythonArgs 2>&1 | Tee-Object -FilePath $EvalLog -Append
  }
  finally {
    Pop-Location
  }
}
