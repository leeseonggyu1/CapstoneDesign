$ErrorActionPreference = "Stop"

$Python = "C:\Users\owner\.conda\envs\CenterNet\python.exe"
$RunRoot = "D:\KDG\global_gamma_runs"
$TrainExp = "global_gamma_bs8_64to120_lr5e5"
$TrainDir = Join-Path $RunRoot $TrainExp
$ModelLast = Join-Path $TrainDir "model_last.pth"
$Model50 = Join-Path $TrainDir "model_50.pth"
$TrainLog = Join-Path $RunRoot "_train_logs\$TrainExp.txt"
$EvalScript = "D:\KDG\eval_global_gamma_64to120_checkpoints.ps1"
$GammaScript = "D:\KDG\visualize_global_gamma_checkpoints.py"
$SummaryScript = "D:\KDG\summarize_global_gamma_eval_logs.py"
$ReportScript = "D:\KDG\build_global_gamma_epoch_report.py"
$WatchLogDir = "D:\KDG\global_gamma_64to120_watch_logs"
$WatchLog = Join-Path $WatchLogDir "watch_eval_and_gamma.txt"
New-Item -ItemType Directory -Force -Path $WatchLogDir | Out-Null

function Log-Line($Message) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
  Write-Host $line
  $line | Out-File -FilePath $WatchLog -Encoding utf8 -Append
}

function Get-TrainingProcess {
  Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object {
      $_.CommandLine -like "*main.py*" -and
      $_.CommandLine -like "*$TrainExp*"
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
  throw "Missing Python: $Python"
}
if (-not (Test-Path -LiteralPath $EvalScript)) {
  throw "Missing eval script: $EvalScript"
}
if (-not (Test-Path -LiteralPath $GammaScript)) {
  throw "Missing gamma visualization script: $GammaScript"
}
if (-not (Test-Path -LiteralPath $ReportScript)) {
  throw "Missing report script: $ReportScript"
}

"===== watcher start / $(Get-Date) =====" | Out-File -FilePath $WatchLog -Encoding utf8
Log-Line "Waiting for training process to finish: $TrainExp"

$StableCount = 0
$LastSignature = ""
$ObservedTraining = $false
while ($true) {
  $trainProcess = @(Get-TrainingProcess)
  if ($trainProcess.Count -gt 0) {
    $ObservedTraining = $true
    Log-Line "training is still running; pid=$($trainProcess[0].ProcessId)"
    Start-Sleep -Seconds 60
    continue
  }

  if (-not $ObservedTraining -and -not (Test-Path -LiteralPath $Model50)) {
    Log-Line "training process not seen yet, and model_50.pth is not ready"
    Start-Sleep -Seconds 60
    continue
  }

  if ($ObservedTraining -and -not (Test-Path -LiteralPath $Model50)) {
    Log-Line "training process ended before model_50.pth was created; stop to avoid partial evaluation"
    exit 1
  }

  if (-not (Test-Path -LiteralPath $ModelLast)) {
    Log-Line "model_last.pth not found yet"
    Start-Sleep -Seconds 60
    continue
  }

  $modelItem = Get-Item -LiteralPath $ModelLast
  $logItem = $null
  if (Test-Path -LiteralPath $TrainLog) {
    $logItem = Get-Item -LiteralPath $TrainLog
  }

  if ($logItem -and ($modelItem.LastWriteTime -lt $logItem.LastWriteTime.AddMinutes(-1))) {
    Log-Line "old model_last detected; waiting for this run to update it"
    Start-Sleep -Seconds 60
    continue
  }

  $signature = "{0}|{1}" -f $modelItem.Length, $modelItem.LastWriteTimeUtc.Ticks
  if ($signature -eq $LastSignature) {
    $StableCount += 1
  } else {
    $StableCount = 0
    $LastSignature = $signature
  }

  if ($StableCount -ge 2) {
    Log-Line "training is finished and final checkpoint looks stable"
    break
  }

  Log-Line "model_last found; waiting until file is stable"
  Start-Sleep -Seconds 60
}

Log-Line "Running checkpoint evaluation"
powershell -NoProfile -ExecutionPolicy Bypass -File $EvalScript 2>&1 |
  Tee-Object -FilePath $WatchLog -Append

Log-Line "Summarizing evaluation logs"
& $Python $SummaryScript 2>&1 |
  Tee-Object -FilePath $WatchLog -Append

Log-Line "Visualizing gamma changes"
& $Python $GammaScript `
  --run-root $RunRoot `
  --exp $TrainExp `
  --image-dir "D:\KDG\Yeoju_rain\coco_llvip_rgb\val2017" `
  --out-dir "D:\KDG\global_gamma_64to120_gamma_visuals" `
  --sample-count 0 `
  --batch-size 32 2>&1 |
  Tee-Object -FilePath $WatchLog -Append

Log-Line "Building combined gamma/AP report"
& $Python $ReportScript 2>&1 |
  Tee-Object -FilePath $WatchLog -Append

Log-Line "Done"
