$ErrorActionPreference = "Stop"

$Python = "C:\Users\owner\.conda\envs\CenterNet\python.exe"
$RunRoot = "D:\KDG\global_gamma_trace_runs"
$TrainExp = "global_gamma_from0_bs12_50ep_trace"
$TrainDir = Join-Path $RunRoot $TrainExp
$Model50 = Join-Path $TrainDir "model_50.pth"
$TrainLog = Join-Path $RunRoot "_train_logs\$TrainExp.txt"
$EvalScript = "D:\KDG\eval_global_gamma_from0_bs12_50ep_trace_checkpoints.ps1"
$GammaScript = "D:\KDG\visualize_global_gamma_checkpoints.py"
$ReportScript = "D:\KDG\build_global_gamma_from0_trace_report.py"
$WatchLogDir = "D:\KDG\global_gamma_from0_trace_watch_logs"
$WatchLog = Join-Path $WatchLogDir "watch_eval_and_report.txt"
New-Item -ItemType Directory -Force -Path $WatchLogDir | Out-Null

function Log-Line($Message) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
  Write-Host $line
  $line | Out-File -FilePath $WatchLog -Encoding utf8 -Append
}

function Get-TrainingProcess {
  Get-Process python -ErrorAction SilentlyContinue |
    Where-Object {
      $p = $_
      try {
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId = $($p.Id)" -ErrorAction Stop).CommandLine
        $cmd -like "*main.py*" -and $cmd -like "*$TrainExp*"
      } catch {
        $false
      }
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
  throw "Missing Python: $Python"
}

"===== watcher start / $(Get-Date) =====" | Out-File -FilePath $WatchLog -Encoding utf8
Log-Line "Waiting for model_50 checkpoint: $Model50"

$StableCount = 0
$LastSignature = ""
while ($true) {
  $trainProcess = @(Get-TrainingProcess)
  if ($trainProcess.Count -gt 0) {
    Log-Line "training is still running; pid=$($trainProcess[0].Id)"
    Start-Sleep -Seconds 60
    continue
  }

  if (-not (Test-Path -LiteralPath $Model50)) {
    Log-Line "model_50.pth not found yet"
    Start-Sleep -Seconds 60
    continue
  }

  $modelItem = Get-Item -LiteralPath $Model50
  $signature = "{0}|{1}" -f $modelItem.Length, $modelItem.LastWriteTimeUtc.Ticks
  if ($signature -eq $LastSignature) {
    $StableCount += 1
  } else {
    $StableCount = 0
    $LastSignature = $signature
  }

  if ($StableCount -ge 2) {
    Log-Line "model_50 checkpoint looks stable"
    break
  }

  Log-Line "model_50 found; waiting until file is stable"
  Start-Sleep -Seconds 60
}

Log-Line "Running checkpoint evaluation"
powershell -NoProfile -ExecutionPolicy Bypass -File $EvalScript 2>&1 |
  Tee-Object -FilePath $WatchLog -Append

Log-Line "Visualizing gamma changes"
& $Python $GammaScript `
  --run-root $RunRoot `
  --exp $TrainExp `
  --image-dir "D:\KDG\Yeoju_rain\coco_llvip_rgb\val2017" `
  --out-dir "D:\KDG\global_gamma_from0_trace_gamma_visuals" `
  --sample-count 0 `
  --batch-size 32 2>&1 |
  Tee-Object -FilePath $WatchLog -Append

Log-Line "Building trace report"
& $Python $ReportScript 2>&1 |
  Tee-Object -FilePath $WatchLog -Append

Log-Line "Done"
