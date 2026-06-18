param(
  [int]$Epochs = 10,
  [int]$BatchSize = 64,
  [double]$Lr = 0.001,
  [int]$ImageSize = 512,
  [int]$Workers = 4
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
$DatasetRoot = "D:\KDG\llvip\coco_llvip_rgb"
$Labels = "D:\KDG\llvip_gamma_brightness\fixed_merged_predictions.txt"
$OutDir = "D:\KDG\paper_filter_runs\illumination_llvip"
$LogDir = Join-Path $OutDir "_logs"
$Log = Join-Path $LogDir "pretrain_illumination_llvip.txt"

if (-not (Test-Path -LiteralPath $Python)) {
  throw "Missing Python: $Python"
}
if (-not (Test-Path -LiteralPath (Join-Path $Src "main.py"))) {
  throw "Missing main.py: $Src"
}
if (-not (Test-Path -LiteralPath $DatasetRoot)) {
  throw "Missing LLVIP dataset: $DatasetRoot"
}
if (-not (Test-Path -LiteralPath $Labels)) {
  throw "Missing day/night labels: $Labels"
}

New-Item -ItemType Directory -Force -Path $OutDir, $LogDir | Out-Null

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONUNBUFFERED = "1"

"===== illumination pretrain / $(Get-Date) =====" | Out-File -FilePath $Log -Encoding utf8
"Dataset: $DatasetRoot" | Out-File -FilePath $Log -Encoding utf8 -Append
"Labels:  $Labels" | Out-File -FilePath $Log -Encoding utf8 -Append
"Config:  paper random 7:3 split, epochs=$Epochs batch=$BatchSize lr=$Lr image_size=$ImageSize workers=$Workers" |
  Out-File -FilePath $Log -Encoding utf8 -Append

Write-Host "Step 1/2: pretrain day-night illumination classifier"
Write-Host "Split:   paper random 7:3 over LLVIP"
Write-Host "Project: $Project"
Write-Host "Data:    $DatasetRoot"
Write-Host "Labels:  $Labels"
Write-Host "Output:  $OutDir"
Write-Host "Log:     $Log"

$PythonArgs = @(
  "D:\KDG\pretrain_illumination_classifier.py",
  "--src", $Src,
  "--dataset-root", $DatasetRoot,
  "--labels", $Labels,
  "--output-dir", $OutDir,
  "--epochs", "$Epochs",
  "--batch-size", "$BatchSize",
  "--lr", "$Lr",
  "--image-size", "$ImageSize",
  "--num-workers", "$Workers"
)

Push-Location "D:\KDG"
try {
  & $Python @PythonArgs 2>&1 | Tee-Object -FilePath $Log -Append
  $ExitCode = $LASTEXITCODE
}
finally {
  Pop-Location
}

exit $ExitCode
