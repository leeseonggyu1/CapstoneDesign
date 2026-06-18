param(
  [ValidateSet("llvip", "yeoju", "both")]
  [string]$Dataset = "yeoju",
  [int]$Epochs = 10,
  [int]$BatchSize = 2,
  [double]$Lr = 0.000125,
  [string]$Gpus = "0"
)

$ErrorActionPreference = "Stop"

$Python = "C:\Users\owner\.conda\envs\CenterNet\python.exe"
$Project = Get-ChildItem -LiteralPath "D:\KDG" -Directory |
  Where-Object {
    $_.Name -like "CenterNet_Origin - *" -and
    -not (Test-Path -LiteralPath (Join-Path $_.FullName "gamma_values_epoch_None.txt"))
  } |
  Select-Object -First 1 -ExpandProperty FullName
$Src = Join-Path $Project "src"
$Pretrained = Join-Path $Project "models\ctdet_coco_hg.pth"
$LogDir = "D:\KDG\predecessor_gap_logs"

if (-not $Project) {
  throw "predecessor CenterNet project not found"
}
if (-not (Test-Path -LiteralPath $Python)) {
  throw "Python not found: $Python"
}
if (-not (Test-Path -LiteralPath (Join-Path $Src "main.py"))) {
  throw "main.py not found under: $Src"
}
if (-not (Test-Path -LiteralPath $Pretrained)) {
  throw "pretrained model not found: $Pretrained"
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Set-GapDatasetEnv {
  param([string]$Name)

  if ($Name -eq "llvip") {
    $env:KDG_COCO_DATA_DIR = "D:\KDG\llvip_GAP_filtering\coco_llvip_rgb"
    $env:KDG_COCO_IMAGE_SUBDIR = ""
    $env:KDG_NUM_CLASSES = "1"
    $env:KDG_VALID_IDS = "1"
    $env:KDG_CLASS_NAMES = "person"
    return "predecessor_llvip_gap_bs${BatchSize}_${Epochs}ep"
  }

  if ($Name -eq "yeoju") {
    $env:KDG_COCO_DATA_DIR = "D:\KDG\Yeoju_rain_filtering\coco_llvip_rgb"
    $env:KDG_COCO_IMAGE_SUBDIR = "images"
    $env:KDG_NUM_CLASSES = "2"
    $env:KDG_VALID_IDS = "1,2"
    $env:KDG_CLASS_NAMES = "person,car"
    return "predecessor_yeoju_gap_bs${BatchSize}_${Epochs}ep"
  }

  throw "Unknown dataset: $Name"
}

function Invoke-Train {
  param([string]$Name)

  $ExpId = Set-GapDatasetEnv -Name $Name
  $LogPath = Join-Path $LogDir "$ExpId.txt"

  Write-Host "===== train $ExpId / $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
  Write-Host "project: $Project"
  Write-Host "data: $env:KDG_COCO_DATA_DIR"
  Write-Host "classes: $env:KDG_CLASS_NAMES"
  Write-Host "log: $LogPath"

  Push-Location $Src
  try {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python main.py `
      --exp_id $ExpId `
      --arch hourglass `
      --gpus $Gpus `
      --debug 0 `
      --batch_size $BatchSize `
      --num_epochs $Epochs `
      --lr $Lr `
      --val_intervals 5 `
      --load_model $Pretrained 2>&1 | Tee-Object -FilePath $LogPath
    $ErrorActionPreference = $previousErrorActionPreference

    if ($LASTEXITCODE -ne 0) {
      throw "training failed: $ExpId"
    }
  }
  finally {
    if ($previousErrorActionPreference) {
      $ErrorActionPreference = $previousErrorActionPreference
    }
    Pop-Location
  }
}

if ($Dataset -eq "both") {
  Invoke-Train -Name "llvip"
  Invoke-Train -Name "yeoju"
} else {
  Invoke-Train -Name $Dataset
}
