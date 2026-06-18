$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/leeseonggyu1/CapstoneDesign.git"
$DefaultBranch = "main"
$RepoPath = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$SafeRepoPath = $RepoPath -replace "\\", "/"

git config --global --add safe.directory $SafeRepoPath

Push-Location $RepoPath
try {
  if (-not (Test-Path -LiteralPath ".git")) {
    git -C $RepoPath init
  }

  git -C $RepoPath config user.name "leeseonggyu1"
  git -C $RepoPath config user.email "leeseonggyu1@users.noreply.github.com"

  $RemoteNames = git -C $RepoPath remote
  if ($RemoteNames -notcontains "origin") {
    git -C $RepoPath remote add origin $RepoUrl
  } else {
    $Remote = git -C $RepoPath remote get-url origin
    if ($Remote -ne $RepoUrl) {
    git -C $RepoPath remote set-url origin $RepoUrl
    }
  }

  git -C $RepoPath add .
  $Changes = git -C $RepoPath status --porcelain
  if ($Changes) {
    git -C $RepoPath commit -m "Add capstone experiment materials"
  } else {
    Write-Host "No new changes to commit."
  }

  git -C $RepoPath branch -M $DefaultBranch
  git -C $RepoPath push -u origin $DefaultBranch
}
finally {
  Pop-Location
}
