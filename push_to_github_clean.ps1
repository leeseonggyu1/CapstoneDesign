$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/leeseonggyu1/CapstoneDesign.git"
$Branch = "main"
$RepoPath = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$SafeRepoPath = $RepoPath -replace "\\", "/"

Write-Host "Repository folder: $RepoPath"

git config --global --add safe.directory $SafeRepoPath

if (-not (Test-Path -LiteralPath (Join-Path $RepoPath ".git"))) {
  git -C $RepoPath init
}

git -C $RepoPath config user.name "leeseonggyu1"
git -C $RepoPath config user.email "leeseonggyu1@users.noreply.github.com"

$RemoteNames = @(git -C $RepoPath remote)
if ($RemoteNames -contains "origin") {
  git -C $RepoPath remote set-url origin $RepoUrl
} else {
  git -C $RepoPath remote add origin $RepoUrl
}

git -C $RepoPath add --all

$Changes = git -C $RepoPath status --porcelain
if ($Changes) {
  git -C $RepoPath commit -m "Add capstone experiment materials"
} else {
  Write-Host "No new changes to commit."
}

git -C $RepoPath branch -M $Branch
git -C $RepoPath push -u origin $Branch
