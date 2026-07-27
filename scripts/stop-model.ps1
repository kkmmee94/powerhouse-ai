[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pidPath = Join-Path $projectRoot "logs\model-server.pid"
$expectedRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $projectRoot "tools\llama-cuda")
).TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar

if (-not (Test-Path -LiteralPath $pidPath)) {
    Write-Host "No saved model-server process was found."
    exit 0
}

$serverPid = [int](Get-Content -LiteralPath $pidPath -Raw)
$process = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
if (-not $process) {
    Write-Host "The saved model-server process is no longer running."
    Remove-Item -LiteralPath $pidPath
    exit 0
}

$actualPath = [System.IO.Path]::GetFullPath($process.Path)
if (-not $actualPath.StartsWith($expectedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to stop PID $serverPid because its executable is outside $expectedRoot."
}

Stop-Process -Id $serverPid
Remove-Item -LiteralPath $pidPath
Write-Host "Stopped model-server process $serverPid."
