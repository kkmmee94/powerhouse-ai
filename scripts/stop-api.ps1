[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pidPath = Join-Path $projectRoot "logs\api.pid"
$expectedRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot ".venv")).TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar

if (-not (Test-Path -LiteralPath $pidPath)) {
    Write-Host "No saved API process was found."
    return
}

$apiPid = [int](Get-Content -LiteralPath $pidPath -Raw)
$process = Get-Process -Id $apiPid -ErrorAction SilentlyContinue
if (-not $process) {
    Write-Host "The saved API process is no longer running."
    Remove-Item -LiteralPath $pidPath
    return
}

$actualPath = [System.IO.Path]::GetFullPath($process.Path)
if (-not $actualPath.StartsWith($expectedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to stop PID $apiPid because its executable is outside $expectedRoot."
}

Stop-Process -Id $apiPid
Remove-Item -LiteralPath $pidPath
Write-Host "Stopped API process $apiPid."
