[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$uvExecutable = @(
    (Get-Command uv -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
    (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe")
) |
    Where-Object { $_ -and (Test-Path -LiteralPath $_) } |
    Select-Object -First 1

if (-not $uvExecutable) {
    throw "uv was not found. Install it with: winget install --exact --id astral-sh.uv"
}

Set-Location -LiteralPath $projectRoot
& $uvExecutable sync --extra dev
if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
}
& $uvExecutable run python -m pytest
Write-Host "Powerhouse AI bootstrap completed."
