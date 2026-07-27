[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "stop-api.ps1")
& (Join-Path $PSScriptRoot "stop-model.ps1")
Write-Host "Powerhouse AI is stopped."
