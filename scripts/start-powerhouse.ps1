[CmdletBinding()]
param(
    [ValidateSet("speed", "balanced", "max")]
    [string]$Profile = "balanced",
    [ValidateRange(2048, 32768)]
    [int]$ContextTokens = 8192
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "start-model.ps1") `
    -Profile $Profile `
    -ContextTokens $ContextTokens
& (Join-Path $PSScriptRoot "start-api.ps1")

Write-Host ""
Write-Host "Powerhouse AI is fully running."
Write-Host "Status: .\.venv\Scripts\powerhouse.exe status"
