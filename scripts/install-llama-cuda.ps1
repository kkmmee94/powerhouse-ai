[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$downloadDirectory = Join-Path $projectRoot "tools\downloads"
$runtimeDirectory = Join-Path $projectRoot "tools\llama-cuda"
$release = "b10107"
$baseUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$release"
$runtimeZip = Join-Path $downloadDirectory "llama-cuda.zip"
$cudaZip = Join-Path $downloadDirectory "cudart.zip"
$expectedHashes = @{
    $runtimeZip = "1E43BBEC9691CD0BC636603C366769148FA6265FD261C5F7C67050B450BBC237"
    $cudaZip = "8C79A9B226DE4B3CACFD1F83D24F962D0773BE79F1E7B75C6AF4DED7E32AE1D6"
}

function Assert-Sha256 {
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        [Parameter(Mandatory)]
        [string]$Expected
    )

    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if ($actual -ne $Expected) {
        throw "SHA-256 mismatch for $Path. Expected $Expected but received $actual."
    }
}

New-Item -ItemType Directory -Force -Path $downloadDirectory, $runtimeDirectory | Out-Null

if (-not (Test-Path -LiteralPath $runtimeZip)) {
    Invoke-WebRequest `
        -Uri "$baseUrl/llama-$release-bin-win-cuda-12.4-x64.zip" `
        -OutFile $runtimeZip
}
if (-not (Test-Path -LiteralPath $cudaZip)) {
    Invoke-WebRequest `
        -Uri "$baseUrl/cudart-llama-bin-win-cuda-12.4-x64.zip" `
        -OutFile $cudaZip
}

Assert-Sha256 -Path $runtimeZip -Expected $expectedHashes[$runtimeZip]
Assert-Sha256 -Path $cudaZip -Expected $expectedHashes[$cudaZip]

Expand-Archive -LiteralPath $runtimeZip -DestinationPath $runtimeDirectory -Force
Expand-Archive -LiteralPath $cudaZip -DestinationPath $runtimeDirectory -Force

$server = Join-Path $runtimeDirectory "llama-server.exe"
if (-not (Test-Path -LiteralPath $server)) {
    throw "Installation finished without producing $server."
}

Write-Host "CUDA llama.cpp installed at $runtimeDirectory"
& $server --version
