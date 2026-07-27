[CmdletBinding()]
param(
    [ValidateSet("speed", "balanced", "max")]
    [string]$Profile = "balanced"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$modelDirectory = Join-Path $projectRoot "models"
$profiles = @{
    speed = "Q4_K_M"
    balanced = "Q6_K"
    max = "Q8_0"
}
$expectedHashes = @{
    speed = "D784CE9EDA1A5A7B51E8F705A9E6310844BF4F173654D115823C775FDEA56D43"
    balanced = "073A9275E65D9C8CD2819CF5F77B99FBAA6E87BA591DA6BBAA86EC073A64BFEF"
    max = "B58FE056B5435070240DE259F3F981AA38FEE96825BBD78C088D5FD90E46F2B5"
}
$revision = "182be2fd6c7bc44887d88a91cb03ff009cc9f549"
$quantization = $profiles[$Profile]
$fileName = "Qwen_Qwen3.5-9B-$quantization.gguf"
$downloaded = Join-Path $modelDirectory $fileName
$url = (
    "https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF/" +
    "resolve/$revision/$fileName" +
    "?download=true"
)

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

$hfExecutable = @(
    (Join-Path $projectRoot ".venv\Scripts\hf.exe"),
    (Get-Command hf -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
) |
    Where-Object { $_ -and (Test-Path -LiteralPath $_) } |
    Select-Object -First 1

New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null
if ($hfExecutable) {
    $env:HF_HOME = Join-Path $modelDirectory "xet-cache"
    & $hfExecutable download `
        "bartowski/Qwen_Qwen3.5-9B-GGUF" `
        $fileName `
        "--revision" `
        $revision `
        "--local-dir" `
        $modelDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "Hugging Face model download failed with exit code $LASTEXITCODE."
    }
    if (-not (Test-Path -LiteralPath $downloaded)) {
        throw "Download completed without producing $downloaded."
    }
    Assert-Sha256 -Path $downloaded -Expected $expectedHashes[$Profile]
    Write-Host "Downloaded and SHA-256 verified $fileName."
    return
}

$aria = Get-Command aria2c -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $aria) {
    $candidate = Join-Path $env:LOCALAPPDATA (
        "Microsoft\WinGet\Packages\aria2.aria2_Microsoft.Winget.Source_8wekyb3d8bbwe\" +
        "aria2-1.37.0-win-64bit-build1\aria2c.exe"
    )
    if (Test-Path -LiteralPath $candidate) {
        $aria = $candidate
    }
}
if (-not $aria) {
    throw "aria2c was not found. Install it with: winget install --exact --id aria2.aria2"
}

& $aria `
    --continue=true `
    --disable-ipv6=true `
    --max-connection-per-server=8 `
    --split=8 `
    --min-split-size=16M `
    --file-allocation=none `
    "--dir=$modelDirectory" `
    "--out=$fileName" `
    $url

if ($LASTEXITCODE -ne 0) {
    throw "Model download failed with exit code $LASTEXITCODE."
}
Assert-Sha256 -Path $downloaded -Expected $expectedHashes[$Profile]
Write-Host "Downloaded and SHA-256 verified $fileName."
