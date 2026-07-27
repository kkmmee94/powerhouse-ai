[CmdletBinding()]
param(
    [ValidateSet("speed", "balanced", "max")]
    [string]$Profile = "balanced",
    [ValidateRange(2048, 32768)]
    [int]$ContextTokens = 8192
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$server = Join-Path $projectRoot "tools\llama-cuda\llama-server.exe"
$logDirectory = Join-Path $projectRoot "logs"

if (-not (Test-Path -LiteralPath $server)) {
    throw "CUDA llama-server was not found at $server. Run scripts\install-llama-cuda.ps1."
}

$profiles = @{
    speed = "Q4_K_M"
    balanced = "Q6_K"
    max = "Q8_0"
}
$quantization = $profiles[$Profile]
$localModel = Join-Path $projectRoot "models\Qwen_Qwen3.5-9B-$quantization.gguf"

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$stdoutLog = Join-Path $logDirectory "model-server.out.log"
$stderrLog = Join-Path $logDirectory "model-server.err.log"
$pidPath = Join-Path $logDirectory "model-server.pid"

$existing = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    throw "Port 8080 is already in use by process $($existing.OwningProcess)."
}

if (
    -not (Test-Path -LiteralPath $localModel) -or
    (Test-Path -LiteralPath "$localModel.aria2")
) {
    throw "Verified model file not found. Run scripts\download-model.ps1 -Profile $Profile first."
}

$arguments = @(
    "--model", $localModel,
    "--alias", "local-model",
    "--api-key", "local",
    "--host", "127.0.0.1",
    "--port", "8080",
    "--ctx-size", "$ContextTokens",
    "--n-gpu-layers", "99",
    "--flash-attn", "on",
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "--parallel", "1"
)
$quotedArguments = $arguments | ForEach-Object {
    if ($_ -match "\s") {
        '"' + $_ + '"'
    }
    else {
        $_
    }
}

$process = Start-Process `
    -FilePath $server `
    -ArgumentList ($quotedArguments -join " ") `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Set-Content -LiteralPath $pidPath -Value $process.Id
Write-Host "Started Qwen3.5-9B $quantization as process $($process.Id)."
Write-Host "Logs: $stderrLog"

for ($attempt = 0; $attempt -lt 1440; $attempt++) {
    if ($process.HasExited) {
        Get-Content -LiteralPath $stderrLog -Tail 60
        throw "Model server exited with code $($process.ExitCode)."
    }
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8080/health" -TimeoutSec 2
        if ($response.status -eq "ok") {
            Write-Host "Model server is ready at http://127.0.0.1:8080."
            return
        }
    }
    catch {
        # Downloading or loading. Progress is available in the server log.
    }
    Start-Sleep -Seconds 5
}

throw "Model server did not become ready within two hours. Check $stderrLog."
