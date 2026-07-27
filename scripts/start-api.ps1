[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$executable = Join-Path $projectRoot ".venv\Scripts\powerhouse.exe"
$logDirectory = Join-Path $projectRoot "logs"
$stdoutLog = Join-Path $logDirectory "api.out.log"
$stderrLog = Join-Path $logDirectory "api.err.log"
$pidPath = Join-Path $logDirectory "api.pid"

if (-not (Test-Path -LiteralPath $executable)) {
    throw "Powerhouse is not installed in .venv. Run scripts\bootstrap.ps1 first."
}
$existing = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    throw "Port 8000 is already in use by process $($existing.OwningProcess)."
}

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$process = Start-Process `
    -FilePath $executable `
    -ArgumentList @("serve", "--port", "8000") `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru
Set-Content -LiteralPath $pidPath -Value $process.Id

for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if ($process.HasExited) {
        Get-Content -LiteralPath $stderrLog -Tail 60
        throw "API exited with code $($process.ExitCode)."
    }
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2
        if ($response.status -eq "ok") {
            Write-Host "Powerhouse API is ready at http://127.0.0.1:8000."
            Write-Host "Interactive API documentation: http://127.0.0.1:8000/docs"
            return
        }
    }
    catch {
        # The API is still starting.
    }
    Start-Sleep -Milliseconds 500
}

throw "API did not become ready. Check $stderrLog."
