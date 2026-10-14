$ErrorActionPreference = "Stop"

$DeployDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExamplePath = Join-Path $DeployDir ".env.example"
$EnvPath = Join-Path $DeployDir ".env"

if (Test-Path -LiteralPath $EnvPath) {
    Write-Host ".env already exists; leaving it unchanged."
    exit 0
}

function New-RandomSecret {
    $bytes = New-Object byte[] 32
    $generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $generator.GetBytes($bytes)
    }
    finally {
        $generator.Dispose()
    }
    return (($bytes | ForEach-Object { $_.ToString("x2") }) -join "")
}

$content = [System.IO.File]::ReadAllText($ExamplePath)
$content = $content.Replace(
    "WEBUI_SECRET_KEY=replace-with-a-random-secret",
    "WEBUI_SECRET_KEY=$(New-RandomSecret)"
)
$content = $content.Replace(
    "OS_AGENT_MEMORY_API_KEY=replace-with-a-different-random-secret",
    "OS_AGENT_MEMORY_API_KEY=$(New-RandomSecret)"
)
[System.IO.File]::WriteAllText(
    $EnvPath,
    $content,
    [System.Text.UTF8Encoding]::new($false)
)
Write-Host "Created deploy/open-webui/.env with random local secrets."
