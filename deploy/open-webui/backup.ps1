param(
    [string]$BackupDirectory = ""
)

$ErrorActionPreference = "Stop"
$DeployDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $BackupDirectory) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $BackupDirectory = Join-Path $DeployDir "backup-$stamp"
}
$BackupDirectory = [System.IO.Path]::GetFullPath($BackupDirectory)
[System.IO.Directory]::CreateDirectory($BackupDirectory) | Out-Null

function Read-EnvValue([string]$Name, [string]$Default) {
    $envPath = Join-Path $DeployDir ".env"
    if (Test-Path -LiteralPath $envPath) {
        foreach ($line in [System.IO.File]::ReadAllLines($envPath)) {
            if ($line.StartsWith("$Name=")) {
                $value = $line.Substring($Name.Length + 1).Trim()
                if ($value) { return $value }
            }
        }
    }
    return $Default
}

function Export-DockerVolume([string]$Volume, [string]$Archive) {
    docker volume inspect $Volume *> $null
    if ($LASTEXITCODE -ne 0) { throw "Docker volume does not exist: $Volume" }
    docker run --rm `
        --mount "type=volume,src=$Volume,dst=/source,readonly" `
        --mount "type=bind,src=$BackupDirectory,dst=/backup" `
        --entrypoint sh `
        osagent-memory-web:0.1.0 `
        -c "tar -C /source -czf /backup/$Archive ."
    if ($LASTEXITCODE -ne 0) { throw "Failed to export Docker volume: $Volume" }
}

$webuiVolume = Read-EnvValue "OPEN_WEBUI_DATA_VOLUME" "osagent-open-webui-data"
$memoryVolume = Read-EnvValue "OS_AGENT_MEMORY_DATA_VOLUME" "osagent-memory-data"

Push-Location $DeployDir
try {
    docker compose stop
    if ($LASTEXITCODE -ne 0) { throw "Unable to stop the Compose services" }
    Export-DockerVolume $webuiVolume "open-webui-data.tar.gz"
    Export-DockerVolume $memoryVolume "osagent-memory-data.tar.gz"
}
finally {
    docker compose start
    Pop-Location
}

Write-Host "Backup written to $BackupDirectory"
