param(
    [Parameter(Mandatory = $true)]
    [string]$BackupDirectory,
    [string]$Suffix = ""
)

$ErrorActionPreference = "Stop"
$BackupDirectory = [System.IO.Path]::GetFullPath($BackupDirectory)
if (-not $Suffix) { $Suffix = Get-Date -Format "yyyyMMdd-HHmmss" }

$archives = @{
    "osagent-open-webui-data-restored-$Suffix" = "open-webui-data.tar.gz"
    "osagent-memory-data-restored-$Suffix" = "osagent-memory-data.tar.gz"
}

foreach ($entry in $archives.GetEnumerator()) {
    $volume = $entry.Key
    $archive = $entry.Value
    $archivePath = Join-Path $BackupDirectory $archive
    if (-not (Test-Path -LiteralPath $archivePath)) {
        throw "Archive does not exist: $archivePath"
    }
    docker volume inspect $volume *> $null
    if ($LASTEXITCODE -eq 0) {
        throw "Refusing to overwrite existing Docker volume: $volume"
    }
    docker volume create $volume | Out-Null
    docker run --rm `
        --mount "type=volume,src=$volume,dst=/target" `
        --mount "type=bind,src=$BackupDirectory,dst=/backup,readonly" `
        --entrypoint sh `
        osagent-memory-web:0.1.0 `
        -c "tar -C /target -xzf /backup/$archive"
    if ($LASTEXITCODE -ne 0) { throw "Failed to restore $archive" }
}

Write-Host "Restore completed into new volumes. Set these values in .env:"
Write-Host "OPEN_WEBUI_DATA_VOLUME=osagent-open-webui-data-restored-$Suffix"
Write-Host "OS_AGENT_MEMORY_DATA_VOLUME=osagent-memory-data-restored-$Suffix"
