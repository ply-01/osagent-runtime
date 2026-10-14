param(
    [switch]$Build
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $repoRoot 'vendor/open-webui'
$overlayFiles = @(
	'src/routes/(app)/workspace/+layout.svelte',
	'src/routes/(app)/workspace/plugins/+page.svelte',
	'vite.config.ts'
)

if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot 'package.json'))) {
    throw 'vendor/open-webui is missing; run scripts/sync-openwebui-source.sh first.'
}

Push-Location $sourceRoot
try {
    & npx.cmd prettier --check @overlayFiles
    if ($LASTEXITCODE -ne 0) {
        throw 'Prettier failed for the OSagent Open WebUI overlay.'
    }

    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $checkOutput = @(& npx.cmd svelte-check --output machine 2>&1 | ForEach-Object { "$_" })
        $globalExit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
    }
    $overlayDiagnostics = @(
        $checkOutput | Where-Object {
            $line = $_.Replace('\', '/')
            ($overlayFiles | Where-Object { $line.Contains($_) }).Count -gt 0
        }
    )
    if ($overlayDiagnostics.Count -gt 0) {
        $overlayDiagnostics | Write-Error
        throw "svelte-check reported $($overlayDiagnostics.Count) diagnostics in OSagent overlay files."
    }

    if ($globalExit -ne 0) {
        Write-Warning 'The pinned Open WebUI upstream still has global svelte-check debt; no diagnostic belongs to the OSagent overlay.'
    }
    Write-Host 'OSagent Open WebUI overlay diagnostics: 0'

    if ($Build) {
        & npx.cmd vite build
        if ($LASTEXITCODE -ne 0) {
            throw 'The Open WebUI production build failed.'
        }
    }
}
finally {
    Pop-Location
}
