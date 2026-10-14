param(
    [string]$LlmConfig = ""
)

$ErrorActionPreference = "Stop"
$DeployDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $DeployDir "../.."))
if (-not $LlmConfig) {
    $LlmConfig = Join-Path $RepoRoot "config/llm.local.json"
}
$EnvPath = Join-Path $DeployDir ".env"

if (-not (Test-Path -LiteralPath $EnvPath)) {
    throw "Run init.ps1 before configuring the model endpoint."
}
if (-not (Test-Path -LiteralPath $LlmConfig)) {
    throw "LLM config does not exist: $LlmConfig"
}

$config = [System.IO.File]::ReadAllText($LlmConfig) | ConvertFrom-Json
$baseUrl = [string]$config.base_url
$apiKey = [string]$config.api_key
$model = [string]$config.model
if (-not $baseUrl -or -not $apiKey -or -not $model) {
    throw "LLM config must contain base_url, api_key, and model."
}

$content = [System.IO.File]::ReadAllText($EnvPath)
function Set-EnvLine([string]$Text, [string]$Name, [string]$Value) {
    if ($Value.Contains("`r") -or $Value.Contains("`n")) {
        throw "$Name must be a single-line value."
    }
    $pattern = "(?m)^" + [regex]::Escape($Name) + "=.*$"
    if (-not [regex]::IsMatch($Text, $pattern)) {
        return $Text.TrimEnd() + "`n$Name=$Value`n"
    }
    return [regex]::Replace(
        $Text,
        $pattern,
        [System.Text.RegularExpressions.MatchEvaluator]{ param($match) "$Name=$Value" }
    )
}

$content = Set-EnvLine $content "OPENAI_API_BASE_URL" $baseUrl.TrimEnd("/")
$content = Set-EnvLine $content "OPENAI_API_KEY" $apiKey
$content = Set-EnvLine $content "OS_AGENT_BASE_URL" $baseUrl.TrimEnd("/")
$content = Set-EnvLine $content "OS_AGENT_API_KEY" $apiKey
$content = Set-EnvLine $content "OS_AGENT_MODEL" $model
[System.IO.File]::WriteAllText(
    $EnvPath,
    $content,
    [System.Text.UTF8Encoding]::new($false)
)
Write-Host "Configured the Open WebUI chat endpoint from OSagent (secret not displayed)."
