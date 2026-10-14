param([Parameter(Mandatory=$true)][string]$OutputPath)
$ErrorActionPreference = 'Stop'
$key = Join-Path $env:USERPROFILE '.ssh/galaxy_kylin_osagent'
$remote = 'veritas@127.0.0.1'
$stage = '/home/veritas/.cache/osagent-v11-latency-500-20260910'
& ssh -o BatchMode=yes -o ConnectTimeout=10 -i $key -p 2223 $remote "bash $stage/run-native-memory-eval.sh $stage results-candidate"
if ($LASTEXITCODE -ne 0) { throw 'V11 evaluation failed; inspect sanitized status only.' }
& scp -q -i $key -P 2223 "${remote}:$stage/results-candidate/round-1.json" $OutputPath
if ($LASTEXITCODE -ne 0) { throw 'Unable to retrieve aggregate report.' }
