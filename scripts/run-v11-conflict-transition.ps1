param([Parameter(Mandatory=$true)][string]$Stage, [Parameter(Mandatory=$true)][string]$OutputPath)
$ErrorActionPreference = 'Stop'
$key = Join-Path $env:USERPROFILE '.ssh/galaxy_kylin_osagent'
$remote = 'veritas@127.0.0.1'
$root = "/home/veritas/.cache/$Stage"
& ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -i $key -p 2223 $remote "OS_AGENT_MEMORY_EXTRACTOR=rules OS_AGENT_EVAL_WORKERS=1 /home/veritas/.local/share/osagent/venv/bin/python $root/benchmarks/conflict_transition_eval.py --dataset-root /home/veritas/.cache/osagent-v11-eval-20260909/data/benchmarks/kylin-memory-eval-v5 --output $root/conflict-transition.json"
if ($LASTEXITCODE -ne 0) { throw 'Conflict evaluation failed.' }
& scp -q -i $key -P 2223 "${remote}:$root/conflict-transition.json" $OutputPath
if ($LASTEXITCODE -ne 0) { throw 'Aggregate download failed.' }
