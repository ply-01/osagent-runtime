$ErrorActionPreference = 'Stop'
$key = "$env:USERPROFILE/.ssh/galaxy_kylin_osagent"
$remote = '/home/veritas/.cache/osagent-conflict-contract-v2-20260911'
& ssh -o BatchMode=yes -o ConnectTimeout=8 -i $key -p 2223 veritas@127.0.0.1 "env OS_AGENT_MEMORY_EXTRACTOR=rules OS_AGENT_EVAL_WORKERS=1 /home/veritas/.local/share/osagent/venv/bin/python $remote/benchmarks/evaluate_isolated_state_v2.py --dataset-root /home/veritas/.cache/osagent-v11-eval-20260909/data/benchmarks/kylin-memory-eval-v5 --annotations $remote/annotations/annotations.json --output $remote/results/isolated-state-v2.json"
if ($LASTEXITCODE -ne 0) { throw 'V11 evaluation failed' }
& scp -q -i $key -P 2223 "veritas@127.0.0.1:$remote/results/isolated-state-v2.json" 'doc/evaluation/results/v11-isolated-state-v2-20260911.json'
if ($LASTEXITCODE -ne 0) { throw 'V11 result transfer failed' }
