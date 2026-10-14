#!/usr/bin/env bash
set -euo pipefail

stage_root="${1:-$HOME/.cache/osagent-v11-eval-20260909}"
result_name="${2:-results-v11-native-$(date +%Y%m%d-%H%M%S)}"
results="$stage_root/$result_name"
runner="$stage_root/benchmarks/kylin_memory_eval.py"
dataset="$stage_root/data/benchmarks/kylin-memory-eval-v5"
python_bin="$HOME/.local/share/osagent/venv/bin/python"

if [[ -e "$results/status.log" ]]; then
    echo "result directory already contains a status log: $results" >&2
    exit 2
fi
if [[ ! -f "$runner" || ! -d "$dataset" || ! -x "$python_bin" ]]; then
    echo "frozen stage, dataset, or Python environment is unavailable" >&2
    exit 2
fi

mkdir -p "$results/vector-state"
cd "$stage_root"

set -a
# shellcheck disable=SC1090
source "$HOME/.config/osagent/osagent-memory.env"
set +a

export OS_AGENT_EVAL_WORKERS=1
export OS_AGENT_KYLIN_VECTOR_COLLECTION_PREFIX="osagent_eval_native_$(date +%Y%m%d%H%M%S)"
export OS_AGENT_KYLIN_VECTOR_STATE_DIR="$results/vector-state"

started=$(date --iso-8601=seconds)
printf 'state=running started=%s collection=%s\n' \
    "$started" "$OS_AGENT_KYLIN_VECTOR_COLLECTION_PREFIX" > "$results/status.log"

set +e
/usr/bin/time -v -o "$results/round-1.time" \
    "$python_bin" "$runner" \
    --dataset-root "$dataset" \
    --output "$results/round-1.json" \
    --require-kylin-vector \
    > "$results/round-1.stdout" 2> "$results/round-1.stderr"
code=$?
set -e

finished=$(date --iso-8601=seconds)
printf 'state=finished exit=%s finished=%s\n' "$code" "$finished" >> "$results/status.log"
exit "$code"
