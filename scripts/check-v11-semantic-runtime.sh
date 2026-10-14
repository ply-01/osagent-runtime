#!/usr/bin/env bash
set -euo pipefail
set -a
source "$HOME/.config/osagent/osagent-memory.env"
set +a
export OS_AGENT_MEMORY_EXTRACTOR=semantic_state
export OS_AGENT_TIMEOUT_SECONDS=20
export OS_AGENT_LLM_MAX_RETRIES=0
export PYTHONPATH="$HOME/.cache/osagent-v11-eval-20260909/src"
"$HOME/.local/share/osagent/venv/bin/python" "$HOME/.cache/check-semantic-memory-runtime.py"
