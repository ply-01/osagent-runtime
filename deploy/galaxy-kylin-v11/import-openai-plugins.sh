#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 /path/to/openai-plugins-checkout [full-git-revision]" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source_checkout=$(realpath "$1")
plugin_root=${OS_AGENT_PLUGIN_ROOT:-"$HOME/.local/share/osagent/plugins"}
python_bin="$HOME/.local/share/osagent/venv/bin/python"

if [[ ! -x $python_bin ]]; then
  python_bin=python3
fi

args=(
  -m os_agent.plugins.openai_catalog
  --source "$source_checkout"
  --plugin-root "$plugin_root"
)
if [[ $# -eq 2 ]]; then
  args+=(--revision "$2")
fi

PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" "${args[@]}"
