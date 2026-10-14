#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
env_file=$script_dir/.env

if [[ ! -r $env_file ]]; then
    echo "Missing $env_file. Run ./install.sh first." >&2
    exit 1
fi
docker_bin=${DOCKER_BIN:-$(command -v docker || true)}
if [[ -z $docker_bin && -x /opt/system/bin/docker ]]; then
    docker_bin=/opt/system/bin/docker
fi
if [[ -z $docker_bin ]]; then
    echo "Docker is not installed." >&2
    exit 1
fi
if ! "$docker_bin" compose version >/dev/null 2>&1; then
    echo "The Docker Compose plugin is not installed." >&2
    exit 1
fi

"$docker_bin" compose --env-file "$env_file" -f "$script_dir/compose.yaml" up -d
"$docker_bin" compose --env-file "$env_file" -f "$script_dir/compose.yaml" ps
webui_port=$(sed -n 's/^WEBUI_PORT=//p' "$env_file" | tail -n 1 | tr -d '"')
echo "Open WebUI: http://127.0.0.1:${webui_port:-3000}"
