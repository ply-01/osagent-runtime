#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
config_dir=${XDG_CONFIG_HOME:-"$HOME/.config"}/osagent
data_dir=${XDG_DATA_HOME:-"$HOME/.local/share"}/osagent
service_dir=${XDG_CONFIG_HOME:-"$HOME/.config"}/systemd/user
venv_dir=$data_dir/venv
memory_dir=$data_dir/memory
workspace=$HOME/OSagent-workspace
port=8765
webui_port=
webui_bind_address=127.0.0.1
webui_public_url=
install_system_deps=false
skip_os_check=false

usage() {
    cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --workspace PATH       Workspace exposed to OSagent tools.
  --port PORT            Loopback service port (default: 8765).
  --webui-port PORT      Open WebUI host-network port (default: 10200).
  --webui-bind-address ADDRESS
                         Open WebUI bind address (default: 127.0.0.1).
  --webui-public-url URL Browser-visible origin used for OAuth callbacks.
  --install-system-deps  Install host tools and matching Kylin vector SDK packages.
  --skip-os-check        Permit installation on a non-V11 test host.
  -h, --help             Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --workspace)
            workspace=${2:?--workspace requires a path}
            shift 2
            ;;
        --port)
            port=${2:?--port requires a value}
            shift 2
            ;;
        --webui-port)
            webui_port=${2:?--webui-port requires a value}
            shift 2
            ;;
        --webui-bind-address)
            webui_bind_address=${2:?--webui-bind-address requires a value}
            shift 2
            ;;
        --webui-public-url)
            webui_public_url=${2:?--webui-public-url requires a value}
            shift 2
            ;;
        --install-system-deps)
            install_system_deps=true
            shift
            ;;
        --skip-os-check)
            skip_os_check=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ $EUID -eq 0 ]]; then
    echo "Run this installer as the desktop user, not root." >&2
    exit 1
fi
if [[ ! $port =~ ^[0-9]+$ ]] || ((port < 1024 || port > 65535)); then
    echo "Port must be an integer between 1024 and 65535." >&2
    exit 1
fi

if [[ $skip_os_check != true ]]; then
    if [[ ! -r /etc/os-release ]]; then
        echo "Cannot identify the operating system: /etc/os-release is missing." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source /etc/os-release
    os_identity="${ID:-} ${ID_LIKE:-} ${NAME:-} ${VERSION_ID:-}"
    os_id=${ID:-}
    version_id=${VERSION_ID:-}
    if [[ ${os_id,,} != kylin || ${version_id,,} != v11 ]]; then
        echo "This installer targets Galaxy Kylin Desktop V11; detected: $os_identity" >&2
        exit 1
    fi
fi

if [[ -z $webui_port ]]; then
    webui_port=10200
fi
if [[ ! $webui_port =~ ^[0-9]+$ ]] || ((webui_port < 1024 || webui_port > 65535)); then
    echo "WebUI port must be an integer between 1024 and 65535." >&2
    exit 1
fi
if [[ $webui_bind_address != 127.0.0.1 && $webui_bind_address != 0.0.0.0 ]]; then
    echo "WebUI bind address must be 127.0.0.1 or 0.0.0.0." >&2
    exit 1
fi
if [[ -z $webui_public_url ]]; then
    webui_public_url="http://127.0.0.1:$webui_port"
fi
if [[ $webui_public_url != http://* && $webui_public_url != https://* ]]; then
    echo "WebUI public URL must use http:// or https://." >&2
    exit 1
fi
webui_public_url=${webui_public_url%/}

if [[ $install_system_deps == true ]]; then
    sudo apt-get update
    sudo apt-get install -y \
        ca-certificates curl g++ nlohmann-json3-dev openssl \
        python3 python3-pip python3-venv \
        libkysdk-vector-engine-client libkysdk-vector-engine-client-dev
fi

for command_name in python3 curl dpkg-query g++ systemctl; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required command is missing: $command_name" >&2
        exit 1
    fi
done
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || {
    echo "OSagent requires Python 3.11 or newer." >&2
    exit 1
}

vector_runtime_version=$(dpkg-query -W -f='${Version}' libkysdk-vector-engine-client 2>/dev/null || true)
vector_development_version=$(dpkg-query -W -f='${Version}' libkysdk-vector-engine-client-dev 2>/dev/null || true)
if [[ -z $vector_runtime_version || -z $vector_development_version ]]; then
    echo "Kylin vector runtime and development packages are both required." >&2
    echo "Re-run with --install-system-deps." >&2
    exit 1
fi
if [[ $vector_runtime_version != "$vector_development_version" ]]; then
    echo "Kylin vector SDK package versions do not match:" >&2
    echo "  runtime=$vector_runtime_version" >&2
    echo "  development=$vector_development_version" >&2
    echo "Upgrade both packages together, then retry." >&2
    exit 1
fi

bridge_path=$HOME/.local/lib/libosagent_kylin_vector_bridge.so
mkdir -p "$(dirname "$bridge_path")"
bash "$repo_root/native/kylin_vector_bridge/build.sh" "$bridge_path"
if command -v kyexectl >/dev/null 2>&1; then
    # Kylin execution control is path-aware. Trust the one installed bridge;
    # copying it to /tmp would create an unverified shared object.
    sudo kyexectl -s "$bridge_path"
fi

mkdir -p "$config_dir" "$data_dir" "$service_dir" "$memory_dir" "$workspace"
python3 -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install "$repo_root"

embedding_config=$config_dir/embedding.json
if [[ ! -e $embedding_config ]]; then
    cp "$repo_root/config/embedding.kylin.example.json" "$embedding_config"
fi

host_env=$config_dir/osagent-memory.env
webui_env=$script_dir/.env
touch "$host_env" "$webui_env"
chmod 600 "$host_env" "$webui_env"

read_env() {
    local file=$1
    local name=$2
    sed -n "s/^${name}=//p" "$file" | tail -n 1 | tr -d '"'
}

set_env() {
    local file=$1
    local name=$2
    local value=$3
    python3 - "$file" "$name" "$value" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
name = sys.argv[2]
value = sys.argv[3]
if "\n" in value or "\r" in value:
    raise SystemExit(f"{name} must be a single-line value")
escaped = value.replace("\\", "\\\\").replace('"', '\\"')
line = f'{name}="{escaped}"'
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
for index, current in enumerate(lines):
    if current.startswith(f"{name}="):
        lines[index] = line
        break
else:
    lines.append(line)
path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
PY
}

memory_key=$(read_env "$host_env" OS_AGENT_MEMORY_API_KEY)
if [[ -z $memory_key ]]; then
    memory_key=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
fi
mobile_key=$(read_env "$host_env" OS_AGENT_MOBILE_API_KEY)
if [[ -z $mobile_key ]]; then
    mobile_key=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
fi
webui_secret=$(read_env "$webui_env" WEBUI_SECRET_KEY)
if [[ -z $webui_secret ]]; then
    webui_secret=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
fi

set_env "$host_env" OS_AGENT_MEMORY_API_KEY "$memory_key"
set_env "$host_env" OS_AGENT_MOBILE_API_KEY "$mobile_key"
set_env "$host_env" OS_AGENT_MEMORY_HOST "127.0.0.1"
set_env "$host_env" OS_AGENT_MEMORY_PORT "$port"
set_env "$host_env" OS_AGENT_MCP_WORKSPACE "$workspace"
set_env "$host_env" OS_AGENT_MCP_MEMORY_ROOT "$memory_dir"
set_env "$host_env" OS_AGENT_MCP_ALLOW_MEMORY_MUTATIONS "true"
set_env "$host_env" OS_AGENT_MCP_ALLOW_WORKSPACE_MUTATIONS "true"
set_env "$host_env" OS_AGENT_MCP_NO_LLM "false"
set_env "$host_env" OS_AGENT_EMBEDDING_CONFIG "$embedding_config"
set_env "$host_env" OS_AGENT_PLUGIN_INSTALL_ENABLED "false"
set_env "$host_env" OS_AGENT_COMPUTER_USE_ENABLED "false"
set_env "$host_env" OS_AGENT_COMPUTER_USE_MODE "allowlist"
set_env "$host_env" OS_AGENT_KYLIN_VECTOR_BRIDGE_LIBRARY "$bridge_path"
set_env "$host_env" OS_AGENT_KYLIN_VECTOR_STAGE_DIR ""
set_env "$host_env" OS_AGENT_BASE_URL "http://127.0.0.1:11434/v1"
set_env "$host_env" OS_AGENT_API_KEY "local"
set_env "$host_env" OS_AGENT_MODEL "qwen3:8b"

set_env "$webui_env" WEBUI_SECRET_KEY "$webui_secret"
set_env "$webui_env" OS_AGENT_MEMORY_API_KEY "$memory_key"
set_env "$webui_env" OS_AGENT_MOBILE_API_KEY "$mobile_key"
set_env "$webui_env" OS_AGENT_MEMORY_PORT "$port"
set_env "$webui_env" WEBUI_PORT "$webui_port"
set_env "$webui_env" WEBUI_BIND_ADDRESS "$webui_bind_address"
set_env "$webui_env" WEBUI_PUBLIC_URL "$webui_public_url"
set_env "$webui_env" OPEN_WEBUI_IMAGE "ghcr.io/open-webui/open-webui:v0.11.0-slim"
set_env "$webui_env" OPEN_WEBUI_DATA_VOLUME "osagent-galaxy-kylin-v11-webui-data"
set_env "$webui_env" OPENAI_API_BASE_URL "http://127.0.0.1:11434/v1"
set_env "$webui_env" OPENAI_API_KEY "local"

install -m 644 "$script_dir/osagent-memory.service" "$service_dir/osagent-memory.service"
systemctl --user import-environment DISPLAY WAYLAND_DISPLAY XAUTHORITY DBUS_SESSION_BUS_ADDRESS XDG_RUNTIME_DIR 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable --now osagent-memory.service

echo "OSagent native service installed."
echo "Host environment: $host_env"
echo "Memory directory: $memory_dir"
echo "Workspace: $workspace"
echo "Next: edit model settings if needed, then run ./start-webui.sh"
