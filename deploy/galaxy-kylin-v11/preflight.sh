#!/usr/bin/env bash
set -uo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
config_dir=${XDG_CONFIG_HOME:-"$HOME/.config"}/osagent
host_env=$config_dir/osagent-memory.env
webui_env=$script_dir/.env
failures=0
warnings=0

pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; failures=$((failures + 1)); }
warn() { printf 'WARN  %s\n' "$1"; warnings=$((warnings + 1)); }

os_id=
os_version=
if [[ -r /etc/os-release ]]; then
    # Keep unset release fields from tripping set -u on non-Kylin hosts.
    . /etc/os-release
    os_id=${ID:-}
    os_version=${VERSION_ID:-}
fi
if [[ ${os_id,,} == kylin && ${os_version,,} == v11 ]]; then
    pass "Galaxy Kylin Desktop V11 detected"
else
    fail "Galaxy Kylin Desktop V11 not detected"
fi

if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    pass "Python 3.11 or newer"
else
    fail "Python 3.11 or newer is required"
fi

if systemctl --user is-active --quiet osagent-memory.service; then
    pass "osagent-memory.service is active"
else
    fail "osagent-memory.service is not active"
fi

if [[ -r $host_env ]]; then
    port=$(sed -n 's/^OS_AGENT_MEMORY_PORT=//p' "$host_env" | tail -n 1 | tr -d '"')
    port=${port:-8765}
    if curl --fail --silent --show-error "http://127.0.0.1:${port}/health" >/dev/null; then
        pass "OSagent health endpoint"
    else
        fail "OSagent health endpoint is unavailable"
    fi
    set -a
    # shellcheck disable=SC1090
    . "$host_env"
    set +a
    osagent_python=$HOME/.local/share/osagent/venv/bin/python
    if "$osagent_python" -m os_agent competition preflight >/dev/null; then
        pass "Galaxy Kylin competition readiness"
    else
        fail "Galaxy Kylin competition readiness"
    fi
    if "$osagent_python" -m os_agent competition vector-smoke >/dev/null; then
        pass "Kylin embedding and vector SDK round trip"
    else
        fail "Kylin embedding and vector SDK round trip"
    fi
else
    fail "missing $host_env"
fi

if python3 -c "import ctypes.util; raise SystemExit(0 if ctypes.util.find_library('kysdk-coreai-embedding') else 1)"; then
    pass "Kylin embedding library"
else
    warn "Kylin embedding library is not installed"
fi

uid=$(id -u)
if [[ -S /tmp/.kylin-ai-runtime-unix/$uid/core-textembedding.sock ]]; then
    pass "Kylin embedding runtime socket"
else
    warn "embedding runtime socket is unavailable for uid $uid"
fi

if python3 -c "import ctypes.util; raise SystemExit(0 if ctypes.util.find_library('kysdk-vector-engine-client') else 1)"; then
    pass "Kylin vector client library"
else
    warn "Kylin vector client library is not installed"
fi

if [[ -S /tmp/kylin-ai-vector-engine-$uid.sock ]]; then
    pass "Kylin vector engine socket"
else
    warn "vector engine socket is unavailable for uid $uid"
fi

docker_bin=${DOCKER_BIN:-$(command -v docker || true)}
if [[ -z $docker_bin && -x /opt/system/bin/docker ]]; then
    docker_bin=/opt/system/bin/docker
fi
if [[ -n $docker_bin ]] && "$docker_bin" compose version >/dev/null 2>&1; then
    pass "Docker Compose"
    if [[ -r $webui_env ]] && "$docker_bin" compose --env-file "$webui_env" -f "$script_dir/compose.yaml" config --quiet; then
        pass "Open WebUI Compose configuration"
        webui_port=$(sed -n 's/^WEBUI_PORT=//p' "$webui_env" | tail -n 1 | tr -d '"')
        webui_port=${webui_port:-3000}
        if curl --fail --silent --show-error "http://127.0.0.1:${webui_port}" >/dev/null; then
            pass "Open WebUI HTTP endpoint"
        else
            fail "Open WebUI HTTP endpoint is unavailable"
        fi
    else
        fail "Open WebUI environment or Compose configuration is invalid"
    fi
else
    fail "Docker Engine and Compose plugin are required for Open WebUI"
fi

printf '\nRequired failures: %d; optional Kylin AI warnings: %d\n' "$failures" "$warnings"
exit "$failures"
