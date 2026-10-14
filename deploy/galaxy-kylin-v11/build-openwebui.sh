#!/usr/bin/env bash
set -euo pipefail

# Build the pinned Open WebUI checkout with the native OSagent workspace page.
# The default mode is an offline overlay that reuses the cached V11 base image.
# Set OSAGENT_OPENWEBUI_BUILD_MODE=full only when a registry and Node toolchain
# are available on the build host.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${OPEN_WEBUI_CUSTOM_IMAGE:-osagent-open-webui:v0.11.0-osagent}"
DOCKER_BIN="${DOCKER_BIN:-$(command -v docker || true)}"
if [[ -z "$DOCKER_BIN" && -x /opt/system/bin/docker ]]; then
  DOCKER_BIN=/opt/system/bin/docker
fi
if [[ -z "$DOCKER_BIN" ]]; then
  echo "Docker CLI not found; install Docker or set DOCKER_BIN" >&2
  exit 1
fi

if [[ ${OSAGENT_OPENWEBUI_BUILD_MODE:-overlay} == full ]]; then
  "$DOCKER_BIN" build \
    --build-arg BUILDPLATFORM=linux/amd64 \
    --build-arg USE_SLIM=true \
    --build-arg BUILD_HASH="osagent-${IMAGE##*:}" \
    -t "$IMAGE" \
    "$ROOT_DIR/vendor/open-webui"
else
  OVERLAY_CONTEXT="$(mktemp -d "${TMPDIR:-/tmp}/osagent-openwebui-overlay.XXXXXX")"
  cleanup_overlay_context() {
    rm -rf -- "$OVERLAY_CONTEXT"
  }
  trap cleanup_overlay_context EXIT
  mkdir -p "$OVERLAY_CONTEXT/backend/open_webui"
  cp -a "$ROOT_DIR/vendor/open-webui/build" "$OVERLAY_CONTEXT/build"
  cp "$ROOT_DIR/vendor/open-webui/backend/open_webui/main.py" \
    "$OVERLAY_CONTEXT/backend/open_webui/main.py"
  "$DOCKER_BIN" build \
    -f "$ROOT_DIR/deploy/galaxy-kylin-v11/Dockerfile.native" \
    -t "$IMAGE" \
    "$OVERLAY_CONTEXT"
fi

echo "Built $IMAGE"
echo "Set OPEN_WEBUI_IMAGE=$IMAGE in deploy/galaxy-kylin-v11/.env, then run ./start-webui.sh"
