# OSagent on Galaxy Kylin Desktop V11

This directory is the deployment boundary for **Galaxy Kylin Desktop V11**.
The installer rejects other distributions so a deployment cannot be mistaken
for a V11 installation.

This deployment keeps OSagent on the Kylin host and runs only Open WebUI in
Docker. The split is intentional: the Kylin AI SDK connects to per-user runtime
sockets under `/tmp`, which are directly available to the native user service.

```text
Browser -> Open WebUI container (host network)
                    -> 127.0.0.1:8765
                    -> native OSagent user service
                    -> Kylin embedding/runtime sockets
```

The installer does not import benchmark data and does not tune retrieval rules.

## Supported target

- Galaxy Kylin Desktop V11 x86_64 (the current VirtualBox guest), with its
  preinstalled Kylin AI SDK/runtime and Docker packages.
- Python 3.11 or newer is required by the current OSagent package.
- Docker Engine and the Compose plugin are required only for Open WebUI.
- The Kylin embedding SDK supports x86 and ARM, but this deployment must still
  be verified on the exact V11 image and architecture used for delivery.

## Install

Run these commands as the desktop user, not as root:

```bash
cd ~/OSagent/deploy/galaxy-kylin-v11
chmod +x install.sh start-webui.sh preflight.sh import-openai-plugins.sh
./install.sh --install-system-deps
```

The dependency flag also installs or upgrades
`libkysdk-vector-engine-client` and
`libkysdk-vector-engine-client-dev` together. The installer requires their
exact package versions to match, builds the native bridge with unresolved-symbol
checks enabled, and then records its path in the service environment. The bridge
build probes both libstdc++ ABI variants against the installed runtime and only
installs a candidate that passes `ldd -r`; set `OS_AGENT_KYLIN_CXX_ABI=0` or `1`
only when the target image has a documented SDK requirement.

If Python, the compiler, JSON headers, and matching Kylin vector SDK packages
are already installed, omit the dependency flag:

```bash
./install.sh
```

On Galaxy Kylin V11, the installer selects guest port `10200` because the
system `KSC_PUBLIC_INPUT` policy rejects new connections to port `3000`. For a
VirtualBox NAT VM, allow the host-only port forward to reach the guest address:

```bash
./install.sh --webui-port 10200 --webui-bind-address 0.0.0.0
```

Then add a VirtualBox NAT rule bound on Windows only:

```text
127.0.0.1:33001 -> guest TCP 10200
```

Keep the VM network in NAT mode. Do not use `0.0.0.0` with bridged networking
while `WEBUI_AUTH=False`; that would expose an unauthenticated UI to the LAN.

The installer creates:

| Path | Purpose |
|---|---|
| `~/.local/share/osagent/venv` | Installed OSagent Python environment |
| `~/.local/share/osagent/memory` | Authoritative memory data |
| `~/OSagent-workspace` | Default writable tool workspace |
| `~/.config/osagent/embedding.json` | Kylin system embedding configuration |
| `~/.config/osagent/osagent-memory.env` | Native service settings and secret |
| `~/.config/systemd/user/osagent-memory.service` | User service |
| `deploy/galaxy-kylin-v11/.env` | Matching Open WebUI connection settings |

## Build the native Open WebUI page

The default Compose file uses the upstream slim image and the gateway bridge,
so an existing deployment keeps working. To use the native Svelte Plugins page
from the pinned source checkout, build the offline overlay image on the V11
host:

```bash
./build-openwebui.sh
printf '\nOPEN_WEBUI_IMAGE=osagent-open-webui:v0.11.0-osagent\n' >> .env
./start-webui.sh
```

The default build reuses the cached `ghcr.io/open-webui/open-webui:v0.11.0-slim`
image and copies the already verified `vendor/open-webui/build` output plus the
pinned `backend/open_webui/main.py` integration, so the V11 guest does not need
Node or registry access. The backend integration restores the managed OSagent
MCP connection from deployment environment variables whenever Open WebUI starts.
Overlay mode stages only those two inputs in a temporary build context instead
of sending the full Open WebUI checkout and `node_modules` to the legacy Docker
builder.
For a full source build on a
host with a working registry and Node toolchain, set
`OSAGENT_OPENWEBUI_BUILD_MODE=full`. The OSagent backend remains the native user
service; only the Open WebUI image is rebuilt.

The V11 Compose file sets `OFFLINE_MODE=True`. This prevents Open WebUI's
optional local RAG model loader from retrying Hugging Face during cold start
when its own model cache is empty. It does not disable or change OSagent's
native Kylin embedding provider. Open WebUI Knowledge requires a separately
provisioned RAG model or endpoint before it should be enabled in an offline VM.

Use a different workspace when the model must not access the repository:

```bash
./install.sh --workspace "$HOME/Documents/AgentWorkspace"
```

The installer defaults the service to `127.0.0.1`. The Open WebUI container uses
host networking so it can reach that loopback endpoint without exposing the
memory API to the LAN. Open WebUI also binds to `127.0.0.1` by default because the
default single-user configuration has authentication disabled. Every non-health
OSagent endpoint still requires the generated bearer token.

For LAN access, enable Open WebUI authentication first and then set
`WEBUI_BIND_ADDRESS="0.0.0.0"` in `deploy/galaxy-kylin-v11/.env`. Do not expose the
unauthenticated default UI on a shared network.

## Mobile commands

The native service exposes a bounded asynchronous command queue for a phone or
other remote client:

```text
POST   /v1/mobile/commands
GET    /v1/mobile/commands/{command_id}
DELETE /v1/mobile/commands/{command_id}
```

Requests use the service bearer key when sent directly to port `8765`. When
sent through the gateway (`/osagent/...`), use the separate
`OS_AGENT_MOBILE_API_KEY` generated by `install.sh`; the gateway terminates that
token and injects the private service key. A queued command can be cancelled;
running commands are never falsely reported as cancelled. The queue keeps at
most 200 records and prompt text is limited to 16,000 characters.

Example from a LAN client after the gateway is deliberately bound to a
firewalled LAN address:

```bash
curl -X POST http://<kylin-host>:10200/osagent/v1/mobile/commands \
  -H "Authorization: Bearer <mobile-api-key>" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"列出工作区中最近修改的文件","session_id":"phone-1"}'
curl -H "Authorization: Bearer <mobile-api-key>" \
  http://<kylin-host>:10200/osagent/v1/mobile/commands/<command-id>
```

Do not reuse the service key in a mobile app. Keep the gateway behind a
firewall or VPN, and rotate both keys by editing
`~/.config/osagent/osagent-memory.env` and `deploy/galaxy-kylin-v11/.env`, then
restarting the user service and Compose stack.

## Configure models

The installer configures OSagent embedding through the system SDK:

```json
{
  "embedding_backend": "kylin_system",
  "embedding_model": "ensemble-embd_gte-base_uint8-text",
  "embedding_library_path": null
}
```

Edit `~/.config/osagent/osagent-memory.env` when `os_agent_run` should use a
specific OpenAI-compatible model:

```dotenv
OS_AGENT_BASE_URL="http://127.0.0.1:11434/v1"
OS_AGENT_API_KEY="local"
OS_AGENT_MODEL="qwen3:8b"
```

Then reload the native service:

```bash
systemctl --user restart osagent-memory.service
journalctl --user -u osagent-memory.service -n 100 --no-pager
```

Configure the chat model separately in `deploy/galaxy-kylin-v11/.env`. Chat requests
go directly from Open WebUI to that endpoint; the `OS_AGENT_*` model is used
only by the explicit `os_agent_run` MCP tool.

To apply one existing OpenAI-compatible JSON configuration to both services:

```bash
python3 deploy/galaxy-kylin-v11/configure_llm.py ~/.config/osagent/llm.local.json
```

## Start Open WebUI

```bash
./start-webui.sh
```

Open the port written to `WEBUI_PORT` in `.env` (normally
`http://127.0.0.1:10200` inside V11). For the VirtualBox NAT rule above, open
`http://127.0.0.1:33001` on Windows and set
`WEBUI_PUBLIC_URL=http://127.0.0.1:33001` in `.env`. This browser-visible origin
is required for third-party OAuth callbacks; it is deliberately separate from
the guest bind address and port. The customized Open WebUI synchronizes this
explicit value into its persistent `webui.url` setting during startup. The customized Open WebUI
source under `vendor/open-webui` adds a native **Plugins** page to the workspace
navigation. It provides search, category filtering, install, enable, disable,
and uninstall. Knowledge upload is intentionally kept out of the plugin
marketplace. The stock image remains compatible through the
gateway's `integrations/open_webui/marketplace-bridge.js` fallback.

### Refresh a customized frontend

`127.0.0.1:33001` serves the static build inside the V11 container; it does not
serve the Windows source tree directly. After changing the customized frontend,
run the overlay checks and production build on the development host, copy the
resulting `vendor/open-webui/build` directory to the same path on V11, and then
rebuild and recreate both containers:

```bash
./build-openwebui.sh
/opt/system/bin/docker compose --env-file .env -f compose.yaml up -d --force-recreate
```

Do not diagnose an old page as browser cache until the root document's
`Last-Modified`/`ETag` and the running image ID have changed. Open WebUI uses
content-hashed JavaScript assets, so a normal reload after a successful container
replacement loads the new bundle; an already-open tab may still need one reload.

For a new Open WebUI volume, register the automatic memory filter with an
administrator session token and the generated memory key:

```bash
export OPEN_WEBUI_TOKEN="<admin-session-token>"
set -a
. ~/.config/osagent/osagent-memory.env
. deploy/galaxy-kylin-v11/.env
set +a
python3 register_open_webui.py
```

The script installs the automatic memory filter and also verifies the MCP
Streamable HTTP endpoint at `http://127.0.0.1:8765/mcp`. The customized Open
WebUI image independently restores the managed MCP connection on every startup
when `OS_AGENT_MCP_AUTO_REGISTER=True`, so a reboot does not require rerunning
the script merely to recover the nine `os_agent_*` tools. Agent MCP is registered
and available by default; plugin installation and Computer Use actions remain
separately permission-gated. An administrator may provide a
default Qianxin executable and browser policy:

```dotenv
OS_AGENT_COMPUTER_USE_ENABLED="true"
OS_AGENT_QIANXIN_BROWSER_EXECUTABLE="/path/to/qianxin-browser"
OS_AGENT_BROWSER_ALLOWED_ORIGINS="https://intranet.example"
OS_AGENT_COMPUTER_USE_MODE="allowlist"
```

No OSagent extension panel is injected into the chat composer. The OSagent MCP
server remains registered and enabled globally, so models can discover the
`os_agent_*` tools according to their normal tool permissions without a second
chat-box switch. Plugin catalog search and lifecycle management live in the
**Workspace -> Plugins** page. Computer Use and file-access policy are backend
capabilities, not chat composer controls; they remain disabled or restricted
by the authenticated service policy unless an administrator explicitly enables
them. The runtime policy files are stored beside OSagent data at
`data/settings/computer-use.json` and `data/settings/file-access.json`;
deleting either file restores the corresponding environment defaults. The
systemd user unit imports the graphical session variables required to launch
the browser from the native service.

Computer Use has three modes. `allowlist` is the default and permits only exact
origins in `OS_AGENT_BROWSER_ALLOWED_ORIGINS`. `public` permits resolved public
HTTP(S) hosts but blocks localhost, loopback, private, link-local, and reserved
addresses; it also requires
`OS_AGENT_COMPUTER_USE_PUBLIC_CONFIRM=I_UNDERSTAND_PUBLIC_WEB`. `unrestricted`
permits any HTTP(S) origin, including private hosts, only after setting
`OS_AGENT_COMPUTER_USE_UNRESTRICTED_CONFIRM=I_UNDERSTAND_UNRESTRICTED_WEB`.
All browser modes continue to reject non-HTTP schemes. Desktop input is exposed
only through the authenticated `target="desktop"` Computer Use path; browser
URL modes do not grant desktop access. There is no chat-dialog switch for this
capability in the current UI.
The installer leaves both elevated modes locked. To make `public` selectable,
the host administrator must add
`OS_AGENT_COMPUTER_USE_PUBLIC_CONFIRM=I_UNDERSTAND_PUBLIC_WEB` to
`~/.config/osagent/osagent-memory.env` and restart the service. `unrestricted`
requires its separate confirmation and remains independently locked.

File mutation policy is also selectable in the customized workspace as
`read_only`, `workspace`, or `full_access`. The first two are available in the
normal V11 deployment. To make `full_access` selectable, the host administrator
must add the following line to `~/.config/osagent/osagent-memory.env` and
restart `osagent-memory.service`:

```dotenv
OS_AGENT_FILE_FULL_ACCESS_CONFIRM="I_UNDERSTAND_FULL_FILESYSTEM_ACCESS"
```

`full_access` accepts absolute paths and `~`, subject to the V11 user account's
normal filesystem permissions. It does not grant root privileges. The selected
mode is persisted in `data/settings/file-access.json`.

Install the optional browser dependency in the OSagent environment with
`pip install 'os-agent-memory[computer-use]'`. The browser target requires the
trusted browser's Chromium DevTools endpoint. Full desktop actions use the V11
session's `xdotool` or `ydotool` input backend and a Wayland/X11 screenshot
provider; on a Wayland V11 desktop install `grim` with `sudo apt install grim`.
They are available only when the backend Computer Use policy and graphical
session prerequisites are explicitly enabled; the current chat UI does not
provide a Computer Use toggle.

The MCP endpoint can also be checked independently with the installed OSagent
environment:

```bash
~/.local/share/osagent/venv/bin/python deploy/galaxy-kylin-v11/verify_mcp.py
```

To verify the real Qianxin browser through the same MCP tool exposed to the
agent, keep the V11 desktop logged in and run:

```bash
~/.local/share/osagent/venv/bin/python \
  deploy/galaxy-kylin-v11/verify_qianxin_browser.py \
  --query '北京航空航天大学' \
  --input-selector 'textarea:visible' \
  --submit-key Enter
```

The verifier leaves the browser open for visual inspection and stores its
screenshot below the configured Workspace. The MCP tool uses `target="browser"`
for page-scoped DOM actions and direct CDP mouse/keyboard events. Browser
`click` accepts screenshot coordinates, while `type`/`keypress` can target the
current focus without a selector. Use `target="desktop"` with `screenshot`, `move`,
`click`, `double_click`, `right_click`, `middle_click`, `drag`, `scroll`,
`keypress`, or `type` for full desktop control. Desktop coordinates are bounded
to the screenshot coordinate space and never become shell commands. See the
[V11 Qianxin acceptance report](../../doc/evaluation/2026-09-03-galaxy-kylin-qianxin-computer-use.md)
for the verified executable, CDP identity, limitations, and captured evidence.

Computer Use screenshots returned through MCP contain both the original JSON
metadata and a standard `ImageContent` block (`image_transport=mcp_image_content`).
A multimodal model can therefore inspect the actual V11
screen and choose coordinates for the next desktop event. Text-only models can
still use the path, width, and height fields, but cannot perform reliable visual
recognition; changing selectors or adding fixed coordinate rules does not solve
that limitation. The image payload is read only from the OSagent workspace and
is capped at 8 MiB by default. Set
`OS_AGENT_COMPUTER_USE_MAX_IMAGE_BYTES` only when the MCP host has a larger
approved response limit.

The V11 service is currently configured with `OS_AGENT_MODEL=deepseek-v4-flash`,
which is a text model and cannot inspect the returned image. For DeepSeek API,
select the separate `deepseek-v4-flash-vision-exp` model for visual Computer Use,
or configure another model that declares image input support. The Open WebUI chat
model and the model used by the explicit `os_agent_run` tool are configured
independently; changing only one does not change the other.

To verify the image block itself on V11 (not just the saved file), run:

```bash
~/.local/share/osagent/venv/bin/python \
  deploy/galaxy-kylin-v11/verify_mcp_image.py
```

The command reports `mcp_ok`, image block count, MIME type, and encoded payload
length. A `mcp_ok=false` result means the MCP host received metadata only and the
connected model cannot perform visual coordinate selection.

The Compose configuration disables Open WebUI's independent Memories and Notes
features. OSagent remains the only memory source.

## Plugin marketplace

Open the Open WebUI URL (for the VirtualBox setup,
`http://127.0.0.1:33001`) and select **Workspace -> Plugins**. The customized
source renders the marketplace directly in the workspace instead of opening a
separate site. The gateway forwards `/osagent/*` to the native OSagent service
and injects the service bearer key for this local UI only.

The catalog starts empty because OSagent does not silently download or trust a
third-party registry. To import an existing checkout of the official
[`openai/plugins`](https://github.com/openai/plugins) marketplace, pin the
checkout to a reviewed commit and run:

```bash
./import-openai-plugins.sh /path/to/openai-plugins <full-40-character-commit>
systemctl --user restart osagent-memory.service
```

The importer reads `.agents/plugins/marketplace.json`, copies local bundles to
the OSagent data directory, preserves custom catalog entries, and records the
repository and exact revision. URL-based entries are reported and skipped; the
importer never follows them automatically.

The native OSagent service also keeps a standalone administrator page at
`http://127.0.0.1:8765/plugins`. For direct troubleshooting, create a local SSH
tunnel from Windows and open `http://127.0.0.1:33002/plugins`:

```powershell
ssh -i $env:USERPROFILE\.ssh\galaxy_kylin_osagent -N `
  -L 127.0.0.1:33002:127.0.0.1:8765 -p 2223 veritas@127.0.0.1
```

Keep that terminal open. Enter the value of `OS_AGENT_MEMORY_API_KEY`, then use
the one-click **Install**, **Enable**, **Disable**, and **Uninstall** actions.

This VM image may set `PermitOpen none` in `/etc/ssh/sshd_config`. If the
tunnel is reset, an administrator can allow only this local destination after
checking the configuration:

```bash
sudo sed -i 's/^PermitOpen none$/PermitOpen 127.0.0.1:8765/' /etc/ssh/sshd_config
sudo sshd -t && sudo systemctl reload ssh
```

Do not remove `PermitOpen` or enable unrestricted forwarding.

### Use an installed plugin

Installation only verifies and copies a bundle. The marketplace then reports
one of the following runtime states:

| State | Meaning | Next action |
|---|---|---|
| `可使用` | Declarative instructions exist, or the remote MCP is connected and authorized | Click **在对话中使用** when available |
| `需要连接` | A safe remote HTTPS MCP declaration was found but is not registered | Click **连接** |
| `需要授权` | The MCP connection exists but the current user has not completed OAuth | Click **继续授权** |
| `暂不兼容` | No callable declaration exists, or the declaration requires local commands/static credentials | Add a reviewed adapter; do not run the bundle directly |

For Linear, open **Workspace -> Plugins**, install Linear, click **连接**, and
complete the Linear account consent page. Open WebUI stores that OAuth grant for
the signed-in user and returns to a chat with `server:mcp:osagent-plugin-linear-linear`
selected. OSagent never receives the Linear token. A model without tool calling
support can still chat but cannot invoke the selected plugin.

If the provider shows a callback containing the guest-only port (for example
`127.0.0.1:10201`), correct `WEBUI_PUBLIC_URL`, recreate Open WebUI, uninstall
and reinstall the unapproved plugin, then connect again so dynamic client
registration uses the new callback. Do not approve a client with an unreachable
or unexpected callback URI.

The catalog is `<OS_AGENT_MCP_MEMORY_ROOT>/../plugins/catalog.json`. Install is
disabled by default; an administrator must set
`OS_AGENT_PLUGIN_INSTALL_ENABLED="true"` and restart
`osagent-memory.service`. Remote archives must use HTTPS and provide a matching
SHA-256 digest. Imported local bundles must stay below the fixed source root,
contain `.codex-plugin/plugin.json`, contain no symbolic links, and remain below
100 MiB. Installed state is persisted in
`plugins/marketplace-state.json`, so enablement survives service restarts.
Disabling only removes the plugin from the enabled inventory; OSagent never
imports or executes downloaded plugin code automatically. Installed plugins are
managed from **Workspace -> Plugins**. The chat composer has no OSagent plugin
call panel; an agent may still call the MCP marketplace tool through its normal
globally registered tool set, subject to the same installation and permission
checks. The runtime returns bounded `SKILL.md` instructions as declarative
context and does not run bundle code.

## Verify

```bash
./preflight.sh
```

Required failures produce a nonzero exit code. On V11 the script loads the
native service environment, runs the strict `competition preflight`, and then
runs `competition vector-smoke`. Missing Kylin embedding/vector libraries,
runtime sockets, the bridge, or a successful write/filter/update/search/delete
round trip is therefore a deployment failure. The later library and socket
rows are retained as diagnostics; they do not weaken the strict competition
result.

Useful diagnostics:

```bash
systemctl --user status osagent-memory.service
curl http://127.0.0.1:8765/health
docker compose --env-file .env -f compose.yaml ps
```

The Python runtime uses the Kylin embedding API and the C++
`libkysdk-vector-engine-client` bridge for the semantic fact and session
indexes. Core memory records remain authoritative in OSagent JSON/SQLite; the
vector payloads and nearest-neighbor search are stored and executed by the
Kylin vector engine. The Python wrapper loads the installed bridge directly.
On V11, `install.sh` registers that single file with `kyexectl`; copying it to
`/tmp` would lose the path-specific trust state and can make `dlopen` fail with
`failed to map segment`. `OS_AGENT_KYLIN_VECTOR_STAGE_DIR` remains an explicit
compatibility override, but the selected destination must be permitted by the
target OS execution policy.

The deployment preflight is the repeatable native-path evidence: it embeds two
smoke texts with the Kylin system provider, writes them to a dimension-scoped
system collection, verifies filtered Top-K retrieval, swaps their vectors to
verify update semantics, removes both rows, and verifies deletion. A missing
bridge or failed round trip must not be described as native acceleration.

If preflight reports an undefined `VectorDB` symbol, both C++ ABI candidates
were rejected. Upgrade the runtime and development packages together through
the V11 package manager and rebuild; extracting the runtime into a user-writable
directory is not a valid workaround because V11 can reject executable mappings
from those locations:

```bash
sudo apt-get update
sudo apt-get install -y \
  libkysdk-vector-engine-client \
  libkysdk-vector-engine-client-dev \
  nlohmann-json3-dev g++
./install.sh
./preflight.sh
```
