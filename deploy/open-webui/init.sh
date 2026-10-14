#!/usr/bin/env sh
set -eu

deploy_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
env_file="$deploy_dir/.env"

if [ -f "$env_file" ]; then
    echo ".env already exists; leaving it unchanged."
    exit 0
fi

webui_secret=$(openssl rand -hex 32)
memory_secret=$(openssl rand -hex 32)
sed \
    -e "s/WEBUI_SECRET_KEY=replace-with-a-random-secret/WEBUI_SECRET_KEY=$webui_secret/" \
    -e "s/OS_AGENT_MEMORY_API_KEY=replace-with-a-different-random-secret/OS_AGENT_MEMORY_API_KEY=$memory_secret/" \
    "$deploy_dir/.env.example" > "$env_file"
chmod 600 "$env_file"
echo "Created deploy/open-webui/.env with random local secrets."
