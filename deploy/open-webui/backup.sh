#!/usr/bin/env sh
set -eu

deploy_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
backup_dir=${1:-"$deploy_dir/backup-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$backup_dir"
backup_dir=$(CDPATH= cd -- "$backup_dir" && pwd)

env_value() {
    name=$1
    fallback=$2
    value=$(sed -n "s/^$name=//p" "$deploy_dir/.env" 2>/dev/null | tail -n 1)
    printf '%s\n' "${value:-$fallback}"
}

export_volume() {
    volume=$1
    archive=$2
    docker volume inspect "$volume" >/dev/null
    docker run --rm \
        --mount "type=volume,src=$volume,dst=/source,readonly" \
        --mount "type=bind,src=$backup_dir,dst=/backup" \
        --entrypoint sh \
        osagent-memory-web:0.1.0 \
        -c "tar -C /source -czf /backup/$archive ."
}

webui_volume=$(env_value OPEN_WEBUI_DATA_VOLUME osagent-open-webui-data)
memory_volume=$(env_value OS_AGENT_MEMORY_DATA_VOLUME osagent-memory-data)

cd "$deploy_dir"
docker compose stop
trap 'docker compose start' EXIT
export_volume "$webui_volume" open-webui-data.tar.gz
export_volume "$memory_volume" osagent-memory-data.tar.gz
docker compose start
trap - EXIT
echo "Backup written to $backup_dir"
