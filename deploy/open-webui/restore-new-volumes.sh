#!/usr/bin/env sh
set -eu

if [ "$#" -lt 1 ]; then
    echo "usage: $0 BACKUP_DIRECTORY [SUFFIX]" >&2
    exit 2
fi

backup_dir=$(CDPATH= cd -- "$1" && pwd)
suffix=${2:-$(date +%Y%m%d-%H%M%S)}
webui_volume="osagent-open-webui-data-restored-$suffix"
memory_volume="osagent-memory-data-restored-$suffix"

restore_volume() {
    volume=$1
    archive=$2
    test -f "$backup_dir/$archive"
    if docker volume inspect "$volume" >/dev/null 2>&1; then
        echo "refusing to overwrite existing Docker volume: $volume" >&2
        exit 1
    fi
    docker volume create "$volume" >/dev/null
    docker run --rm \
        --mount "type=volume,src=$volume,dst=/target" \
        --mount "type=bind,src=$backup_dir,dst=/backup,readonly" \
        --entrypoint sh \
        osagent-memory-web:0.1.0 \
        -c "tar -C /target -xzf /backup/$archive"
}

restore_volume "$webui_volume" open-webui-data.tar.gz
restore_volume "$memory_volume" osagent-memory-data.tar.gz

echo "Restore completed into new volumes. Set these values in .env:"
echo "OPEN_WEBUI_DATA_VOLUME=$webui_volume"
echo "OS_AGENT_MEMORY_DATA_VOLUME=$memory_volume"
