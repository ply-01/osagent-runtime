#!/usr/bin/env bash

shopt -s nullglob
PATCHES=("$PATCH_DIR"/000*.patch)
if ((${#PATCHES[@]} == 0)); then
  echo "No Open WebUI overlay patches found under $PATCH_DIR" >&2
  exit 1
fi

for patch in "${PATCHES[@]}"; do
  if git -C "$SOURCE_DIR" apply --check "$patch" >/dev/null 2>&1; then
    git -C "$SOURCE_DIR" apply "$patch"
    echo "Applied $(basename "$patch") to Open WebUI $TAG"
  else
    echo "Skipped $(basename "$patch"): already applied or conflicts with the checkout"
  fi
done

git -C "$SOURCE_DIR" rev-parse HEAD
