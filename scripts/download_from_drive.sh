#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
drive_remote="${1:-gdrive}"
case "$drive_remote" in
  *[!a-zA-Z0-9_-]*|'') echo 'Use a configured rclone remote name without a colon.' >&2; exit 2 ;;
esac
if ! command -v rclone >/dev/null 2>&1; then
  echo 'Install rclone and configure an authorized remote; see docs/data_access.md.' >&2
  exit 1
fi
mkdir -p "$repo_root/artifacts/data_v1"
rclone copy "$drive_remote:" "$repo_root/artifacts/data_v1" \
  --drive-root-folder-id 1F8PnLDto77jWNNTvXChJH-DefKO3Z6Gp \
  --include '/text_data_v1.zip' \
  --include '/taukadial_data_v1.zip' \
  --include '/SHA256SUMS' \
  --include '/README_DATA.md' \
  --progress
echo 'Downloaded. Run: python scripts/restore_data.py'
