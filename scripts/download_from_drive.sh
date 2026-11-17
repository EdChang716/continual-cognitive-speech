#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
drive_remote="${1:-gdrive}"
release_number="${2:-3}"
selection="${3:-training}"
case "$drive_remote" in
  *[!a-zA-Z0-9_-]*|'') echo 'Use a configured rclone remote name without a colon.' >&2; exit 2 ;;
esac
case "$release_number" in 1|2|3) ;; *) echo 'Release must be 1, 2, or 3.' >&2; exit 2 ;; esac
case "$selection" in training|all|text|taukadial|speech_v2|speech_v3) ;; *) echo 'Select training, all, text, taukadial, speech_v2, or speech_v3.' >&2; exit 2 ;; esac
if ! command -v rclone >/dev/null 2>&1; then
  echo 'Install rclone and configure an authorized remote; see docs/data_access.md.' >&2
  exit 1
fi
release_manifest="$repo_root/configs/data/release_v${release_number}.json"
filter_file="$(mktemp)"
trap 'rm -f "$filter_file"' EXIT
python3 - "$release_manifest" "$selection" "$release_number" > "$filter_file" <<'PY'
import json
from pathlib import Path
import re
import sys

manifest, selection, version = sys.argv[1:]
release = json.loads(Path(manifest).read_text())
bundles = release['bundles']
if selection == 'training':
    speech = [b['dataset'] for b in bundles if b['dataset'].startswith('speech_v')]
    wanted = {'text', max(speech)} if speech else {'text', 'taukadial'}
elif selection == 'all':
    wanted = {b['dataset'] for b in bundles}
else:
    wanted = {selection}
if not wanted <= {b['dataset'] for b in bundles}:
    raise SystemExit('The requested dataset is unavailable in this release.')
for bundle in bundles:
    if bundle['dataset'] in wanted:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', bundle['file']):
            raise SystemExit('Unsafe release filename.')
        print('/' + bundle['file'])
suffix = '' if version == '1' else '_v' + version
print('/SHA256SUMS' + suffix)
print('/README_DATA' + suffix + '.md')
PY
bundle_dir="$repo_root/artifacts/data_v${release_number}"
mkdir -p "$bundle_dir"
rclone copy "$drive_remote:" "$bundle_dir" \
  --drive-root-folder-id 1F8PnLDto77jWNNTvXChJH-DefKO3Z6Gp \
  --include-from "$filter_file" \
  --progress
echo "Downloaded. Run: python scripts/restore_data.py --release $release_number --dataset $selection"
