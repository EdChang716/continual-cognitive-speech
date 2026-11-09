"""Download pinned public source files, resume partial transfers, and verify SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes data root: {relative}")
    return path


def verify(path: Path, item: dict) -> None:
    if path.stat().st_size != item["size_bytes"]:
        raise ValueError(f"Unexpected size: {path}")
    if sha256_file(path) != item["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def download_verified(target: Path, item: dict, url: str) -> None:
    """Resume once, then restart once if a partial transfer is unusable.

    Failed partial files are quarantined instead of deleting user data. A final
    file is published only after both its size and checksum pass verification.
    """
    temporary = target.with_suffix(target.suffix + ".part")
    if temporary.is_symlink():
        raise ValueError(f"Refusing a symlink download destination: {temporary}")
    for attempt in range(2):
        try:
            # An interrupted run may have finished downloading just before the
            # rename. Verify that file directly instead of requesting EOF.
            if not temporary.exists() or temporary.stat().st_size < item["size_bytes"]:
                subprocess.run([
                    "curl", "--fail", "--location", "--retry", "5", "--retry-delay", "3",
                    "--connect-timeout", "30", "--continue-at", "-", "--output", str(temporary), url,
                ], check=True)
            verify(temporary, item)
            temporary.replace(target)
            return
        except (ValueError, subprocess.CalledProcessError):
            if attempt == 1:
                raise
            if temporary.exists():
                quarantine = temporary.with_name(f"{temporary.name}.invalid-{time.time_ns()}")
                temporary.replace(quarantine)
                print(f"Retained failed partial as {quarantine.name}; retrying from zero", flush=True)


def acquire(data_root: Path, sources: Path) -> None:
    manifest = json.loads(sources.read_text())
    for name, dataset in manifest["datasets"].items():
        for item in dataset["files"]:
            target = safe_path(data_root, item["local_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                verify(target, item)
                print(f"Verified existing {name}/{target.name}", flush=True)
                continue
            url = (f"https://huggingface.co/datasets/{dataset['repository']}/resolve/"
                   f"{dataset['revision']}/{item['remote_path']}?download=true")
            print(f"Downloading {name}/{target.name} ({item['size_bytes']:,} bytes)", flush=True)
            download_verified(target, item, url)
            print(f"Verified SHA-256: {name}/{target.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--sources", type=Path, default=Path("configs/data/sources.json"))
    args = parser.parse_args()
    acquire(args.data_root, args.sources)


if __name__ == "__main__":
    main()
