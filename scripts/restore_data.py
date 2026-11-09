"""Restore selected release ZIPs after checking the committed checksums."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ccspeech.data.bundle import restore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "text", "taukadial"], default="all")
    parser.add_argument("--bundle-dir", type=Path, default=REPO / "artifacts/data_v1")
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    args = parser.parse_args()
    release = json.loads((REPO / "configs/data/release_v1.json").read_text())
    for bundle in release["bundles"]:
        if args.dataset not in ("all", bundle["dataset"]):
            continue
        if Path(bundle["file"]).name != bundle["file"]:
            raise ValueError("Release bundle names must be basenames")
        restore(args.bundle_dir / bundle["file"], args.data_root, bundle["sha256"])


if __name__ == "__main__":
    main()
