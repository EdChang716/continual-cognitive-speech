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
    parser.add_argument("--dataset", choices=["all", "training", "text", "taukadial", "speech_v2"], default="training")
    parser.add_argument("--bundle-dir", type=Path, default=REPO / "artifacts/data_v2")
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--release", choices=[1, 2], type=int, default=2)
    args = parser.parse_args()
    release = json.loads((REPO / f"configs/data/release_v{args.release}.json").read_text())
    for bundle in release["bundles"]:
        selected = args.dataset == "all" or args.dataset == bundle["dataset"] or (
            args.dataset == "training" and bundle["dataset"] in {"text", "speech_v2"})
        if not selected:
            continue
        if Path(bundle["file"]).name != bundle["file"]:
            raise ValueError("Release bundle names must be basenames")
        restore(args.bundle_dir / bundle["file"], args.data_root, bundle["sha256"])


if __name__ == "__main__":
    main()
