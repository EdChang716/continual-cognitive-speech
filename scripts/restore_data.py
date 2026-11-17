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
    parser.add_argument("--dataset", choices=["all", "training", "text", "taukadial", "speech_v2", "speech_v3"], default="training")
    parser.add_argument("--bundle-dir", type=Path, help="Downloaded ZIP directory (default: artifacts/data_v<release>)")
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--release", choices=[1, 2, 3], type=int, default=3)
    args = parser.parse_args()
    release = json.loads((REPO / f"configs/data/release_v{args.release}.json").read_text())
    bundle_dir = args.bundle_dir or REPO / f"artifacts/data_v{args.release}"
    speech = {bundle["dataset"] for bundle in release["bundles"]} & {"speech_v2", "speech_v3"}
    training = {"text"} | ({max(speech)} if speech else set())
    selected_bundles = []
    for bundle in release["bundles"]:
        selected = args.dataset == "all" or args.dataset == bundle["dataset"] or (
            args.dataset == "training" and bundle["dataset"] in training)
        if not selected:
            continue
        selected_bundles.append(bundle)
        if Path(bundle["file"]).name != bundle["file"]:
            raise ValueError("Release bundle names must be basenames")
    if not selected_bundles:
        raise ValueError(f"Dataset {args.dataset} is unavailable in release v{args.release}")
    for bundle in selected_bundles:
        restore(bundle_dir / bundle["file"], args.data_root, bundle["sha256"])


if __name__ == "__main__":
    main()
