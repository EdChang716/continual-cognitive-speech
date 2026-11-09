"""Build, verify, and restore portable data bundles without publishing datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED, ZIP_STORED

from .download import sha256_file

MANIFEST = "BUNDLE_MANIFEST.json"
ALLOWED_ROOTS = {"raw", "processed", "locked", "reports"}


def safe_destination(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (relative.is_absolute() or ".." in relative.parts or "\\" in name
            or not relative.parts or relative.parts[0] not in ALLOWED_ROOTS):
        raise ValueError(f"Unsafe bundle member: {name}")
    destination = root.joinpath(*relative.parts)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Bundle member escapes data root: {name}")
    current = destination
    while current != root:
        if current.is_symlink():
            raise ValueError(f"Symlink in extraction destination: {name}")
        current = current.parent
    return destination


def source_files(root: Path, dataset: str) -> list[Path]:
    if dataset == "text":
        prefixes = [
            "raw/text/amazon/train.csv", "raw/text/amazon/test.csv",
            "raw/text/yelp/train.parquet", "raw/text/yelp/test.parquet",
            "processed/text/index.csv", "reports/text.json",
        ] + [f"processed/text/{domain}/{split}.parquet"
             for domain in ("amazon", "yelp") for split in ("train", "validation", "test")]
    else:
        prefixes = [
            "raw/taukadial/archives/TAUKADIAL-24-train.tgz",
            "raw/taukadial/archives/TAUKADIAL-24-test.tgz",
            "raw/taukadial/meta_test.csv", "raw/taukadial/language_map.csv",
            "locked/taukadial/testgroundtruth.csv", "locked/taukadial/test_labels.csv",
            "reports/taukadial/audit.json", "reports/taukadial/file_hashes.json",
        ] + [f"processed/taukadial/{split}.csv" for split in ("train", "validation", "test")]
    files = []
    for prefix in prefixes:
        path = root / prefix
        if not path.exists():
            if prefix.startswith("locked/taukadial/"):
                continue
            raise FileNotFoundError(path)
        if not path.is_file():
            raise ValueError(f"Expected a regular dataset file: {path}")
        files.append(path)
    for path in files:
        if path.is_symlink() or path.suffix in {".part", ".sqlite", ".db"}:
            raise ValueError(f"Unexpected intermediate or linked file: {path}")
    return sorted(files)


def build(root: Path, output: Path, dataset: str) -> dict:
    root = root.resolve()
    files = source_files(root, dataset)
    manifest = {"format_version": 1, "dataset": dataset, "files": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with ZipFile(temporary, "w", allowZip64=True) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            digest = sha256_file(path)
            manifest["files"].append({"path": relative, "size_bytes": path.stat().st_size, "sha256": digest})
            # Archives and Parquet are already compressed. Fixed timestamps and
            # permission metadata make repeated builds byte-reproducible.
            info = ZipInfo(relative, date_time=(2026, 11, 9, 0, 0, 0))
            info.compress_type = ZIP_STORED if path.suffix in {".tgz", ".parquet"} else ZIP_DEFLATED
            info.external_attr = 0o100600 << 16
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as dest:
                shutil.copyfileobj(source, dest, length=8 * 1024 * 1024)
        info = ZipInfo(MANIFEST, date_time=(2026, 11, 9, 0, 0, 0))
        info.external_attr = 0o100600 << 16
        archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    result = {"file": output.name, "size_bytes": output.stat().st_size, "sha256": sha256_file(output),
              "dataset": dataset, "members": len(files)}
    print(json.dumps(result, indent=2), flush=True)
    return result


def verify_members(archive: ZipFile) -> dict:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate archive entries")
    manifest = json.loads(archive.read(MANIFEST))
    if manifest.get("format_version") != 1 or manifest.get("dataset") not in {"text", "taukadial"}:
        raise ValueError("Unsupported data bundle format")
    records = manifest["files"]
    expected = {record["path"] for record in records}
    if len(expected) != len(records) or set(names) != expected | {MANIFEST}:
        raise ValueError("Bundle manifest/member mismatch")
    for record in records:
        safe_destination(Path("/bundle-check"), record["path"])
        info = archive.getinfo(record["path"])
        if stat.S_ISLNK(info.external_attr >> 16) or info.is_dir():
            raise ValueError("Bundle entries must be regular files")
        if info.file_size != record["size_bytes"]:
            raise ValueError(f"Bundle size mismatch: {record['path']}")
        digest = hashlib.sha256()
        with archive.open(info) as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {record['path']}")
    return manifest


def restore(bundle: Path, root: Path, expected_sha256: str) -> None:
    if sha256_file(bundle) != expected_sha256:
        raise ValueError("Bundle SHA-256 differs from the published release manifest")
    root = root.resolve()
    with ZipFile(bundle) as archive:
        manifest = verify_members(archive)
        # Preflight all conflicts before creating files. Never replace differing
        # user data or implicitly combine different prepared dataset versions.
        for record in manifest["files"]:
            target = safe_destination(root, record["path"])
            if target.exists() and sha256_file(target) != record["sha256"]:
                raise ValueError(f"Existing data differ: {target}")
        for record in manifest["files"]:
            target = safe_destination(root, record["path"])
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = safe_destination(root, record["path"] + ".part")
            with archive.open(record["path"]) as source, temporary.open("wb") as dest:
                shutil.copyfileobj(source, dest, length=8 * 1024 * 1024)
            if sha256_file(temporary) != record["sha256"]:
                raise ValueError(f"Restore checksum mismatch: {target}")
            temporary.replace(target)
    if manifest["dataset"] == "taukadial":
        from .taukadial import ARCHIVES, extract_archive
        raw = root / "raw" / "taukadial"
        expected = json.loads((root / "reports/taukadial/file_hashes.json").read_text())
        found = {}
        for split, name in ARCHIVES.items():
            found.update(extract_archive(raw / "archives" / name, raw, split))
        if found != expected:
            raise ValueError("Restored TAUKADIAL audio differs from the recorded audit")
    print(f"Restored and verified {manifest['dataset']} under {root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    create = subs.add_parser("build")
    create.add_argument("--dataset", choices=["text", "taukadial"], required=True)
    create.add_argument("--data-root", type=Path, default=Path("data"))
    create.add_argument("--output", type=Path, required=True)
    install = subs.add_parser("restore")
    install.add_argument("bundle", type=Path)
    install.add_argument("--sha256", required=True)
    install.add_argument("--data-root", type=Path, default=Path("data"))
    check = subs.add_parser("verify")
    check.add_argument("bundle", type=Path)
    check.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.action == "build":
        build(args.data_root, args.output, args.dataset)
    elif args.action == "restore":
        restore(args.bundle, args.data_root, args.sha256)
    else:
        if sha256_file(args.bundle) != args.sha256:
            raise ValueError("Bundle SHA-256 mismatch")
        with ZipFile(args.bundle) as archive:
            manifest = verify_members(archive)
        print(f"Verified {len(manifest['files'])} files in {args.bundle.name}")


if __name__ == "__main__":
    main()
