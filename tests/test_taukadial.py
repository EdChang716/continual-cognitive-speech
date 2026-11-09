"""Leakage/safety checks plus an optional local prepared-manifest regression."""
import csv
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from ccspeech.data.taukadial import assert_disjoint, check_participants, copy_verified, extract_archive, prepare, safe_member_path, split_participants


class TaukadialSafetyTests(unittest.TestCase):
    def test_subject_cannot_cross_partitions(self):
        with self.assertRaisesRegex(ValueError, "Participant leakage"):
            assert_disjoint({
                "train": [{"participant_id": "p1", "sha256": "a"}],
                "test": [{"participant_id": "p1", "sha256": "b"}],
            })

    def test_different_subject_ids_do_not_hide_duplicate_audio(self):
        with self.assertRaisesRegex(ValueError, "Identical audio"):
            assert_disjoint({
                "train": [{"participant_id": "p1", "sha256": "same"}],
                "validation": [{"participant_id": "p2", "sha256": "same"}],
            })

    def test_stratified_split_is_order_independent_and_covers_every_person(self):
        people = {
            f"p{group}-{i}": {"language": language, "label": label}
            for group, (language, label) in enumerate([("en", 0), ("en", 1), ("zh", 0), ("zh", 1)])
            for i in range(10)
        }
        assignment = split_participants(people, 0.2, 42)
        self.assertEqual(assignment, split_participants(dict(reversed(list(people.items()))), 0.2, 42))
        self.assertEqual(set(assignment), set(people))
        for language in ("en", "zh"):
            for label in (0, 1):
                members = [pid for pid, p in people.items() if (p["language"], p["label"]) == (language, label)]
                self.assertEqual(sum(assignment[p] == "validation" for p in members), 2)

    def test_three_recordings_must_share_diagnosis_and_language(self):
        rows = [{"participant_id": "taukdial-001", "recording_id": f"taukdial-001-{i}", "language": "en", "label": 0} for i in (1, 2, 3)]
        self.assertEqual(len(check_participants(rows)), 1)
        rows[2]["label"] = 1
        with self.assertRaisesRegex(ValueError, "Inconsistent participant"):
            check_participants(rows)

    def test_archive_traversal_and_absolute_paths_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("../outside.wav", "/tmp/outside.wav", "TAUKADIAL-24/train/../../../../x", "a\\b"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    safe_member_path(name, Path(folder))

    def test_archive_link_rejected_without_writing_outside(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "bad.tgz"
            with tarfile.open(archive, "w:gz") as tar:
                member = tarfile.TarInfo("TAUKADIAL-24/train/taukdial-001-1.wav")
                member.type = tarfile.SYMTYPE
                member.linkname = "/tmp/sensitive"
                tar.addfile(member)
            with self.assertRaisesRegex(ValueError, "Unsupported archive entry"):
                extract_archive(archive, root / "out", "train")
            self.assertFalse((root / "out/TAUKADIAL-24/train/taukdial-001-1.wav").exists())

    def test_archive_duplicate_entry_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "duplicates.tgz"
            with tarfile.open(archive, "w:gz") as tar:
                for _ in range(2):
                    entry = tarfile.TarInfo("TAUKADIAL-24/train/groundtruth.csv")
                    entry.size = 1
                    tar.addfile(entry, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "Duplicate archive entry"):
                extract_archive(archive, root / "out", "train")

    def test_extraction_temporary_symlink_cannot_overwrite_outside_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            outside = root / "outside.csv"
            outside.write_bytes(b"preserve outside file")
            temporary = root / "out/TAUKADIAL-24/train/groundtruth.csv.part"
            temporary.parent.mkdir(parents=True)
            temporary.symlink_to(outside)
            archive = root / "temporary-link.tgz"
            with tarfile.open(archive, "w:gz") as tar:
                entry = tarfile.TarInfo("TAUKADIAL-24/train/groundtruth.csv")
                entry.size = 3
                tar.addfile(entry, io.BytesIO(b"new"))
            with self.assertRaisesRegex(ValueError, "escapes destination|Symlink"):
                extract_archive(archive, root / "out", "train")
            self.assertEqual(outside.read_bytes(), b"preserve outside file")
            self.assertFalse(temporary.with_suffix("").exists())

    def test_import_destination_and_temporary_symlinks_are_rejected(self):
        for filename in ("copied.csv", "copied.csv.part"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = root / "source.csv"
                source.write_bytes(b"new source")
                outside = root / "outside.csv"
                outside.write_bytes(b"preserve outside file")
                destination = root / "out/copied.csv"
                destination.parent.mkdir()
                (destination.parent / filename).symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "Symlink in import destination"):
                    copy_verified(source, destination)
                self.assertEqual(outside.read_bytes(), b"preserve outside file")

    def test_missing_optional_source_labels_cannot_leave_stale_locked_labels(self):
        for filename in ("testgroundtruth.csv", "test_labels.csv"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = root / "source"
                source.mkdir()
                stale = root / "data/locked/taukadial" / filename
                stale.parent.mkdir(parents=True)
                stale.write_bytes(b"previous held-out labels\n")
                with self.assertRaisesRegex(ValueError, "Existing locked test labels"):
                    # The guard runs before reading inputs or copying any audio.
                    prepare(source, root / "data", source / "language_map.csv")
                self.assertEqual(stale.read_bytes(), b"previous held-out labels\n")
                self.assertFalse((root / "data/reports/taukadial/audit.json").exists())

    def test_prepared_real_manifests_reproduce_the_recorded_participant_split(self):
        data = Path(__file__).resolve().parents[1] / "data"
        manifest_root = data / "processed/taukadial"
        audit_path = data / "reports/taukadial/audit.json"
        paths = [manifest_root / f"{name}.csv" for name in ("train", "validation")]
        if not audit_path.is_file() or not all(path.is_file() for path in paths):
            self.skipTest("Private prepared TAUKADIAL manifests are not installed")
        audit = json.loads(audit_path.read_text())
        rows = []
        for path in paths:
            with path.open(newline="", encoding="utf-8") as stream:
                rows.extend(csv.DictReader(stream))
        for row in rows:
            row["label"] = int(row["label"])
        people = check_participants(rows)
        expected = {row["participant_id"]: row["split"] for row in rows}
        kwargs = {"validation_fraction": audit["validation_fraction"], "seed": audit["seed"]}
        first = split_participants(people, **kwargs)
        second = split_participants(check_participants(list(reversed(rows))), **kwargs)
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual(len(people), 129)
        for split in ("train", "validation"):
            self.assertEqual(
                sum(value == split for value in first.values()),
                audit["partitions"][split]["participants"],
            )


if __name__ == "__main__":
    unittest.main()
