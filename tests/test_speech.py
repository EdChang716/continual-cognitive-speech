"""WhisperCARE window policy, transcript identity, and source-time mapping."""
import hashlib
import csv
import json
from pathlib import Path
import tempfile
import unittest

from ccspeech.data.speech import (
    load_recording_splits, match_reference_span, match_word_timestamps,
    original_time_pieces, prepare_windows, stable_write, validate_time_map,
    verify_candidate_pcm, window_bounds,
)


class SpeechPreparationTests(unittest.TestCase):
    def test_short_recording_is_not_silently_padded_into_a_segment(self):
        kept, dropped = window_bounds(14 * 16000)
        self.assertEqual(kept, [])
        self.assertEqual(dropped, [(0, 14 * 16000)])

    def test_exact_window_has_no_redundant_overlap_tail(self):
        self.assertEqual(window_bounds(30 * 16000), ([(0, 30 * 16000)], []))
        self.assertEqual(window_bounds(57 * 16000), ([(0, 30 * 16000), (27 * 16000, 57 * 16000)], []))

    def test_tail_threshold_and_stride_match_whispercare(self):
        self.assertEqual(window_bounds(42 * 16000), ([(0, 30 * 16000), (27 * 16000, 42 * 16000)], []))
        self.assertEqual(window_bounds(41 * 16000), ([(0, 30 * 16000)], [(27 * 16000, 41 * 16000)]))
        kept, dropped = window_bounds(60 * 16000)
        self.assertEqual(kept, [(0, 480000), (432000, 912000)])
        self.assertEqual(dropped, [(864000, 960000)])

    def test_matching_uses_timing_and_flags_boundary_words(self):
        words = [
            {"word": " first", "start": 26.0, "end": 26.5},
            {"word": " overlap", "start": 27.0, "end": 28.0},
            {"word": " boundary", "start": 29.5, "end": 30.2},
            {"word": " later", "start": 35.0, "end": 35.5},
        ]
        left = match_word_timestamps(words, 0.0, 30.0, 42.0)
        right = match_word_timestamps(words, 27.0, 42.0, 42.0)
        self.assertEqual(left["matched_transcript"], "first overlap boundary")
        self.assertEqual(right["matched_transcript"], "overlap boundary later")
        self.assertIn("word_crosses_segment_boundary", left["quality_flags"])
        self.assertEqual(left["transcript_source"], "independent_whisper_large_v3_asr")

    def test_empty_asr_does_not_borrow_unspoken_neighbor_text(self):
        result = match_word_timestamps([], 0.0, 20.0, 20.0)
        self.assertEqual(result["matched_transcript"], "")
        self.assertIn("empty_asr_transcript", result["quality_flags"])

    def test_wrong_audio_hash_cannot_reuse_reference_match(self):
        reference = "One original sentence."
        match = {"segment_id": "seg1", "audio_sha256": "correct", "whisper_model": "medium", "matching_model": "gpt-4o", "reference_text_sha256": hashlib.sha256(reference.encode()).hexdigest(), "matched_transcript": "original"}
        self.assertEqual(match_reference_span(match, "seg1", "correct", reference)["matched_transcript"], "original")
        with self.assertRaisesRegex(ValueError, "exact segment audio"):
            match_reference_span(match, "seg1", "wrong", reference)
        match["matched_transcript"] = "invented phrase"
        with self.assertRaisesRegex(ValueError, "verbatim"):
            match_reference_span(match, "seg1", "correct", reference)

    def test_source_clock_uses_concat_map_not_clean_clock(self):
        mapping = [
            {"clean_start": 0.0, "clean_end": 10.0, "source_start": 5.0, "source_end": 15.0},
            {"clean_start": 10.0, "clean_end": 30.0, "source_start": 20.0, "source_end": 40.0},
        ]
        pieces = original_time_pieces(mapping, 8.0, 15.0)
        self.assertEqual([(p["original_start_seconds"], p["original_end_seconds"]) for p in pieces], [(13.0, 15.0), (20.0, 25.0)])
        with self.assertRaisesRegex(ValueError, "cover"):
            original_time_pieces(mapping[:1], 8.0, 15.0)

    def test_candidate_pcm_audit_detects_wrong_source_interval(self):
        import numpy as np

        source = np.arange(100, dtype=np.int16)
        mapping = [{"source_start": 10 / 16000, "source_end": 30 / 16000,
                    "clean_start": 0, "clean_end": 20 / 16000}]
        verify_candidate_pcm(source[10:30], source, mapping)
        with self.assertRaisesRegex(ValueError, "PCM differs"):
            verify_candidate_pcm(source[11:31], source, mapping)
        mapping[0]["clean_start"] = 1 / 16000
        with self.assertRaisesRegex(ValueError, "gap"):
            validate_time_map(mapping, 20, len(source))

    def test_stable_write_rejects_symlink_without_touching_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.write_bytes(b"untouched")
            output = root / "output"
            output.with_name("output.part").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "Temporary"):
                stable_write(output, b"new")
            self.assertEqual(outside.read_bytes(), b"untouched")


class SpeechPortableIntegrationTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import soundfile as sf

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "sources" / "speech_v2"
        (self.source / "records").mkdir(parents=True)
        (self.source / "audio").mkdir()
        split_dir = self.root / "processed" / "taukadial"
        split_dir.mkdir(parents=True)
        portable = []
        for i, split in enumerate(("train", "validation", "test"), 1):
            rid, pid = f"taukdial-{i:03d}-1", f"taukdial-{i:03d}"
            samples = np.random.default_rng(i).integers(-2000, 2000, 42 * 16000, dtype=np.int16)
            audio = self.source / "audio" / f"{rid}.wav"
            sf.write(audio, samples, 16000, subtype="PCM_16")
            digest = hashlib.sha256(audio.read_bytes()).hexdigest()
            known = {"recording_id": rid, "participant_id": pid, "language": "en", "split": split,
                     "sha256": f"raw{i}", "audio_path": f"raw/{rid}.wav"}
            if split != "test":
                known["label"] = i % 2
            with (split_dir / f"{split}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(known))
                writer.writeheader()
                writer.writerow(known)
            item = {"recording_id": rid, "participant_id": pid, "language": "en", "audio_path": f"audio/{rid}.wav",
                    "sha256": digest, "original_wav_sha256": f"raw{i}", "num_samples": len(samples),
                    "normalized_original_num_samples": len(samples) + 16000,
                    "time_map": [{"source_start": 1.0, "source_end": 43.0, "clean_start": 0, "clean_end": 42.0}],
                    "words": [{"word": " spoken", "start": 27, "end": 28}],
                    "quality_flags": ["speaker_purity_not_manually_verified"], "task_scope_verified": True,
                    "manual_speaker_qa_complete": False, "alignment_issues": [], "omitted_low_confidence_turns": [],
                    "extraction_condition": "test_candidate"}
            (self.source / "records" / f"{rid}.json").write_text(json.dumps(item))
            portable.append({k: item[k] for k in ("recording_id", "participant_id", "language", "audio_path", "sha256", "original_wav_sha256")})
        (self.source / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in portable))

    def test_portable_preparation_is_reproducible_and_label_safe(self):
        first = prepare_windows(self.root)
        second = prepare_windows(self.root)
        self.assertEqual(first, second)
        self.assertEqual(first["retained_segments"], 6)
        self.assertEqual(first["retained_participants"], 3)
        output = self.root / "processed" / "speech_v2"
        with (output / "test.csv").open() as stream:
            reader = csv.DictReader(stream)
            self.assertNotIn("label", reader.fieldnames)
            test = list(reader)
        self.assertEqual([float(row["duration_seconds"]) for row in test], [30.0, 15.0])
        self.assertEqual(test[0]["matched_transcript"], "spoken")
        self.assertFalse(Path(test[0]["audio_path"]).is_absolute())
        self.assertNotIn(str(self.root), (output / "segments.jsonl").read_text())

    def test_same_size_source_corruption_is_rejected_before_use(self):
        path = self.source / "audio" / "taukdial-001-1.wav"
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            prepare_windows(self.root)

    def test_portable_manifest_path_cannot_escape_sources(self):
        path = self.source / "records" / "taukdial-001-1.json"
        item = json.loads(path.read_text())
        item["audio_path"] = "../outside.wav"
        path.write_text(json.dumps(item))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            prepare_windows(self.root)

    def test_test_labels_are_refused_even_when_raw_audio_is_available(self):
        path = self.root / "processed" / "taukadial" / "test.csv"
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["label"] = 1
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "must not contain labels"):
            load_recording_splits(self.root)


if __name__ == "__main__":
    unittest.main()
