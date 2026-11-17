"""Local ASR's reference boundary, overlap clocks, integrity, and resumption."""
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from ccspeech.data import asr
from ccspeech.data.speech import ASR_REVISION, SAMPLE_RATE, json_bytes


def response(*words):
    return {"text": "".join(w["word"] for w in words), "segments": [
        {"start": 0., "end": 30., "text": "", "words": list(words)}]}


def word(text, start, end):
    return {"word": text, "start": start, "end": end, "probability": .9}


class ASRClockTests(unittest.TestCase):
    def test_complete_coverage_includes_short_audio_and_short_tails(self):
        self.assertEqual(asr.asr_windows(14 * SAMPLE_RATE), [(0, 14 * SAMPLE_RATE)])
        self.assertEqual(asr.asr_windows(30 * SAMPLE_RATE), [(0, 30 * SAMPLE_RATE)])
        self.assertEqual(asr.asr_windows(57 * SAMPLE_RATE), [
            (0, 30 * SAMPLE_RATE), (27 * SAMPLE_RATE, 57 * SAMPLE_RATE)])
        self.assertEqual(asr.asr_windows(60 * SAMPLE_RATE)[-1], (54 * SAMPLE_RATE, 60 * SAMPLE_RATE))

    def test_overlap_words_use_unique_clock_ownership_not_text_dedup(self):
        answers = iter([
            response(word(" same", 27.9, 28.1), word(" seam", 28.4, 28.6)),
            response(word(" same", .9, 1.1), word(" seam", 1.4, 1.6), word(" same", 4, 5)),
        ])
        seen = []

        def decode(audio, language):
            seen.append((len(audio), language))
            return next(answers)

        result = asr.transcribe_audio(np.zeros(42 * SAMPLE_RATE, dtype=np.float32), "zh", decode)
        self.assertEqual(result["text"], " same seam same")
        self.assertEqual([w["window_index"] for w in result["words"]], [0, 1, 1])
        self.assertAlmostEqual(result["words"][1]["start"], 28.4)
        self.assertEqual(seen, [(30 * SAMPLE_RATE, "zh"), (15 * SAMPLE_RATE, "zh")])

    def test_empty_output_is_explicit_and_never_fills_from_reference(self):
        result = asr.transcribe_audio(np.zeros(SAMPLE_RATE, dtype=np.float32), "en",
                                      lambda *_: {"text": "", "segments": []})
        self.assertEqual(result["words"], [])
        self.assertEqual(result["text"], "")
        self.assertEqual(result["quality_flags"], ["empty_asr_transcript"])

    def test_invalid_word_timestamp_is_not_silently_retained(self):
        with self.assertRaisesRegex(ValueError, "invalid"):
            asr.transcribe_audio(np.zeros(SAMPLE_RATE, dtype=np.float32), "en",
                                 lambda *_: response(word(" invalid", float("nan"), .2)))


class ASRResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        # Fake decoders must not wait on or interfere with real local GPU jobs.
        lock_patch = patch.object(asr, "_gpu_lock", side_effect=nullcontext)
        lock_patch.start()
        self.addCleanup(lock_patch.stop)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        (self.model / "weights.npz").write_bytes(b"unit-test-local-checkpoint")
        (self.model / "download_provenance.json").write_text(json.dumps({
            "repo": asr.MODEL_REPO, "revision": ASR_REVISION}))
        hashes = {name: hashlib.sha256((self.model / name).read_bytes()).hexdigest()
                  for name in ("config.json", "weights.npz")}
        self.model_patch = patch.object(asr, "MODEL_FILES", hashes)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.audio = self.root / "audio.wav"
        sf.write(self.audio, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
        self.row = {"recording_id": "recording-1", "language": "en", "audio_path": "audio.wav",
                    "sha256": hashlib.sha256(self.audio.read_bytes()).hexdigest(),
                    "participant_id": "participant-1", "split": "train"}
        self.manifest = self.root / "asr.jsonl"
        self.write_manifest([self.row])
        self.output = self.root / "asr_output"

    def write_manifest(self, rows):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def backend(self, _):
        return lambda audio, language: response(word(" spoken", .1, .3))

    def execute(self, **kwargs):
        return asr.run_asr(self.manifest, self.model, self.output,
                           backend_factory=kwargs.pop("backend_factory", self.backend), **kwargs)

    def test_forbidden_reference_fields_and_path_escape_are_rejected(self):
        for extra in ({"transcript": "secret reference"}, {"label": 1},
                      {"audio_path": "../outside.wav"}):
            self.write_manifest([{**self.row, **extra}])
            with self.assertRaises(ValueError):
                self.execute(backend_factory=lambda _: self.fail("Model must not load"))

    def test_corrupt_audio_and_checkpoint_fail_before_backend_load(self):
        self.audio.write_bytes(self.audio.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Source audio hash mismatch"):
            self.execute(backend_factory=lambda _: self.fail("Model must not load"))
        (self.model / "weights.npz").write_bytes(b"changed-model")
        with self.assertRaisesRegex(ValueError, "model hash mismatch"):
            self.execute(backend_factory=lambda _: self.fail("Model must not load"))

    def test_noncanonical_audio_format_is_rejected_before_backend_load(self):
        sf.write(self.audio, np.zeros(8000, dtype=np.float32), 8000)
        self.row["sha256"] = hashlib.sha256(self.audio.read_bytes()).hexdigest()
        self.write_manifest([self.row])
        with self.assertRaisesRegex(ValueError, "16kHz mono"):
            self.execute(backend_factory=lambda _: self.fail("Model must not load"))

    def test_resume_uses_hash_settings_and_does_not_load_backend(self):
        result = self.execute()
        self.assertEqual(result["new_recordings"], 1)
        cached_path = self.output / "recording-1.json"
        before = cached_path.read_bytes()
        result = self.execute(backend_factory=lambda _: self.fail("Cached run must not load backend"))
        self.assertEqual(result["reused_recordings"], 1)
        self.assertEqual(cached_path.read_bytes(), before)
        cached = json.loads(before)
        self.assertTrue(cached["reference_blind"])
        self.assertEqual(cached["source_audio_path"], "audio.wav")
        self.assertEqual(cached["run_settings_sha256"], hashlib.sha256((self.output / "run.json").read_bytes()).hexdigest())
        cached["run_settings_sha256"] = "wrong-settings"
        cached_path.write_bytes(json_bytes(cached))
        with self.assertRaisesRegex(ValueError, "settings changed"):
            self.execute(backend_factory=lambda _: self.fail("Invalid cache must not load backend"))

    def test_limit_is_explicit_and_full_resume_finishes_remaining_recording(self):
        second = {**self.row, "recording_id": "recording-2"}
        self.write_manifest([self.row, second])
        first = self.execute(limit=1)
        self.assertEqual(first["state"], "selected_limit_complete")
        resumed = self.execute()
        self.assertEqual(resumed["state"], "complete")
        self.assertEqual(resumed["reused_recordings"], 1)
        self.assertEqual(resumed["new_recordings"], 1)

    def test_cached_duration_and_words_are_checked_against_physical_audio(self):
        self.execute()
        path = self.output / "recording-1.json"
        original = path.read_text()
        for duration, start, end, message in (
            (100., 50., 51., "physical source audio"),
            (1., 50., 51., "outside the audio clock"),
            (float("nan"), .1, .3, "physical source audio"),
        ):
            with self.subTest(duration=duration, start=start):
                cached = json.loads(original)
                cached["duration_seconds"] = duration
                cached["words"][0].update(start=start, end=end)
                path.write_text(json.dumps(cached))
                with self.assertRaisesRegex(ValueError, message):
                    self.execute(backend_factory=lambda _: self.fail("Invalid cache must not load backend"))
        path.write_text(original)
        resumed = self.execute(backend_factory=lambda _: self.fail("Valid cache must remain reusable"))
        self.assertEqual(resumed["reused_recordings"], 1)

    def test_interrupted_temporary_file_is_not_treated_as_completed_output(self):
        self.output.mkdir()
        (self.output / ".asr-interrupted.tmp").write_text("partial JSON")
        result = self.execute()
        self.assertEqual(result["new_recordings"], 1)
        self.assertEqual(json.loads((self.output / "recording-1.json").read_text())["text"], " spoken")

    def test_backend_load_failure_marks_failed_without_copying_error_text(self):
        def unavailable(_):
            raise RuntimeError("private clinical text and local secrets")

        with self.assertRaisesRegex(RuntimeError, "private clinical"):
            self.execute(backend_factory=unavailable)
        status_path = self.output / "status.json"
        status = json.loads(status_path.read_text())
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["error_type"], "RuntimeError")
        self.assertFalse(status["interrupted"])
        self.assertEqual(status["completed_recordings"], 0)
        self.assertNotIn("private clinical", status_path.read_text())
        resumed = self.execute()
        self.assertEqual(resumed["state"], "complete")
        self.assertEqual(resumed["new_recordings"], 1)
        self.assertNotIn("error_type", json.loads(status_path.read_text()))

    def test_keyboard_interruption_preserves_completed_count_and_resume(self):
        self.write_manifest([self.row, {**self.row, "recording_id": "recording-2"}])
        calls = 0

        def backend(_):
            def decode(audio, language):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt()
                return response(word(" spoken", .1, .3))
            return decode

        with self.assertRaises(KeyboardInterrupt):
            self.execute(backend_factory=backend)
        status = json.loads((self.output / "status.json").read_text())
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["error_type"], "KeyboardInterrupt")
        self.assertTrue(status["interrupted"])
        self.assertEqual(status["completed_recordings"], 1)
        self.assertEqual(status["new_recordings"], 1)
        first = (self.output / "recording-1.json").read_bytes()
        resumed = self.execute()
        self.assertEqual(resumed["state"], "complete")
        self.assertEqual(resumed["completed_recordings"], 2)
        self.assertEqual(resumed["reused_recordings"], 1)
        self.assertEqual(resumed["new_recordings"], 1)
        self.assertEqual(first, (self.output / "recording-1.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
