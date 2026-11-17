# TAUKADIAL speech inputs, version 2

This page documents the historical v2 release. New experiments use [speech v3](speech_preprocessing_v3.md), which adds conservative interval repair, quarantine and complete short-tail coverage.

The approved `speech_v2` condition uses the existing TAUKADIAL participant-candidate audio and existing, independently decoded Whisper-large-v3 word timestamps. It applies WhisperCARE's verified window policy, then matches ASR tokens locally to each window. It does **not** use GPT to select human-reference spans, and it does not certify the candidate audio as manually verified participant-only speech.

## What matches WhisperCARE

The primary reference is [`SpeechCARE/WhisperCARE`](https://github.com/SpeechCARE/WhisperCARE) at commit `abc2fe7074aa27e0cf1a6b72b07a98ed7440e1d9`, especially `preprocessing/preprocess.py` and `utils/constants.py`:

- 30-second windows; 3-second overlap; 27-second stride.
- Keep the final window if it contains at least 15 seconds. Drop a shorter final tail. An entire recording shorter than 15 seconds produces no window.
- Save retained windows at their actual lengths. Do not pad the WAVs. The training collator pads them to 30 seconds and uses `num_samples` to identify the valid extent.
- Export mono 16 kHz PCM16 WAVs. Preserve the existing participant split before windowing.

For example, a 42-second recording produces `[0,30]` and `[27,42]`; a 41-second recording produces only `[0,30]`. A 60-second recording produces `[0,30]` and `[27,57]`; its `[54,60]` candidate tail is dropped. The excluded-tail list includes overlaps with retained windows and must not be interpreted as unique omitted duration.

## Deliberate differences and source limitations

WhisperCARE's window script loads its already prepared task audio with `librosa.load(..., sr=16000, mono=True)`. The approved TAUKADIAL candidates were previously normalized using `scipy.signal.resample_poly`, saved as mono 16 kHz PCM16, and cut using reference-assisted participant intervals. This version preserves those exact candidate samples; it does not repeat resampling, filtering, loudness normalization, or VAD. The current primary WhisperCARE repository does not specify an upstream VAD, participant-extraction, or low-pass-filter implementation, so none is inferred here.

There are three existing extraction conditions: 221 English recordings use reference-assisted CHAT timing, 25 English recordings use reference-assisted CTC candidates with unresolved source-task scope, and 261 Mandarin recordings use acoustic participant candidates. Manual speaker-purity review is incomplete. Existing candidate metadata marks 104 recordings with unresolved speaker intervals; 17 of the 25 English CTC candidates omit low-confidence source turns. Cuts can remove between-turn pauses; pauses inside retained source intervals remain unchanged. This condition is not appropriate for claims that require an intact original pause distribution.

The older MCIVoice matching script `dataset/build_matched_transcripts.py` used Whisper-medium segment ASR plus GPT-4o to find a contiguous human-reference span. `speech_v2` instead reuses reference-blind ASR from `mlx-community/whisper-large-v3-mlx`, revision `49e6aa286ad60c14352c404340ded53710378a11`. The archived decoding settings are temperature 0, no initial prompt, no conditioning on previous text, word timestamps, and overlapping ASR windows with midpoint ownership. No ASR, GPT, or paid GPU/API job runs during preparation.

A token belongs to a model window when its parent-audio timestamp midpoint falls within that window. Overlap words can appear in both neighboring model windows. Words crossing a boundary are flagged and their window-relative times are clipped. These are **ASR transcripts matched by time**, not corrected human transcripts or forced-alignment gold labels. An empty match remains empty and receives a quality flag; neighboring text or a full-recording transcript is never substituted. The current completed export has zero empty matches.

## Portable directory contract

All paths below are relative to the data root and remain outside Git:

```text
data/
  sources/speech_v2/
    audio/<recording_id>.wav       # exact, complete candidate recordings
    records/<recording_id>.json    # sanitized ASR words, original-time map, hashes, flags
    manifest.jsonl
    provenance.json               # source manifest/run hashes and explicit preprocessing
  processed/speech_v2/
    audio/{train,validation,test}/<segment_id>.wav
    train.csv
    validation.csv
    test.csv
    segments.jsonl                # CSV data plus matched word timestamps
    excluded_tails.jsonl          # every omitted tail / short whole recording
    unresolved_cases.jsonl        # per-recording quality flags and alignment issues
  reports/speech_v2/
    audit.json
    file_hashes.json
```

The model-facing CSV has `segment_id`, `recording_id`, `participant_id`, `audio_path`, `language`, `split`, `start_seconds`, `end_seconds`, `duration_seconds`, `num_samples`, `sample_rate`, `channels`, `sha256`, `source_sha256`, `original_wav_sha256`, `matched_transcript`, `match_method`, `transcript_source`, `speaker_condition`, `quality_flags`, and `original_time_pieces`. Only train and validation include `label` (`0=NC`, `1=MCI`). `audio_path` is relative to the **data root**, not the CSV directory. `sha256` hashes the complete derived WAV file; `source_sha256` hashes its complete candidate parent. Array fields are JSON-encoded inside CSV cells.

`start_seconds`/`end_seconds` use the concatenated candidate-audio clock. `original_time_pieces` maps every window back to possibly discontinuous original recording intervals. Do not treat a candidate start time as an original recording start time.

Source metadata are exported through an explicit field allowlist. Original source filenames that may encode diagnoses, absolute workstation paths, prior prediction results, and human reference transcripts are excluded. The portable source package is sufficient to regenerate every window without the original workstation project. Official raw TAUKADIAL files remain in the version-1 package. The version-2 speech package includes its required recording-level split CSVs and a separate copy of locked test labels; preprocessing never reads the locked labels.

## Build and regenerate

For a first local import, point to the existing source project containing `data/taukadial_asr` and `outputs/taukadial_asr`:

```bash
PYTHONPATH=src python -m ccspeech.data.speech prepare \
  --data-root data \
  --source-project /path/to/existing/speech/project
```

After restoring `data/sources/speech_v2` and the three original recording-level split CSVs, regenerate locally without that source project:

```bash
PYTHONPATH=src python -m ccspeech.data.speech prepare --data-root data
```

The commands are deterministic and idempotent. They reject mismatched source hashes, test labels in the default manifest, cross-split participants, unsafe source paths, unexpected existing segment WAVs, and conflicting outputs. A conflicting existing artifact requires investigation or a new version directory; the command does not silently overwrite it. Install the repository's speech dependencies (`numpy`, `soundfile`) before preparing. Regeneration itself does not require a model download, `librosa`, or a GPU.

## Completed export and validation

| Split | Participants | Retained recordings | Windows | English / Mandarin windows |
|---|---:|---:|---:|---:|
| Train | 103 | 298 | 593 | 294 / 299 |
| Validation | 26 | 74 | 182 | 85 / 97 |
| Test | 40 | 117 | 271 | 152 / 119 |
| Total | 169 | 489 | 1,046 | 531 / 515 |

All 507 complete candidate recordings are preserved in the portable sources (8.0152 hours). Windowing excludes 18 recordings shorter than 15 seconds and 198 additional short final tails. All 169 participants retain at least one window. Retained windows contain 8.1029 hours when overlap is counted. These counts differ from the older 1,262-window source export because that export retained every short tail.

Preparation verifies candidate file hashes against both the source manifest and ASR input hashes; reconstructs each candidate's exact PCM from its normalized original intervals; verifies normalized-original hashes; checks all word times and original-time maps; reads every saved WAV back and compares its samples; checks participant disjointness and cross-split byte duplicates; and writes checksums for every portable source and processed artifact. The completed export has no cross-split byte duplicates and no empty matched transcripts. These are integrity checks, not evidence of correct speaker identity or correct ASR content.

Every recording carries the broad `speaker_purity_not_manually_verified` flag. For targeted review, filter `unresolved_cases.jsonl` by `source_task_scope_unverified`, `source_alignment_requires_review`, `unresolved_speaker_intervals`, or `low_confidence_source_turns_omitted`. The missing historical English `reference_scope_review_complete` field is represented conservatively as `reference_scope_review_not_recorded_complete`; it does not assert that all English task scopes failed verification. Keep these flags available for later sensitivity analyses and never choose preprocessing variants using locked test labels.

Tests cover the exact tail boundary, overlap token matching, source/segment identity, original-time-map correctness, corrupted same-size audio, path traversal, symlink safety, label leakage, and portable regeneration. Run:

```bash
PYTHONPATH=src python -m pytest tests/test_speech.py -q
```

Access to the dataset remains restricted. Keep the source audio, transcripts, participant-level manifests, and reports in the authorized private data package; Git contains code, tests, and aggregate documentation only.
