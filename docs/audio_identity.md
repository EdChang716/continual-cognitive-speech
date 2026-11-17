# Original audio correspondence and transcript clocks

The Chou/Delaware source files can anchor a TAUKADIAL extraction audit, but recording identity, transcript timing, and participant-only speech are separate claims. A matching waveform is not a speaker annotation. A CHAT file must refer to the correct media clock before its participant (`*PAR`) intervals can be transferred.

## Reproduce the private audit

```bash
python -m pip install -e '.[audit]'
python -m ccspeech.data.audio_identity \
  --data-root data \
  --legacy-audit-root /path/to/source-project/data/overlap_audit \
  --chat-resolutions /path/to/source-project/data/taukadial_asr/reference_resolution.jsonl \
  --output data/reports/audio_identity_v1
```

The source-project inputs are the existing authorized recording crosswalk, waveform inventory, processed-task anchor audit, and CHAT clock resolutions. They are not included in the public repository or speech training bundle. Output CSV/JSON files remain under ignored `data/reports/`.

For materialized pairs, this audit checks the current TAUKADIAL file hash and both decoded-PCM hashes against their archived manifests. It independently recomputes waveform correspondence after mono conversion and resampling to 2 kHz. A fixed shift requires complete TAUKADIAL coverage, full correlation ≥0.98, and five distributed nonsilent one-second anchors with correlation ≥0.98 and offset spread ≤2 ms. These are diagnostic thresholds, not manual annotation. Gain/DC residuals and exact decoded-PCM equality are recorded separately. Cloud placeholders are skipped without being hydrated; cached evidence is never reported as a fresh comparison.

## Completed audit — 2026-11-17

| Source relationship | Prior crosswalk | Fresh physical comparisons | Result |
|---|---:|---:|---|
| Chou original recording | 261 | 216 | All fresh pairs have equal duration, zero offset, and full resampled correlation ≥0.99999995. |
| Delaware longer raw session | 240 | 202 | All fresh pairs support the recorded constant crop offset; full resampled correlation ≥0.999765. |
| Delaware edited task recording | 6 | 3 | Fresh checks confirm corresponding local anchors, not complete raw-session identity or one global shift. |

In total, 421 pairs were freshly hash/waveform checked. For 86 pairs, one or both files were macOS cloud placeholders, so only prior evidence is available. The 418 fresh constant-shift pairs had zero offset spread across the five searched anchors. Sixteen fresh Chou pairs have identical decoded PCM; the prior inventory identifies 21 across all 261. Near-perfect resampled correlation does not establish full-band bit identity for the other pairs.

Prior evidence across all 240 Delaware raw-session matches places the TAUKADIAL clip at 1.910–381.765 seconds into the source (median 79.745 seconds). These are excerpts from longer sessions, not merely the same complete recording with a few seconds of leading silence. Across the six edited-task matches, anchor offsets vary within each recording by 2.314–10.286 seconds; do not apply a single timestamp shift.

**The CHAT clock can differ from the matched WAV clock.** One previously resolved English case has a source-WAV offset of 260.0765 seconds but a CHAT offset of 6.0485 seconds. Prior dense word anchors support the latter clock; substituting the waveform offset would misalign the transcript by 254.028 seconds. Of 246 English references, 221 previously passed both scope and extraction-timing checks; 25 remain unresolved. This audit preserves those statuses rather than treating them as newly reviewed speaker labels.

## What transcripts are actually available

Fresh inventories of the user's official challenge archives found 387 train WAV files plus `groundtruth.csv`, and 120 test WAV files plus `README.md`. No accompanying transcript files were present; the train metadata header has no transcript column. A [TAUKADIAL study](https://www.apsipa.org/proceedings/2025/papers/APSIPA2025_P335.pdf) also explicitly describes the challenge audio as lacking accompanying transcriptions.

The source corpora are different: [Chou's official TalkBank page](https://talkbank.org/dementia/access/Mandarin/Chou.html) offers transcripts. However, the local Chou dataset, matching project directories, and available Chou ZIP archives contain WAV files and full-text CSV references, with no CHAT files. These CSVs have no utterance/word start and end times. Delaware has matched local CHAT references, whose media clocks still require verification.

## Extraction implications

For an aligned CHAT clock, use `t_TAUKADIAL = t_CHAT - CHAT_offset`, intersect participant intervals with the TAUKADIAL clip, and preserve an explicit source-to-output time map when removing interviewer turns. A simple leading offset is correctable. Edited audio requires piecewise correspondence or the exact media version named by the CHAT file. Do not replace a TAUKADIAL picture-description clip with an entire Delaware interview.

For Chou, the full-text reference can guide alignment and listening, but cannot by itself define participant boundaries. Obtain matching timed annotations or validate newly aligned speaker intervals before calling the result participant-only ground truth. Compare extraction variants on training/validation recordings; use manually reviewed speaker boundaries to assess participant loss and interviewer contamination. Text error alone cannot identify the best extraction, especially when the same reference text helped select the segments.

This audit did not rerun ASR, change training audio/transcripts, read locked test diagnoses, or upload human references. The current release remains participant-candidate audio with quality flags.
