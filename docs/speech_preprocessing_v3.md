# TAUKADIAL participant inputs, version 3

Version3 keeps TAUKADIAL as the only waveform source. Chou/Delaware references support extraction checks and quality comparisons; their audio never replaces TAUKADIAL model inputs. The release contains all507 source recordings, with482 eligible participant-candidate recordings and25 English recordings kept separately for review. The eligible set represents167 of169 participants. These are evidence-supported automated candidates, not manually certified speaker labels.

## Completed release

| Split | Windows | Recordings | Participants | English / Mandarin windows |
|---|---:|---:|---:|---:|
| Train | 692 | 295 | 102 | 321 / 371 |
| Validation | 196 | 73 | 26 | 80 / 116 |
| Test | 303 | 114 | 39 | 161 / 142 |
| Total | 1,191 | 482 | 167 | 562 / 629 |

All149 changed recordings completed fresh local ASR without an empty full-recording output. Of the1,191 model windows,201 are shorter than15seconds. One6.132second Mandarin validation tail has no matched ASR words; its audio is retained and the empty-text flag is preserved. No reference text is substituted. The current speech model uses audio, so an empty inspection transcript does not remove its input.

## Processing order

1. Preserve the existing official-test and participant-disjoint train/validation assignments.
2. Verify original TAUKADIAL WAV hashes; convert to mono16kHz PCM16 using the pinned normalization procedure.
3. Keep existing supported participant intervals and apply the conservative repair below. Retain original-to-cleaned time maps and unresolved intervals.
4. Transcribe changed complete candidate recordings locally; reuse existing ASR only for byte-identical unchanged audio.
5. Export30second windows with27second stride. Keep every nonempty short recording and final tail, at its real WAV length. Whisper padding happens during model collation.
6. Match each window's text using the ASR word midpoints on that same cleaned-audio clock. Preserve piecewise original-time coordinates and boundary flags.
7. Audit content against the user's final references on matched content scopes. Keep speaker selection, recognition errors and window coverage separate.

## Frozen participant policy

English uses the existing source-assisted timing resolutions:221 recordings pass both reference-scope and extraction-timing checks;25 remain quarantined. Their previous candidate audio/text are retained for inspection but never appear in the default train/validation/test window CSVs. One training participant and one test participant consequently have no eligible English input. The remaining original split assignments are unchanged.

Some inherited source flags still say `reference_scope_review_not_recorded_complete`: this records missing metadata in the original candidate export, not failure of the newer221-recording scope/timing check. The explicit v3 `eligibility` and `exclusion_reasons` fields determine model inclusion; historical flags remain for provenance.

Mandarin starts from the existing strict participant candidates. A previously omitted or partially clipped ASR word can be recovered only if its complete time interval:

- lies within a source-reference match of at least six normalized characters;
- is covered throughout by at least two acoustic windows;
- has every overlapping acoustic window labeled PAR and a gallery similarity at least0.46845851570367814, the already frozen training-only threshold.

At most150ms padding is added, bounded by the lexical anchor and continuous acoustic support. Existing retained intervals and their internal pauses remain intact. The recovery rule never removes existing samples. Previously coarse-unknown speech may be recovered only after these stronger checks; unsupported unknown speech remains outside the candidate and in the review list. Known interviewer exclusions are recorded separately. This is reference-assisted extraction, not independent diarization validation.

The build changed149 Chou recordings, adding110.622seconds of original TAUKADIAL samples. The other358 recordings retain identical audio and ASR. Both old and repaired PCM were checked against their exact normalized-original intervals. Model input/reference versions and source evidence hashes are recorded in the portable metadata. No policy threshold was selected from test CER or diagnoses.

## Transcription and matching

Changed audio is re-decoded with the existing local `mlx-community/whisper-large-v3-mlx` checkpoint, revision `49e6aa286ad60c14352c404340ded53710378a11`. Model files are hash-pinned. No reference transcript, diagnosis, or previous transcript is supplied to Whisper. The [explicit local ASR runner](local_asr.md) uses temperature0, word timestamps, no initial prompt and no previous-text conditioning.

ASR covers the complete recording using30second/27second windows, including short tails. Midpoint ownership handles overlapping decodes approximately: timestamp errors can still duplicate or omit a word near an ASR boundary. This differs from deduplicating the same already-generated word index across downstream model windows, which is exact. These transcripts are ASR output, not manually corrected text. Empty matches remain empty and flagged.

## Use and regenerate the downloaded release

The normal workflow needs neither old workstation files nor a new ASR run:

```bash
bash scripts/download_from_drive.sh gdrive 3 speech_v3
python scripts/restore_data.py --release 3 --dataset speech_v3
python -m ccspeech.data.speech_v3 prepare --data-root data
python -m ccspeech preflight --config configs/experiments/speech_derpp.json
```

The bundle includes portable complete candidates, timed words, interval maps, quarantine decisions, original recording splits and separately locked test labels. Window regeneration verifies hashes and samples and refuses conflicting existing outputs. It never upgrades a quarantined recording to eligible.

| Location under `data/` | Purpose |
|---|---|
| `processed/speech_v3/{train,validation,test}.csv` | Default model inputs; audio paths are relative to the data root. |
| `processed/speech_v3/audio/` | Actual-length16kHz mono WAV windows. |
| `processed/speech_v3/segments.jsonl` | Timed window words and original-time pieces. |
| `processed/speech_v3/quarantined_recordings.jsonl` | Review-only recordings, reasons, source audio paths and words. |
| `sources/speech_v3/records/` | Complete candidate word streams, source hashes, time maps and interval review flags. |
| `sources/speech_v3/sidecars/` | Recovery decisions and source evidence hashes. |
| `reports/speech_v3/audit.json` | Actual release counts, exclusions, coverage and integrity checks. |
| `reports/speech_v3/quality/` | Versioned WER/CER counts and train/validation review priorities. |

Every speech experiment and suite now explicitly selects `processed/speech_v3`. To reproduce v2, restore with `--release 2 --dataset speech_v2` and set the configuration's `data.speech_dir` to `processed/speech_v2`. Do not pool versions or compare methods run on different source versions. Checkpoints/config hashes keep their original data contract.

## Rebuild upstream decisions on the authorized source workstation

This advanced path requires the earlier hashed role/reference/ASR artifacts. It is unnecessary for using or regenerating the downloadable model windows:

```bash
python -m ccspeech.data.participant build \
  --data-root data \
  --legacy-root /path/to/source-project \
  --normalized-originals /path/to/hash-verified/normalized-TAUKADIAL \
  --output data/sources/speech_v3

# Run in the local MLX environment; this is an explicit inference step.
python -m ccspeech.data.asr \
  --manifest data/sources/speech_v3/asr_pending.jsonl \
  --model-path /path/to/pinned/whisper-large-v3-mlx \
  --output data/reports/speech_v3/asr_changed

python -m ccspeech.data.participant finalize \
  --data-root data --output data/sources/speech_v3 \
  --asr-results data/reports/speech_v3/asr_changed
python -m ccspeech.data.speech_v3 prepare --data-root data
```

Finalization verifies the fresh ASR source/model hashes and unchanged reuse against the original pinned v2 records. A pending or failed ASR job cannot be presented as a completed source release. Original audio, v2 packages and references remain available; data revisions do not overwrite them.

## Quality interpretation

English is evaluated with WER and Chinese with CER. Comparisons hold the reference text fixed across raw, v2 and v3 ASR. Only scope-verified English references enter the primary aggregate. Current-window text is unioned by source word index, preserving genuine repetition and avoiding double counting overlap. With v3's complete window coverage, that union should equal the complete v3 candidate transcript.

The review queue prioritizes discrepancies on train/validation only; test quality figures are descriptive and do not choose preprocessing thresholds. Reference agreement does not alone establish speaker purity, and original pauses across removed turns are not preserved. Review interval flags and speaker uncertainty remain part of the release rather than being converted into claims of manual verification.

Measured validation comparison, using the same reference for each condition:

| Language / metric | Recordings | Raw recording ASR | v2 complete candidate | v3 complete candidate / window union |
|---|---:|---:|---:|---:|
| Mandarin / CER | 42 | 14.41% | 15.65% | 14.36% |
| English / WER | 31 | 11.41% | 9.11% | 9.11% |

The raw condition may include interviewer speech and is not a speaker-matched ASR benchmark. Across all splits, Mandarin CER is14.79% for v3 versus15.45% for v2; English WER is unchanged at9.39%. These all-split figures are descriptive only. Some recordings improve and some worsen: among the25 changed validation recordings,17 have lower error,3 equal error and5 higher error. All482 eligible recordings have exact equality between the complete v3 transcript word indices and the deduplicated model-window union. None of these measurements establishes downstream classification gains or human-verified participant purity.
