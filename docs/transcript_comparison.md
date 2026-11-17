# Source-transcript comparison

Use the existing identity-verified **Chou (Mandarin) and Delaware (English)** references to audit TAUKADIAL text coverage. The user confirmed that Delaware was the intended English source. The recording crosswalk does not establish TAUKADIAL-to-ADReSSo matches; dataset names are not interchangeable.

## Run locally

```bash
python -m pip install -e '.[audit]'
python -m ccspeech.data.transcript_audit \
  --data-root data \
  --reference-manifest /path/to/source-project/data/taukadial_asr/reference_manifest.jsonl \
  --resolutions /path/to/source-project/data/taukadial_asr/reference_resolution.jsonl \
  --raw-asr-dir /path/to/source-project/outputs/taukadial_asr/raw_whisper_v1 \
  --output data/reports/transcript_comparison_v1
python -m ccspeech.data.transcript_report \
  --report-dir data/reports/transcript_comparison_v1
```

These archived reference/raw-ASR inputs come from the authorized earlier project; the training-data v2 bundle deliberately excludes human references. The audit checks that the reference manifest matches the SHA256 stored in the speech release's provenance. It verifies recording/participant identities, original and candidate audio-hash bindings, raw ASR model settings, and equality of every current window transcript to its timestamp-selected source words. It does not rerun ASR, modify audio/text, read locked diagnoses, choose preprocessing settings, or contact an external API.

The output remains private under `data/reports/`:

- `review.html`: self-contained comparison with language/split/scope/recording filters and colored text differences; no external assets or network requests.
- `per_recording.csv`: one row per recording and condition, including substitutions, deletions, insertions, reference denominator and error rate.
- `differences.jsonl`: original references/ASR plus normalized alignment operations.
- `summary.json`: grouped counts/rates, checks and input hashes.

## What is compared

Every recording uses the same complete source reference for all three conditions:

1. **Raw:** ASR from the original recording, potentially including interviewer speech.
2. **Candidate:** ASR from the complete previously extracted participant-candidate audio.
3. **Window:** words covered by the current retained30second/27second-stride windows, unioned by original ASR word index. Genuine repeated words remain; overlap is not double counted. Recordings entirely omitted by the15second minimum receive empty hypotheses, rather than disappearing from coverage statistics.

English references use the previously resolved clip-specific participant text. The221 recordings whose reference scope passed prior pipeline checks and the25 unresolved recordings are reported separately. “Verified scope” does not imply new manual speaker annotation. Mandarin uses the Chou full-recording participant text, which does not supply word timestamps; this comparison does not create per-window human gold transcripts.

Scoring matches the earlier audit's convention: remove CHAT annotations, apply NFKC and lowercase, retain repetitions and filled pauses; English is tokenized into Unicode words with apostrophes, and Mandarin is converted traditional-to-simplified for scoring only and counted as alphanumeric characters. Original texts remain unchanged. Micro WER/CER is Σ(S+D+I)/Σ(reference units); lower is better and rates can exceed100%.

## Completed comparison — 2026-11-17

All507 recordings and169participants were compared. Current window transcripts matched their archived word sources exactly. The table pools the derived train/validation/test splits descriptively; the private report also keeps each split separate. It is not used to select a preprocessing variant on the test set.

| Reference group | Recordings | Raw audio ASR | Candidate audio ASR | Retained-window text vs full reference |
|---|---:|---:|---:|---:|
| Delaware, scope checked — WER | 221 | 15.83% | 9.39% | 14.33% |
| Delaware, scope unresolved — WER | 25 | 28.09% | 13.88% | 16.46% |
| Chou — CER | 261 | 13.36% | 15.25% | 18.79% |

For the221 checked English recordings, insertion counts fell from2,297 to338 while deletions increased from1,532 to1,664. For Mandarin, deletions increased from1,621 to2,339; substitutions and insertions changed much less. This identifies cases to inspect, but does not establish that every deletion was caused by cutting audio: recognition failures also produce deletions.

The window column additionally measures coverage lost to the fixed tail rule, including18 whole recordings shorter than15seconds. It is not an isolated ASR-quality estimate. Compared with complete candidate ASR, the window union covers1,544 fewer normalized words in the checked English group and1,779 fewer normalized characters in Mandarin.

Of85 original human-reference source files,79 were available locally and matched their archived hashes; six were macOS cloud placeholders. Those six were not falsely reported as freshly read: the comparison uses the unchanged, hash-verified archived reference manifest, with source availability explicitly recorded.

**Text agreement is not speaker purity.** Insertions can reflect interviewer speech or ASR errors; deletions can reflect removed participant speech, ASR failures, or window exclusions. Reference-assisted extraction also means these scores are not independent diarization validation. Do not silently remove unmatched words, replace ASR with reference text, or invent timestamps for reference words absent from ASR. Use discrepancies to guide listening and speaker-boundary review.
