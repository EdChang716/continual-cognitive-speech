# Validation against the user's final Chou and Delaware references

The user clarified that the Chou text originated in CHAT and that the CHAT markup/timing was removed during their earlier preparation. Missing CHAT files do not prevent full-recording CER evaluation. The requirements are a correct recording match, a fixed reference version, and matching content coverage. They do prevent directly transferring the lost speaker/time annotations.

**Experimental audio remains derived from TAUKADIAL.** The Chou/Delaware final source recordings and texts are validation references, not replacement model inputs. Current speech-source provenance records that all507 candidate recordings were reconstructed from normalized TAUKADIAL intervals; this audit does not alter those recordings, their transcripts, or the training release.

## Reproduce the Chou comparison

```bash
python -m pip install -e '.[audit]'
python -m ccspeech.data.source_cut_audit \
  --data-root data \
  --legacy-audit-root /path/to/source-project/data/overlap_audit \
  --raw-asr-dir /path/to/source-project/outputs/taukadial_asr/raw_whisper_v1 \
  --final-task-splits /path/to/dataset_v2_fromDrive/split_merge \
  --output data/reports/source_cut_reference_v1
```

The audit verifies source/reference identities, current final-window CSV text against the archived559 links, final full text across every window copy, split manifests, ASR settings, original/candidate audio hash bindings and time maps. Outputs include full-recording and window agreement CSVs, per-recording/window coverage JSONL and a summary with input hashes. All261 current candidate word streams remain bound to the portable source records;239 original ASR files were additionally available for fresh hash/word verification, while22 use the pinned portable records with availability explicitly recorded. No new ASR is run.

## Reference version

The user's `dataset_v2_fromDrive/split_merge/{train,validation,test}.csv` files contain full-task text and point to `data/{Chou,Delaware}/lpf_audio/` parents. The current MCIVoice matched-transcript CSVs use exactly those full-task texts, plus separately selected per-window `matched_transcript` text. Their30second/27second-stride windows live under `MCIVoice_multilingual/data/segmented30_audio_ovrlp10/`.

The matched release has559 Chou windows covering260 of261 TAUKADIAL-matched recordings. The omitted recording is12.742seconds long, below the old15second minimum; its full-task reference still exists and belongs in the primary full-recording audit. Across the559 window rows,558 have Whisper/LLM matching provenance and one is marked manually verified. Prior waveform checks place all559 windows at `segment_index * 27` seconds in the corresponding original recording. This windowing is not evidence that interviewer speech was removed.

Use the final **full-task reference** once per recording for the primary Chou comparison. Full texts differ after normalization from the previous source-reference audit in48 of261 recordings. The prior audit remains valid for its pinned reference version, but its numbers must not be labeled as final-release reference scores.

Secondary same-window comparisons use the final matched text, original-time windows, and timestamp-projected existing ASR. They are agreement with the earlier selected text, not a fresh ASR decoding of those windows. The old Whisper hypothesis also helped select that text, so old/new ASR differences are not an independent model comparison. Overlapping windows must not be concatenated into a recording-level transcript or summed as unique audio coverage.

Across those559 overlapping windows, current raw ASR has20.66% character disagreement and current candidate ASR22.19% against the same selected references. The old Whisper output has23.89%, but its model/context and reference-selection process differ. These figures do not establish that either ASR system is better. Source-time audio coverage is unioned independently: the old windows cover14,772.17seconds, of which the candidate retains13,672.08seconds. The1,100.09seconds omitted may contain interviewer speech, silence, or participant speech; they are not all labeled participant loss.

## Full-recording Chou cross-check — 2026-11-17

| Original TAUKADIAL split | Recordings | Raw-audio ASR CER | Candidate-audio ASR CER |
|---|---:|---:|---:|
| Train | 159 | 13.16% | 14.50% |
| Validation | 42 | 14.41% | 15.65% |
| Test, descriptive audit only | 60 | 14.07% | 17.59% |
| All | 261 | 13.59% | 15.45% |

The all-recording reference contains36,371 normalized characters. Raw ASR has1,974 substitutions,1,548 deletions and1,422 insertions; candidate ASR has1,983 substitutions,2,256 deletions and1,379 insertions. The candidate does not improve reference agreement, and deletions warrant listening/boundary review. These counts alone cannot separate recognition failures from participant audio removed during extraction.

Comparing the retained current-window word union to the same complete references yields18.99% CER. This additional figure includes omitted recording/tail coverage and must not be labeled an isolated ASR error rate. Use train/validation for choosing any revised extraction rule; test audit results do not choose that rule.

## Delaware scope must be established separately

The user's final Delaware parents are task-specific edited recordings. Matching a session/task filename does not establish that its complete text covers exactly one TAUKADIAL clip. A fresh waveform check of191 available candidate pairs found no whole-shorter constant-shift match at correlation≥0.98;183 had at least three strong local anchors with offsets varying by more than50ms. Local similarities establish correspondence while variable offsets indicate that one global shift is insufficient. The other candidate pairs and partial/ambiguous matches remain explicitly flagged in the private audit.

Transfer final source intervals to TAUKADIAL using validated piecewise correspondence before treating them as a participant extraction reference. The existing221 English clip-scoped CHAT references remain a separate comparison:15.83% raw WER versus9.39% candidate WER. These are not newly computed scores against all final Delaware task transcripts.

Only nine final full-task texts exactly match the normalized words of a previously scope-verified TAUKADIAL clip reference. On this selected subset (five train, four test;748 reference words), raw WER is16.58% and candidate WER4.95%. Two recordings are wholly omitted by the current minimum-tail window rule; the retained-window/full-reference comparison rises to17.91%. These nine are not representative evidence for all246 English recordings. The other cases are208 different full-task/verified-clip texts,25 unresolved clip scopes, three missing final references, and one unverified task. Text equality establishes the same scoring target, not independently reviewed speaker boundaries.

Private Delaware artifacts include `delaware_final_parent_mapping.csv`, `delaware_final_parent_anchors.jsonl`, `delaware_final_parent_summary.json`, `delaware_final_parent_audit.py`, `delaware_final_text_scope_match.csv` and `delaware_final_text_scope_summary.json`. They retain availability, waveform diagnostics and scope statuses without replacing training inputs.

## Interpretation

There is no acceptance threshold selected by this audit. Evaluate substitutions, deletions and insertions separately, inspect participant-content loss and interviewer contamination on aligned audio, and review large per-recording errors. Chinese CER and English WER use different units and should not be compared numerically across languages. A lower error rate does not by itself certify participant-only speech or downstream cognitive-classification quality.

All private per-recording/segment reports remain under `data/reports/`. Reference files are not added to the public repository or substituted into model inputs.
