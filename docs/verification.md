# Verification record

## Speech v3 release — 2026-11-17

- **203 automated tests passed** in the Python3.12 training environment, including the new conservative interval recovery, source/provenance corruption rejection, ASR cache/resume/error handling, v3 quarantine and complete-tail export, quality comparison and selective download tests. Existing tiny BERT/Whisper, continual-learning, resume and test-access tests remain included.
- **All507 original TAUKADIAL recordings were freshly normalized and hash-checked.** The repaired candidate samples were checked against those exact original intervals.149 Mandarin recordings changed, adding110.622seconds;358 retained identical audio and ASR.
- **Actual local Whisper-large-v3 re-transcription completed for all149 changed recordings.** Model revision and physical weight/config hashes were pinned. The runner received audio and language only, without reference transcripts or diagnoses. No full-recording ASR output was empty. No paid GPU or external transcription API was used.
- **V3 export verified1,191 WAV windows from482 recordings/167 participants.** All nonempty tails are retained, including201 windows under15seconds.25 unresolved English recordings are quarantined. One6.132second Mandarin validation tail has empty matched text and an explicit flag. The full169-participant benchmark is not represented by the default eligible subset.
- **Quality audit verified all507 source identities and scored482 eligible recordings.** Fixed-reference validation Mandarin CER changed from15.65% to14.36%; English WER stayed9.11%. For every eligible recording, the union of model-window source word indices equals the complete candidate transcript. The private review queue excludes test records; descriptive test quality figures did not select preprocessing rules. These checks do not certify speaker purity or classification gains.
- **Real v3 training preflight passed:** train English321/Mandarin371 windows; validation English80/Mandarin116; no test labels read. All78 generated core-speech plan configurations explicitly select `processed/speech_v3`.
- **Portable metadata scan passed:**1,691 JSON/JSONL/CSV metadata files checked for workstation paths and obvious credential markers. Human-reference text remains outside the release; ASR output is included as intended.

- **Standalone archive round-trip passed:** the1,392,281,633-byte ZIP and all3,389 member hashes verified during restoration into a clean temporary directory. After deleting the restored derived windows and their audit files, preparation rebuilt all1,198 derived files byte-for-byte from only the portable source package. Restored-data preflight also passed without opening test labels.
- **Private Drive upload verified:** `speech_data_v3.zip`, `README_DATA_v3.md` and `SHA256SUMS_v3` were uploaded to the existing restricted folder. Rclone's size/hash check reported3 matching files and0 differences. Older releases were retained and folder sharing was unchanged. The committed v3 manifest pins the ZIP SHA256; cloud comparison used Drive's stored checksums, without downloading a second full copy.

Research classifier training remains a separate next step.

## Initial experiment-code release — 2026-11-15

The following checks were completed locally before publishing the experiment code. They establish data integrity and engineering behavior; they are not full research experiments.

- **122 automated tests passed.** Includes real tiny Hugging Face BERT/Whisper forwards and gradients, pretrained weight-loading from local test checkpoints, frozen modules/LoRA, ER/DER++/sparse/MbPA global behavior, nearest/random query adaptation, state restoration, exact interrupted resume with pending gradients, source-cache reuse, relocated-data resume, official-test access gates, participant aggregation, corruption rejection, and suite/summary behavior.
- **Actual pinned pretrained BERT CPU pilot passed.** DER++, Amazon→Yelp, source2batches and target2batches, up to100selected training reviews and20validation reviews per domain. Completed in about53seconds locally. It is explicitly smoke=true and has no final-test research report. GPU performance cannot be inferred from this timing.
- **Real speech collation passed.** The shortest retained and a complete30second English training window yielded finite `[2,80,3000]` Whisper features with valid-frame masks matching their lengths. Tiny Whisper tests exercise the entire training/evaluation pipeline; full pretrained Whisper-medium GPU training has not yet run.
- **Complete speech export verified.** 507 candidate source WAV hashes matched ASR inputs. Each candidate's exact PCM was reconstructed from its normalized original source intervals. All1,046derived windows were read back and verified; no cross-split byte duplicates, no empty matched ASR text, all169participants retained.
- **Standalone speech ZIP verified and restored into a clean temporary directory.** All2,076members and the whole ZIP hash matched. Re-running speech preparation from only that restored package reproduced the prepared artifacts byte-for-byte, without old workstation paths or an API.
- **Portable metadata scan passed.** 519 JSON/JSONL/CSV files checked for workstation paths and obvious credential markers. Source exports use an explicit allowlist and exclude human reference text and old prediction outputs.
- **Private upload verified in Drive.** Owner-only restricted folder; UI reported successful upload of speech_data_v2.zip, SHA256SUMS_v2, and README_DATA_v2.md. Existing v1 data files were retained. Committed release hashes are the download/restore integrity authority.

Validation and pilot outputs are private, ignored artifacts. No final accuracy, clinical effectiveness, improvement over joint training, successful full-GPU reproduction, or publication result is claimed. CUDA resource paths require checking on the intended RunPod hardware.

## Transcript-audit addition — 2026-11-17

138 tests passed in the current Python3.12 environment, including16 new audit/report tests. All507 real recordings were compared with matched Chou/Delaware references; all current window transcripts matched their source words. The self-contained private HTML contains507 recording panels, one fixed script, and no external resource links. See [comparison protocol and aggregate results](transcript_comparison.md). Audio, ASR text, participant splits and model settings were unchanged.

## Original-audio audit addition — 2026-11-17

144 tests passed, including six new tests for original-file integrity, constant shifts, edited audio, silence/partial-coverage rejection, cloud placeholders, and independent CHAT clocks. The physical audit freshly verified421 source/TAUKADIAL pairs:216 Chou zero-offset pairs,202 Delaware raw-session crops, and three edited-task matches using local anchors only. The other86 pairs had at least one cloud placeholder and retain explicitly labeled prior evidence. Fresh constant-shift matches all had five consistent anchors. Original audio, training release, ASR text and participant splits were unchanged. See [audio identity and clock audit](audio_identity.md).

## Final source-cut reference addition — 2026-11-17

152 tests passed, including eight new tests for piecewise word clocks, overlap coverage, short recordings without old windows, reference consistency and provenance failures. The real audit covered all261 Chou full-task references and559 prior windows. An independent recomputation from the prior ASR comparison artifact produced identical full-reference S/D/I counts. The final Chou reference differs from the earlier reference version for48 recordings. A separate private waveform audit checked191 final Delaware task-parent candidates; nine final task texts match an already scope-verified English clip reference exactly. These final-task edited sources are distinct from the raw-session sources in the previous audio-identity audit. See [reference version and results](source_cut_validation.md). Training data and ASR outputs were unchanged; no new ASR/GPU/API job was launched.
