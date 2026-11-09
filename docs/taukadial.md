# TAUKADIAL data preparation

TAUKADIAL-24 is a restricted-access picture-description speech dataset for normal cognition (NC) versus mild cognitive impairment (MCI) classification. The supplied official training set contains **129 participants and 387 recordings**; the official test set contains **40 different participants and 120 recordings**. Each participant has three recordings. The complete 169-participant corpus is not a 169-participant training set.

Acquire the original files through your authorized DementiaBank access: [official challenge page](https://luzs.gitlab.io/taukadial/) and [member download location](https://media.talkbank.org/dementia/English/0extra/TAUKADIAL/). Keep the dataset under the access conditions in the [TalkBank ground rules](https://talkbank.org/0share/rules.html); the repository does not grant redistribution rights.

## Required inputs

Place these files in an authorized local source directory:

- `TAUKADIAL-24-train.tgz`
- `TAUKADIAL-24-test.tgz`
- `meta_test.csv`
- `language_map.csv`, described below
- Optional `testgroundtruth.csv`, for eventual held-out evaluation only

The training archive contains comma-delimited `groundtruth.csv`. The separately supplied test metadata and test ground truth use semicolon delimiters. The importer detects these delimiters, validates filenames, checks diagnoses, and keeps test labels separate.

## Language provenance

The original metadata do **not** include language labels. Consequently the importer requires an explicit language map containing `recording_id`, `language` (`en` or `zh`), and `original_wav_sha256`. It verifies every map entry against the exact original WAV bytes. It never infers language from the numeric participant identifier or WAV sample rate.

For this local preparation, the map was reduced from the user's existing audio-verified source-cohort crosswalk: Chou corresponds to Mandarin (`zh`) and Delaware corresponds to English (`en`). The prior project retains the detailed source-recording evidence. Only recording IDs, language, and original audio checksums enter this project; no previous transcripts, predictions, features, or diagnosis outputs are imported. These labels are provenance-backed cohort assignments, **not official language metadata**. The resulting `data/raw/taukadial/language_map.csv` is a portable private input for future runs. A JSONL source containing the same required fields is also accepted; other fields are ignored.

## Run

From the repository root after installing the package:

```bash
python -m ccspeech.data.taukadial \
  --source-dir /path/to/authorized/TAUKADIAL \
  --language-map /path/to/private/language_map.csv \
  --data-root data \
  --seed 42
```

If `language_map.csv` is already in the source directory, omit `--language-map`. Imported archives are SHA-256 verified; an existing different file causes an error instead of being silently replaced. Extraction rejects path traversal, symbolic/hard links, duplicate entries, and unexpected files.

If a previous run installed locked test labels, a later run without the original `testgroundtruth.csv` fails before changing data. Supply that original file again or choose a fresh data root; the importer does not silently keep stale labels while reporting them as absent and does not delete existing files.

## Outputs and split policy

```text
data/
  raw/taukadial/
    archives/TAUKADIAL-24-{train,test}.tgz
    TAUKADIAL-24/train/{groundtruth.csv,*.wav}
    TAUKADIAL-24/test/*.wav
    meta_test.csv
    language_map.csv
  processed/taukadial/{train,validation,test}.csv
  locked/taukadial/{testgroundtruth,test_labels}.csv  # if supplied
  reports/taukadial/{audit,file_hashes}.json
```

Training and validation use only official training participants. Within each language × diagnosis stratum, sort participants by SHA-256 of `seed|participant_id`, then reserve the rounded 20% for validation (at least one participant on either side). Every participant's three recordings stay together. The official test partition is unchanged, and the default test manifest contains **no diagnosis labels**. Test labels are stored under `data/locked/` and never used to select validation participants. “Locked” is a workflow convention, not filesystem access control; training code must not read that directory.

CSV audio paths are relative to `data/`, making the manifests portable. The label mapping is `NC=0`, `MCI=1`. Manifest fields include recording/participant IDs, language, split, duration, sample rate, channel count, frame count, and audio SHA-256. All generated records stay under the ignored `data/` directory.

The audit validates all WAV frames, one language/diagnosis per participant, exactly three recordings per participant, participant disjointness, language-map audio hashes, and absence of identical audio bytes across partitions. It reports split sizes and training/validation class counts. It does not report test diagnosis distributions for tuning.

## Audio handling

Original WAV bytes remain unchanged. There is no resampling, channel averaging, VAD, silence removal, speaker filtering, ASR, or precomputed embedding extraction in this preparation step. Raw recordings may include interviewer speech. Model-time preprocessing and any participant-only segmentation must be specified and compared separately; this preparation does not certify participant-only speech. A future LoRA/encoder-update experiment needs raw audio rather than fixed embeddings.

The English and Mandarin cohorts also differ in source site and elicitation pictures. Treat the two sequential orders as cohort/domain adaptation experiments; do not interpret language alone as the cause of any performance difference.
