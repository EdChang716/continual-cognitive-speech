# Continual Cognitive Speech

**Continual Adaptation for Multilingual Cognitive Speech Assessment** — Edward Chang's solo course project.

Compare how bounded replay, historical predictions, query-local adaptation, and selective parameter updates balance learning a new domain with retaining an earlier domain. Phase 1 uses **Amazon ↔ Yelp five-star classification** with BERT. Phase 2 uses **English ↔ Mandarin normal cognition / mild cognitive impairment (MCI)** classification with a Whisper encoder. Both are domain-incremental tasks with unchanged label sets.

**Implemented:** data download/restore, training-ready speech windows and matched ASR text, ER/DER++/MbPA++-style experiments, baselines, LoRA/module/frequency ablations, resumable training, validation selection, and separate final-test evaluation. 203 tests pass, including actual tiny BERT and Whisper models; a real pretrained BERT CPU pilot also passed. Speech v3 includes completed local Whisper re-transcription and quality comparisons. Full pretrained classifier GPU experiments and research results have not yet been run. See [verification](docs/verification.md). This is a controlled adaptation of the papers, not reproduction of their published benchmark scores or a full Nested Learning/HOPE implementation.

**Start with the speech experiments**, following the instructor's feedback: finish English-only, Mandarin-only, both sequential orders, joint training, and the module/memory ablations before moving to review text. The text implementation and data remain available for the second stage.

## Prepared data

| Dataset | Train | Validation | Test | Prediction unit |
| --- | ---: | ---: | ---: | --- |
| Amazon Review Full | 115,000 | 10,000 | 7,600 | Review; five rating classes |
| Yelp Review Full | 115,000 | 10,000 | 7,600 | Review; five rating classes |
| TAUKADIAL speech v3 | 692 / 102 | 196 / 26 | 303 / 39 | Windows / participants; NC vs MCI |

All **507 complete TAUKADIAL candidate recordings** remain in the private portable source package. The default model inputs contain **482 recordings, 167 participants and 1,191 windows**; 25 unresolved English recordings are quarantined for review. This leaves one training and one test participant without eligible audio, so this subset must not be presented as the full 169-participant benchmark. Split by participant precedes segmentation.

Inputs are **16 kHz mono PCM16, 30-second windows, 27-second stride (10% overlap)**. All nonempty short recordings and tails are retained at their actual length; the model pads them during collation. Participant extraction uses source-reference and acoustic evidence, with conservative recovery of clipped Mandarin speech. All model audio comes from TAUKADIAL. The 149 changed recordings were re-transcribed locally with pinned Whisper-large-v3; the other 358 reuse ASR only with byte-identical audio. Window text is matched to ASR word timestamps on the same cleaned-audio clock. It is not a human-corrected transcript or GPT matching. One 6.132-second validation tail has empty ASR text and remains explicitly flagged. See the [v3 processing and usage guide](docs/speech_preprocessing_v3.md).

A [source-transcript audit](docs/transcript_comparison.md) compares all507 recordings with their matched Chou/Delaware references, separating raw ASR, candidate ASR, and retained-window coverage. The private report flags discrepancies; it does not certify speaker purity.

An [original-audio audit](docs/audio_identity.md) separately checks source waveform identity and time offsets. Chou recordings align at zero offset; most Delaware recordings are crops of longer sessions, with six edited-task exceptions in the prior crosswalk. CHAT timestamps must use their verified media clock, which can differ from the raw-WAV clock. Local Chou references are untimed CSV text.

The [final-source-reference audit](docs/source_cut_validation.md) established the v2 baseline against the user's final Chou texts and old windows. The v3 comparison holds references fixed: Chinese validation CER changes from **15.65% to 14.36%**; English validation WER remains **9.11%** because its eligible audio is unchanged. English comparisons use verified clip-scope references, since final Delaware task audio has additional edits. Full v3 window coverage preserves every source ASR word index after overlap deduplication. These scores measure content agreement, not independently verified speaker purity; reference-assisted extraction and unresolved boundaries remain documented limitations.

Text preparation preserves full raw sources, uses deterministic balanced splits, and filters normalized duplicates across domains and splits. Official speech-test diagnoses stay under `data/locked/`; training never opens them. Matching text is included for inspection and future work; the current speech classifier consumes **audio only**, using the Whisper encoder and a linear head.

## 1. Install on RunPod or your machine

Use Python 3.10+ and persistent storage. Training dependencies were tested with Python 3.12, PyTorch 2.8.0, and Transformers 4.51.3. Full runs default to CUDA; CPU is supported for smoke tests.

```bash
git clone git@github.com:EdChang716/continual-cognitive-speech.git
cd continual-cognitive-speech
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,audit,dev]'
export TOKENIZERS_PARALLELISM=false
# Use a persistent directory for downloaded pretrained weights.
export HF_HOME="$PWD/artifacts/huggingface"
python -m pytest -q
```

Data-only installation remains available as `python -m pip install -e '.[dev]'`; speech regeneration additionally needs NumPy and SoundFile. Provisioning or starting a paid GPU is not performed by these commands. Check CUDA availability with `python -c 'import torch; print(torch.cuda.is_available())'`.

## 2. Download and restore data

Use the owner-restricted [dataset_CL Drive folder](https://drive.google.com/drive/folders/1F8PnLDto77jWNNTvXChJH-DefKO3Z6Gp). For speech, download **`speech_data_v3.zip`**, `README_DATA_v3.md` and `SHA256SUMS_v3` into `artifacts/data_v3/`. The optional `text_data_v1.zip` contains review text; `taukadial_data_v1.zip` contains original raw speech for additional audits. Checksums are committed in [release_v3.json](configs/data/release_v3.json).

On a headless server, configure an authorized Google Drive `rclone` remote using [these instructions](docs/data_access.md), then:

```bash
bash scripts/download_from_drive.sh gdrive 3 speech_v3
python scripts/restore_data.py --release 3 --dataset speech_v3
# Optional: download and restore text + speech training inputs.
bash scripts/download_from_drive.sh gdrive 3 training
python scripts/restore_data.py
# Optional: download and restore all inputs, including raw TAUKADIAL.
bash scripts/download_from_drive.sh gdrive 3 all
python scripts/restore_data.py --dataset all
```

Restore verifies archive and per-file hashes and refuses conflicting local data. It includes the source audio and sanitized word/time maps needed to regenerate speech inputs without the old workstation project, ASR models, or an API:

```bash
python -m ccspeech.data.speech_v3 prepare --data-root data
python -m ccspeech preflight --config configs/experiments/speech_derpp.json
# After restoring the optional text package:
python -m ccspeech preflight --config configs/experiments/text_derpp.json
```

Preflight verifies split separation and speech WAV hashes without loading models or test labels. Keep around **20 GB for the data, bundles, and package dependencies**, plus model caches and checkpoints. Full suites accumulate substantially more checkpoint storage; see [experiment execution and costs](docs/experiments.md).

The model loader reads `data/processed/speech_v3/{train,validation,test}.csv`; audio paths are relative to `data/`. Detailed word times and original-audio time maps are in `segments.jsonl`; review-only sources are in `quarantined_recordings.jsonl`. Quality comparisons are in `data/reports/speech_v3/quality/`. Do not add quarantined sources to model CSVs or combine v2/v3 windows. All current speech configs explicitly select v3; the [v3 guide](docs/speech_preprocessing_v3.md) also documents how to reproduce v2.

## 3. Run a pilot, then a full configuration

Pilots use the actual configured pretrained backbone but deliberately limit training/evaluation. They are marked `smoke=true` and cannot produce a final-test report through the CLI.

```bash
python -m ccspeech run --config configs/experiments/text_pilot.json
python -m ccspeech run --config configs/experiments/speech_pilot.json

# Full data, DER++ example; reverse order/method through a suite below.
python -m ccspeech run --config configs/experiments/text_derpp.json
python -m ccspeech run --config configs/experiments/speech_derpp.json
```

Use `--device cpu` for a CPU pilot. Use `--data-root /workspace/data` if needed. Source checkpoints are trained once per compatible seed/order/memory/data configuration, then reused across methods and update policies. Adaptation begins with an already-trained classification head. Resume an interrupted run with the **same command plus `--resume`**. Optimizer moments, pending multi-timescale gradients, memory, data cursor, and RNG state are restored. Completed matching runs are skipped.

## 4. Plan the comparisons explicitly

| Family | What it compares |
| --- | --- |
| `core` | Sequential fine-tuning, ER, DER++, sparse replay only, MbPA++ nearest retrieval, random-neighbor local adaptation, joint training, separate-domain models |
| `modules` | Head only, encoder LoRA only, LoRA + head; sequential/DER++/MbPA++ |
| `placement` | Same-rank LoRA in equally many early versus late layers; DER++/MbPA++ |
| `timescales` | Head/encoder update intervals 1/1, 1/4, 4/4; DER++/MbPA++ with LoRA + head |

Both phases have both domain orders, seeds 42/43/44, and two core memory budgets (text: 256/1,024 reviews; speech: 32/128 windows). `configs/suites/` also provides an `all` family with duplicate settings removed. Planning writes reviewable per-run JSON files and starts no jobs:

```bash
python -m ccspeech plan --suite configs/suites/text_core.json --output runs/plans/text_core
python -m ccspeech plan --suite configs/suites/speech_core.json --output runs/plans/speech_core
# Inspect plan.json and individual configs before choosing jobs.
python -m ccspeech run-plan --plan runs/plans/text_core/plan.json --indices 0 --resume
# Explicitly run the whole chosen plan sequentially:
python -m ccspeech run-plan --plan runs/plans/speech_core/plan.json --all --resume
```

Replace `core` with `modules`, `placement`, or `timescales` for those families. All recipes use common splits, backbone revisions, and current-data epochs within each phase. Replay and local adaptation introduce additional computation, which is measured rather than called “equal compute.” MbPA++ validation is expensive: the full text validation set with 32 neighbors × 30 local steps entails **19.2 million neighbor presentations per pass**. Pilot and inspect measured costs before launching a whole suite.

## 5. Freeze settings, then evaluate test data

Each run prints its output directory. Training selects source checkpoints using source-domain validation accuracy only, and adaptation checkpoints using equal-domain mean validation accuracy. After selecting configurations on validation, evaluate each chosen run explicitly:

```bash
python -m ccspeech final-test --run-dir runs/PHASE/ORDER/RUN_NAME
python -m ccspeech summarize --runs-root runs --split validation --output runs/validation.csv
python -m ccspeech summarize --runs-root runs --split test --output runs/test.csv
```

Final-test reports cannot overwrite an existing report. Accuracy, macro-F1, balanced accuracy, and AUROC are available for both modalities. Speech predictions average windows within recordings, then average recordings equally within participants. Speech intervals resample participants, including paired intervals for source-score change. Undefined AUROC is reported as null. Summaries keep individual runs and repeated-seed means/SDs separate.

## Repository organization

```text
configs/data/               Pinned source/release manifests
configs/experiments/        Full and explicitly truncated pilot configs
configs/suites/             Inspectable experiment-family definitions
src/ccspeech/data/          Download, prepare, verify, bundle, restore
src/ccspeech/models/        BERT/Whisper backbones, native LoRA, update policies
src/ccspeech/methods/       Reservoir, replay objectives, local adaptation, optimizer
src/ccspeech/training/      Data adapters, config, loop, orchestration, checkpoints
src/ccspeech/evaluation/    Metrics, participant aggregation, result summaries
scripts/                   Thin download/restore entry points
tests/                     Synthetic offline integrity and end-to-end tests
docs/                      Protocol, method differences, usage and costs
data/                      Private sources, windows, transcripts, labels (ignored)
artifacts/                 Downloaded release/model caches (ignored)
runs/                      Private checkpoints, predictions, reports (ignored)
```

[Experiment protocol](docs/experiments.md) · [Methods](docs/methods.md) · [Models](docs/models.md) · [Speech v3](docs/speech_preprocessing_v3.md) · [Local ASR](docs/local_asr.md) · [Text protocol](docs/text_protocol.md) · [Drive/RunPod access](docs/data_access.md)

Language, recruitment cohort/site, and picture task differ together in TAUKADIAL. Report cohort adaptation without attributing effects solely to language. Historical data remain available for the joint-training reference; the online memory limit is an experimental restriction. Keep clinical data and participant-level outputs private. Dataset access and redistribution rights remain governed by their providers.
