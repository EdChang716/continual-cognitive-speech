# Continual Cognitive Speech

**Continual Adaptation for Multilingual Cognitive Speech Assessment** — Edward Chang's solo course project.

Compare how bounded replay, historical predictions, query-local adaptation, and selective parameter updates balance learning a new domain with retaining an earlier domain. Phase 1 uses **Amazon ↔ Yelp five-star classification** with BERT. Phase 2 uses **English ↔ Mandarin normal cognition / mild cognitive impairment (MCI)** classification with a Whisper encoder. Both are domain-incremental tasks with unchanged label sets.

**Implemented:** data download/restore, training-ready speech windows and matched ASR text, ER/DER++/MbPA++-style experiments, baselines, LoRA/module/frequency ablations, resumable training, validation selection, and separate final-test evaluation. 122 tests pass, including actual tiny BERT and Whisper models; a real pretrained BERT CPU pilot also passed. Full pretrained GPU experiments and research results have not yet been run. See [verification](docs/verification.md). This is a controlled adaptation of the papers, not reproduction of their published benchmark scores or a full Nested Learning/HOPE implementation.

## Prepared data

| Dataset | Train | Validation | Test | Prediction unit |
| --- | ---: | ---: | ---: | --- |
| Amazon Review Full | 115,000 | 10,000 | 7,600 | Review; five rating classes |
| Yelp Review Full | 115,000 | 10,000 | 7,600 | Review; five rating classes |
| TAUKADIAL speech v2 | 593 / 103 | 182 / 26 | 271 / 40 | Windows / participants; NC vs MCI |

All 507 complete TAUKADIAL participant-candidate recordings are retained as portable sources. Derived inputs are **16 kHz mono PCM16, 30-second windows, 27-second stride (10% overlap), tails ≥15 seconds retained**. All 169 participants remain represented; split by participant precedes segmentation. Match existing Whisper-large-v3 ASR words to windows using timestamps. This is local ASR matching, not GPT matching to human transcripts. Uncertain speaker boundaries, omitted short tails, and reference-assisted source selection are recorded in quality flags and review lists. See [speech preprocessing](docs/speech_preprocessing.md).

Text preparation preserves full raw sources, uses deterministic balanced splits, and filters normalized duplicates across domains and splits. Official speech-test diagnoses stay under `data/locked/`; training never opens them. Matching text is included for inspection and future work; the current speech classifier consumes **audio only**, using the Whisper encoder and a linear head.

## 1. Install on RunPod or your machine

Use Python 3.10+ and persistent storage. Training dependencies were tested with Python 3.11, PyTorch 2.8.0, and Transformers 4.51.3. Full runs default to CUDA; CPU is supported for smoke tests.

```bash
git clone git@github.com:EdChang716/continual-cognitive-speech.git
cd continual-cognitive-speech
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,dev]'
export TOKENIZERS_PARALLELISM=false
# Use a persistent directory for downloaded pretrained weights.
export HF_HOME="$PWD/artifacts/huggingface"
python -m pytest -q
```

Data-only installation remains available as `python -m pip install -e '.[dev]'`; speech regeneration additionally needs NumPy and SoundFile. Provisioning or starting a paid GPU is not performed by these commands. Check CUDA availability with `python -c 'import torch; print(torch.cuda.is_available())'`.

## 2. Download and restore data

Use the owner-restricted [dataset_CL Drive folder](https://drive.google.com/drive/folders/1F8PnLDto77jWNNTvXChJH-DefKO3Z6Gp). Download **`text_data_v1.zip` and `speech_data_v2.zip`** into `artifacts/data_v2/`. The optional `taukadial_data_v1.zip` contains original raw speech for additional audits. Checksums are committed in [release_v2.json](configs/data/release_v2.json).

On a headless server, configure an authorized Google Drive `rclone` remote using [these instructions](docs/data_access.md), then:

```bash
bash scripts/download_from_drive.sh gdrive
python scripts/restore_data.py                 # text + training-ready speech
# Optional: also extract the original raw TAUKADIAL archives.
python scripts/restore_data.py --dataset all
```

Restore verifies archive and per-file hashes and refuses conflicting local data. It includes the source audio and sanitized word/time maps needed to regenerate speech inputs without the old workstation project, ASR models, or an API:

```bash
python -m ccspeech.data.speech prepare --data-root data
python -m ccspeech preflight --config configs/experiments/text_derpp.json
python -m ccspeech preflight --config configs/experiments/speech_derpp.json
```

Preflight verifies split separation and speech WAV hashes without loading models or test labels. Keep around **20 GB for the data, bundles, and package dependencies**, plus model caches and checkpoints. Full suites accumulate substantially more checkpoint storage; see [experiment execution and costs](docs/experiments.md).

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

[Experiment protocol](docs/experiments.md) · [Methods](docs/methods.md) · [Models](docs/models.md) · [Speech preprocessing](docs/speech_preprocessing.md) · [Text protocol](docs/text_protocol.md) · [Drive/RunPod access](docs/data_access.md)

Language, recruitment cohort/site, and picture task differ together in TAUKADIAL. Report cohort adaptation without attributing effects solely to language. Historical data remain available for the joint-training reference; the online memory limit is an experimental restriction. Keep clinical data and participant-level outputs private. Dataset access and redistribution rights remain governed by their providers.
