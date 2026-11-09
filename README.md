# Continual Cognitive Speech

**Continual Adaptation for Multilingual Cognitive Speech Assessment**

A solo course research project comparing ER, DER++, and MbPA++-style memory under domain-incremental learning. Phase 1 uses Amazon ↔ Yelp five-star classification; Phase 2 uses English ↔ Mandarin TAUKADIAL normal-cognition versus mild-cognitive-impairment classification. Later experiments compare selective module updates, LoRA placement, and Nested Learning-inspired update frequencies.

**Current status: data preparation is complete; training and model implementations are the next milestone.** The text protocol is a controlled two-domain adaptation of the MbPA++ setup, not a reproduction of its original five-dataset scores.

## Prepared data

| Dataset | Train | Validation | Official-test subset / set | Unit |
| --- | ---: | ---: | ---: | --- |
| Amazon Review Full | 115,000 | 10,000 | 7,600 | Reviews; five rating classes |
| Yelp Review Full | 115,000 | 10,000 | 7,600 | Reviews; five rating classes |
| TAUKADIAL | 103 / 309 | 26 / 78 | 40 / 120 | Participants / recordings; NC vs MCI |

The complete Amazon source (3,000,000 train / 650,000 test) and Yelp source (650,000 / 50,000) are also retained. Text splits use fixed seed 42, balanced classes, and normalized duplicate filtering across both domains and all splits. TAUKADIAL keeps the official test participants unchanged; all three recordings from a participant stay together. Test diagnoses are stored separately from model-facing test manifests.

## 1. Install

Python 3.10+ is required. CPU is sufficient for data preparation; no GPU or model weights are downloaded at this stage.

```bash
git clone git@github.com:EdChang716/continual-cognitive-speech.git
cd continual-cognitive-speech
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Keep approximately 15 GB free for both bundles, raw files, and extracted audio; allow additional space for future training artifacts.

## 2. Download the prepared release

The project data are in [Google Drive — dataset_CL](https://drive.google.com/drive/folders/1F8PnLDto77jWNNTvXChJH-DefKO3Z6Gp). This folder is **restricted to its owner**. Dataset access is not granted by cloning this repository, and anonymous `gdown --folder` does not work for this private folder.

Download `text_data_v1.zip` and `taukadial_data_v1.zip` while signed into the authorized Google account, and put them under `artifacts/data_v1/`. For RunPod or another server, configure an authorized Google Drive `rclone` remote, then use:

```bash
# One-time authentication instructions: docs/data_access.md
bash scripts/download_from_drive.sh gdrive

# Verify the committed release hashes, restore files, and extract TAUKADIAL audio.
python scripts/restore_data.py
python -m ccspeech.data.validate
```

The text ZIP includes complete raw sources, six prepared Parquet files, fixed sample indices, and the preparation audit. The TAUKADIAL ZIP includes original archives, a hash-bound language map, participant splits, held-out labels in a separate directory, and audits. Original audio is extracted during restore; duplicate expanded WAVs are not stored inside the ZIP. `configs/data/release_v1.json` records bundle hashes and sizes.

To restore only the text phase, use `python scripts/restore_data.py --dataset text`, then `python -m ccspeech.data.validate --dataset text`.

## 3. Alternatively, rebuild from source

Public text files can be downloaded directly, without Drive authentication:

```bash
python -m ccspeech.data.download
python -m ccspeech.data.text --seed 42
```

Downloads pin exact upstream revisions and verify size and SHA-256. Existing source files are verified; incomplete transfers can resume. Text preparation verifies source row counts and excludes normalized duplicates before sampling. It refuses to overwrite an existing preparation unless `--overwrite` is explicit.

For TAUKADIAL, use your authorized original files and the private language map:

```bash
python -m ccspeech.data.taukadial \
  --source-dir /path/to/authorized/TAUKADIAL \
  --language-map /path/to/private/language_map.csv \
  --seed 42
```

The original release does not provide a language column. The supplied map comes from an existing source-cohort crosswalk and is verified against all 507 original WAV hashes; it is not an official language annotation. See [TAUKADIAL preparation](docs/taukadial.md).

## Repository layout

```text
configs/data/                Pinned upstream sources and release checksums
src/ccspeech/data/           Download, prepare, bundle, restore, validate
scripts/                    Small command-line orchestration helpers
tests/                      Split, corruption, archive, and reproducibility checks
docs/                       Data access and experimental protocol decisions
data/raw/                   Original source files and WAVs (gitignored)
data/processed/             Model-facing text and speech manifests (gitignored)
data/locked/                Held-out speech labels (gitignored)
data/reports/               Per-build audit and provenance (gitignored)
artifacts/data_v1/           Downloadable release bundles (gitignored)
```

Future model, memory-method, training, and evaluation code should live in separate `ccspeech` subpackages and consume these common data contracts. Raw data, subject-level metadata, credentials, checkpoints, and derived embeddings do not belong in Git.

## Verification and experiment rules

```bash
python -m pytest -q
python -m ccspeech.data.validate --full
```

- Reuse identical prepared records for all methods and both domain orders. Shuffle training records within each domain using the experiment seed.
- Replay and retrieval memory contain only training examples encountered so far. Validation and test labels never enter the memory bank.
- `data/locked/` is a workflow convention, not an access-control mechanism; training code must not read it.
- Preserve raw audio. Resampling, chunking, speaker selection, and feature extraction are later model-preprocessing decisions. Existing recordings may include interviewer speech.
- Language, cohort/site, and picture differences are confounded in TAUKADIAL; report cohort adaptation, without attributing effects solely to language.

Further details: [text protocol](docs/text_protocol.md), [speech protocol](docs/taukadial.md), [private Drive access and RunPod](docs/data_access.md). Dataset terms remain those of their providers; this repository does not grant redistribution rights.
