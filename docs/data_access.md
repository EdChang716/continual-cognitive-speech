# Private data access and RunPod setup

The prepared release is in the owner's [dataset_CL Drive folder](https://drive.google.com/drive/folders/1F8PnLDto77jWNNTvXChJH-DefKO3Z6Gp). Its general access is **Restricted**, with the owner as the only listed user. Do not make the folder public to simplify downloads.

## Browser download

Sign into the authorized Google account. For the current speech project, download `speech_data_v3.zip`, `SHA256SUMS_v3`, and `README_DATA_v3.md`. Place the ZIP in `artifacts/data_v3/`, then run:

```bash
python scripts/restore_data.py --release 3 --dataset speech_v3
python -m ccspeech preflight --config configs/experiments/speech_derpp.json
```

To use the review-text experiments too, download `text_data_v1.zip` into the same directory and run `python scripts/restore_data.py` (default: release3, training inputs). The original `taukadial_data_v1.zip` remains unchanged and is optional for raw-audio audits. Download it too and use `--dataset all` to restore every package. Select one package with `--dataset text`, `--dataset speech_v3`, or `--dataset taukadial`.

Restore compares whole-archive SHA256 against the manifest committed in Git, checks every member, and rejects conflicting existing data. The speech package contains complete candidate audio, matched ASR word times, source interval maps, original recording-level split CSVs and separately locked test labels. Training/window regeneration does not require the larger raw-v1 archive or the previous workstation. V3 retains uncertain English sources separately from default model inputs; see [the v3 protocol](speech_preprocessing_v3.md).

To restore an older release explicitly, use `--release 2 --dataset speech_v2 --bundle-dir artifacts/data_v2` or `--release 1 --dataset all --bundle-dir artifacts/data_v1`. V3 speech configs point to `processed/speech_v3`; reproducing v2 additionally requires setting `data.speech_dir` to `processed/speech_v2`.

## RunPod or another headless Linux server

Install Python3.10+, Git, curl, and [rclone](https://rclone.org/install/) from their official distributions/package sources. Use persistent storage for code, data, model caches, and runs. Clone and install the project using the README. The code does not create a GPU instance or authenticate a future instance automatically.

Configure an authorized Google Drive remote named `gdrive`:

```bash
rclone config
```

Choose Google Drive and the read-only scope (`drive.readonly`) for downloads. Follow the [official Drive configuration guide](https://rclone.org/drive/) for your own OAuth client setup; the installed rclone warns that its old shared client is being retired. On a headless server, answer **No** to opening the browser, run the exact authorization command printed by rclone on your own browser-equipped machine, and paste its response back into the server terminal. See [remote setup](https://rclone.org/remote_setup/). Keep tokens and the rclone configuration out of Git and chat messages. An already authorized remote can be reused by substituting its name for `gdrive`.

```bash
bash scripts/download_from_drive.sh gdrive 3 speech_v3
python scripts/restore_data.py --release 3 --dataset speech_v3
python -m ccspeech preflight --config configs/experiments/speech_derpp.json

# Optional: both text and speech training inputs.
bash scripts/download_from_drive.sh gdrive 3 training
python scripts/restore_data.py
```

The download script selects filenames from the committed release manifest and copies only those bundles and that version's documentation. Its arguments are remote name, release number, and dataset selection; default selection is `training`. Use `all` for the optional raw archive, or `2 speech_v2` for the previous speech release. It does not change cloud files, permissions, or other folders. Anonymous `gdown --folder` cannot download the owner-restricted folder.

For Phase1 alone, Drive is optional:

```bash
python -m ccspeech.data.download
python -m ccspeech.data.text --seed 42
```

These commands rebuild text from pinned public sources. Clinical source acquisition remains an authorized-data step; this repository does not obtain access for new users.

## Access and provenance

TAUKADIAL access is managed through [DementiaBank/TalkBank](https://dementia.talkbank.org/TAUKADIAL/), subject to [TalkBank's rules](https://talkbank.org/0share/rules.html). This private archive is not a public mirror. Amazon/Yelp retain their upstream terms as well.

Version1 preserves original archives and deterministic sample/participant splits. Version2 adds candidate audio and timestamp-matched ASR. Version3 adds conservative supported interval recovery, fresh ASR for changed audio, explicit quarantine, complete short-tail coverage and versioned quality reports. Portable releases exclude workstation paths, credentials, prior diagnostic predictions and human-reference text. Prepared ASR text and speaker labels are not manually certified. See [v3 preprocessing](speech_preprocessing_v3.md) and the [historical v2 protocol](speech_preprocessing.md).
