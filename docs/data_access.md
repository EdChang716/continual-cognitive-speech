# Private data access and RunPod setup

The prepared release is in the owner's [dataset_CL Drive folder](https://drive.google.com/drive/folders/1F8PnLDto77jWNNTvXChJH-DefKO3Z6Gp). Its general access is **Restricted**, with the owner as the only listed user. Do not make the folder public to simplify downloads.

## Browser download

Sign into the authorized Google account. Download `text_data_v1.zip`, `speech_data_v2.zip`, `SHA256SUMS_v2`, and `README_DATA_v2.md`. Place the ZIPs in `artifacts/data_v2/`, then run:

```bash
python scripts/restore_data.py
```

The default restores text and training-ready speech. The original `taukadial_data_v1.zip` remains unchanged and is optional for raw-audio audits. Download it too and use `--dataset all` to restore every package. Select a single package with `--dataset text`, `--dataset speech_v2`, or `--dataset taukadial`.

Restore compares whole-archive SHA256 against the manifest committed in Git, checks every member, and rejects conflicting existing data. The speech-v2 package contains its original recording-level split CSV dependencies and locked test labels, so training/regeneration does not require downloading the much larger raw-v1 archive. To restore an older release explicitly, use `--release 1 --dataset all --bundle-dir artifacts/data_v1`.

## RunPod or another headless Linux server

Install Python3.10+, Git, curl, and [rclone](https://rclone.org/install/) from their official distributions/package sources. Use persistent storage for code, data, model caches, and runs. Clone and install the project using the README. The code does not create a GPU instance or authenticate a future instance automatically.

Configure an authorized Google Drive remote named `gdrive`:

```bash
rclone config
```

Choose Google Drive and the read-only scope (`drive.readonly`) for downloads. Follow the [official Drive configuration guide](https://rclone.org/drive/) for OAuth client setup. On a headless server, answer **No** to opening the browser, run the exact authorization command printed by rclone on your own browser-equipped machine, and paste its response back into the server terminal. See [remote setup](https://rclone.org/remote_setup/). Keep tokens and the rclone configuration out of Git and chat messages.

```bash
bash scripts/download_from_drive.sh gdrive
python scripts/restore_data.py
python -m ccspeech preflight --config configs/experiments/text_derpp.json
python -m ccspeech preflight --config configs/experiments/speech_derpp.json
```

The download script copies only the named release bundles and v2 documentation from this folder. It does not change cloud files, permissions, or other folders. Anonymous `gdown --folder` cannot download the owner-restricted folder.

For Phase1 alone, Drive is optional:

```bash
python -m ccspeech.data.download
python -m ccspeech.data.text --seed 42
```

These commands rebuild text from pinned public sources. Clinical source acquisition remains an authorized-data step; this repository does not obtain access for new users.

## Access and provenance

TAUKADIAL access is managed through [DementiaBank/TalkBank](https://dementia.talkbank.org/TAUKADIAL/), subject to [TalkBank's rules](https://talkbank.org/0share/rules.html). This private archive is not a public mirror. Amazon/Yelp retain their upstream terms as well.

Version1 preserves original archives and deterministic sample/participant splits. Version2 adds approved candidate audio, matched ASR text and word times, quality flags, derived windows, and sanitized provenance. It excludes workstation paths, credentials, prior model predictions, and human-reference transcripts. Prepared ASR text is not human-verified. See [speech preprocessing](speech_preprocessing.md) for the exact source and matching distinctions.
