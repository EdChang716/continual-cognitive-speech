# Private data access and RunPod setup

The prepared release is stored in the owner's [dataset_CL Drive folder](https://drive.google.com/drive/folders/1F8PnLDto77jWNNTvXChJH-DefKO3Z6Gp). On November 9, 2026, the owner requested changing its previous link-visible permission to **Restricted** before uploading the clinical audio. Do not make the folder public to simplify downloads.

## Browser download

Sign into the authorized Google account, open the folder, and download these files individually:

- `text_data_v1.zip`
- `taukadial_data_v1.zip`
- `SHA256SUMS`
- `README_DATA.md`

Place the ZIPs under the repository's `artifacts/data_v1/`, install the project, and run `python scripts/restore_data.py`. This checks against the release manifest committed in Git, verifies every bundled file, and extracts the original speech archives. Files that differ from existing local data cause an error instead of being overwritten.

## RunPod or another headless Linux server

Install Python 3.10+, Git, curl, and [rclone](https://rclone.org/install/) using their official distribution/package sources. Clone the project and install it as shown in the README. Use persistent storage for the repository and data if the GPU pod may be terminated.

Configure a Google Drive remote called `gdrive` with your own authorized Google account:

```bash
rclone config
```

Choose Google Drive and the read-only scope (`drive.readonly`) for downloading. Use the [official Drive configuration guide](https://rclone.org/drive/), including its current OAuth client requirements. The documentation warns that the shared rclone client is being retired; an existing authorized remote or your own OAuth client may be necessary. No OAuth application or credential has been created by this project.

On a headless machine, answer **No** when rclone asks to open a browser. Follow the exact authorization command it prints on your own machine with a browser, and paste the resulting authorization response back into the rclone terminal. Follow the [official remote setup instructions](https://rclone.org/remote_setup/). Keep tokens and the rclone configuration out of Git and chat messages.

Once the remote is authorized:

```bash
bash scripts/download_from_drive.sh gdrive
python scripts/restore_data.py
python -m ccspeech.data.validate --full
```

The download script copies only the four named release files from the specified folder. It does not alter remote files, sharing settings, or other Drive folders. Authentication on a future RunPod instance is a setup step for that instance; successful preparation on the local computer does not imply that a new GPU instance is already authenticated.

For Phase 1 alone, Drive is optional: `python -m ccspeech.data.download` followed by `python -m ccspeech.data.text` reconstructs the same text preparation directly from pinned public sources.

## Access and provenance

TAUKADIAL access is managed through [DementiaBank / TalkBank](https://dementia.talkbank.org/TAUKADIAL/). Follow the [TalkBank ground rules](https://talkbank.org/0share/rules.html); the restricted personal archive is not a public dataset mirror. Do not share the clinical bundle with unauthorized recipients. The Amazon mirror and Yelp dataset also retain their upstream terms; availability of the project's code does not change them.

The release contains original data and deterministic preparation outputs. It does not include previously generated speech transcripts, prior model predictions, precomputed embeddings, authentication files, or workstation-only provenance paths.
