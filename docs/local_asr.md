# Local transcription of changed audio

The participant-audio builder writes `data/sources/speech_v3/asr_pending.jsonl`
for changed audio. Unchanged audio retains its hash-verified existing ASR.
Transcription is a separate, explicit command; preparation never starts it.

Run the following from the repository with a Python environment containing
`mlx==0.32.4`, `mlx-metal==0.32.4`, `mlx-whisper==0.4.3`, NumPy, and SoundFile.
MLX requires supported Apple hardware. The model must already exist locally.

```bash
PYTHONPATH=src python -m ccspeech.data.asr \
  --manifest data/sources/speech_v3/asr_pending.jsonl \
  --model-path /absolute/path/to/whisper-large-v3-mlx \
  --output data/reports/speech_v3/asr_changed
```

`--limit 1` performs a pilot. Repeating the command without the limit resumes
completed recordings and transcribes the remainder. A limit-complete status
does not claim the entire manifest is complete.

The runner verifies the existing model's download provenance and pinned config
and weight hashes. It refuses downloads, cloud-placeholder files, changed
source hashes, stale settings, and reference/label fields in the ASR manifest.
Rows contain `recording_id`, `language`, `audio_path`, and `sha256`; paths are
relative to the manifest directory. Optional `participant_id` and `split`
fields are discarded before decoding.

Only waveform and language enter Whisper. Decoding uses temperature 0, seed 0,
word timestamps, no initial prompt, and no previous-text conditioning. The
30-second windows advance by 27 seconds; timestamp midpoints give overlap
words one owner. ASR covers short recordings and final tails, independently
of which training windows are retained. Empty outputs remain empty and receive
an `empty_asr_transcript` flag.

Each `<recording_id>.json` records the source audio hash, timed words, complete
ASR window details, and `run_settings_sha256`, which is the SHA256 of the actual
`run.json` bytes. Completed files are published atomically and reused only when
their identity, source hash, language, and settings match. Any changed audio or
decoding protocol requires a new output version. These are model-generated
transcripts; source-reference text is never copied into recognition output.
