"""Read prepared records and collate raw text/audio for trainable encoders."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from torch.utils.data import Dataset


def _safe_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError("Dataset paths must be relative to the data root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Dataset path escapes the data root")
    return path


def _label(value: Any, count: int) -> int:
    if isinstance(value, bool) or str(value) not in {str(index) for index in range(count)}:
        raise ValueError(f"Expected an integer label in 0..{count - 1}")
    return int(value)


def load_records(config: dict, split: str, domain: str, *, allow_test_labels: bool = False) -> list[dict]:
    """Load one prepared domain. Test labels require an explicit evaluation gate."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("Unknown split")
    root = Path(config.get("root", "data")).resolve()
    modality = config.get("modality")
    if modality == "text":
        if domain not in {"amazon", "yelp"}:
            raise ValueError("Text domain must be amazon or yelp")
        relative = f"{config.get('text_dir', 'processed/text')}/{domain}/{split}.parquet"
        records = pq.ParquetFile(_safe_path(root, relative)).read().to_pylist()
        for row in records:
            if row.get("domain") != domain or row.get("source_split") != ("test" if split == "test" else "train"):
                raise ValueError("Text record belongs to a different domain or source split")
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError("Text records must contain nonempty input text")
            row["split"] = split
            if split != "test" or allow_test_labels:
                row["label"] = _label(row.get("label"), 5)
            else:
                row.pop("label", None)
    elif modality == "audio":
        if domain not in {"en", "zh"}:
            raise ValueError("Speech domain must be en or zh")
        relative = f"{config.get('speech_dir', 'processed/speech_v2')}/{split}.csv"
        with _safe_path(root, relative).open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if split == "test" and "label" in (reader.fieldnames or []):
                raise ValueError("Model-facing speech test manifest must not contain labels")
            records = [dict(row) for row in reader if row.get("language") == domain]
        truth = {}
        if split == "test" and allow_test_labels:
            truth_path = _safe_path(root, config.get("test_labels", "locked/taukadial/test_labels.csv"))
            with truth_path.open(encoding="utf-8", newline="") as source:
                for row in csv.DictReader(source):
                    recording_id = row["recording_id"]
                    if recording_id in truth:
                        raise ValueError("Duplicate official test label entry")
                    truth[recording_id] = (row["participant_id"], _label(row["label"], 2))
        for row in records:
            if row.get("split") != split:
                raise ValueError("Speech record belongs to a different split")
            for required in ("participant_id", "recording_id", "segment_id", "audio_path", "matched_transcript"):
                if required not in row:
                    raise ValueError(f"Speech manifest is missing {required}")
            if not row["segment_id"] or not row["participant_id"] or not row["recording_id"]:
                raise ValueError("Speech identities must be nonempty")
            path = _safe_path(root, row["audio_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            row["example_id"] = row["segment_id"]
            row["domain"] = domain
            # This private local path is runtime-only and never written back to
            # the portable CSV or used as model input.
            row["_audio_file"] = str(path)
            if split != "test":
                row["label"] = _label(row.get("label"), 2)
            elif allow_test_labels:
                item = truth.get(row["recording_id"])
                if item is None or item[0] != row["participant_id"]:
                    raise ValueError("Official test labels do not match the speech identities")
                row["label"] = item[1]
            else:
                row.pop("label", None)
    else:
        raise ValueError("data.modality must be text or audio")
    if not records:
        raise ValueError(f"No records in {domain}/{split}")
    identifiers = [row["example_id"] for row in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate model-facing example IDs")
    return records


class RecordDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return dict(self.records[index])


class RecordCollator:
    def __init__(self, model_config: dict, max_length: int = 256):
        self.config = dict(model_config)
        self.modality = model_config["modality"]
        self.max_length = max_length
        if max_length < 1:
            raise ValueError("max_length must be positive")
        if self.modality == "text":
            from transformers import AutoTokenizer, BertTokenizerFast
            if model_config.get("vocab_file"):
                self.processor = BertTokenizerFast(vocab_file=str(model_config["vocab_file"]), do_lower_case=True)
            else:
                kwargs = {"local_files_only": bool(model_config.get("local_files_only", False))}
                if model_config.get("revision"):
                    kwargs["revision"] = model_config["revision"]
                if model_config.get("cache_dir"):
                    kwargs["cache_dir"] = model_config["cache_dir"]
                self.processor = AutoTokenizer.from_pretrained(model_config.get("name", "bert-base-uncased"), **kwargs)
        elif self.modality == "audio":
            from transformers import WhisperFeatureExtractor
            if model_config.get("pretrained", True):
                kwargs = {"local_files_only": bool(model_config.get("local_files_only", False))}
                if model_config.get("revision"):
                    kwargs["revision"] = model_config["revision"]
                if model_config.get("cache_dir"):
                    kwargs["cache_dir"] = model_config["cache_dir"]
                self.processor = WhisperFeatureExtractor.from_pretrained(
                    model_config.get("name", "openai/whisper-medium"), **kwargs)
                self.max_samples = int(self.processor.n_samples)
                if self.processor.sampling_rate != 16000:
                    raise ValueError("This prepared speech protocol requires a 16 kHz feature extractor")
            else:
                # Explicit random configs permit small, fully offline tests.
                random = model_config.get("random_config", {})
                mel_bins = random.get("num_mel_bins", 80)
                frames = random.get("max_source_positions", 1500) * 2
                self.max_samples = frames * 160
                self.processor = WhisperFeatureExtractor(feature_size=mel_bins, sampling_rate=16000,
                                                         hop_length=160, chunk_length=self.max_samples / 16000,
                                                         return_attention_mask=True)
        else:
            raise ValueError("Unknown collator modality")

    def __call__(self, records: list[dict]) -> dict:
        if not records:
            raise ValueError("Cannot collate an empty batch")
        labeled = ["label" in row for row in records]
        if any(labeled) and not all(labeled):
            raise ValueError("A batch cannot mix labeled and unlabeled records")
        labels = torch.tensor([row["label"] for row in records], dtype=torch.long) if all(labeled) else None
        if self.modality == "text":
            if any(not isinstance(row.get("text"), str) or not row["text"].strip() for row in records):
                raise ValueError("Missing or empty text input")
            inputs = dict(self.processor([row["text"] for row in records], padding=True, truncation=True,
                                         max_length=self.max_length, return_tensors="pt"))
        else:
            waves = []
            for row in records:
                if "_audio_file" not in row:
                    raise ValueError("Audio records must be resolved by load_records before collation")
                waveform, sample_rate = sf.read(row["_audio_file"], dtype="float32")
                if sample_rate != 16000 or waveform.ndim != 1:
                    raise ValueError("Prepared speech must be 16 kHz mono; no silent resampling")
                if not len(waveform) or len(waveform) > self.max_samples:
                    raise ValueError("Prepared segment must be nonempty and fit the model window")
                if not np.isfinite(waveform).all():
                    raise ValueError("Nonfinite audio samples")
                if "num_samples" in row and int(row["num_samples"]) != len(waveform):
                    raise ValueError("Audio sample count differs from the prepared manifest")
                waves.append(waveform)
            features = self.processor(waves, sampling_rate=16000, padding="max_length",
                                      max_length=self.max_samples, truncation=False,
                                      return_attention_mask=True, return_tensors="pt")
            inputs = {"input_features": features["input_features"], "attention_mask": features["attention_mask"]}
        return {"inputs": inputs, "labels": labels, "records": records}


def make_collator(model_config: dict, max_length: int = 256) -> RecordCollator:
    return RecordCollator(model_config, max_length=max_length)
