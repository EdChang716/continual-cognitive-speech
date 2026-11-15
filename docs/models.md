# Shared model and data interfaces

All methods use the same backbone within a phase. The default text model is pretrained `bert-base-uncased`, using the last-layer `[CLS]` representation and a linear five-class head. The default audio model is the pretrained `openai/whisper-medium` **encoder**, followed by duration-masked mean pooling and a linear NC/MCI head. No Whisper decoder or generated transcript enters the audio classifier. Matched transcripts are retained in the prepared speech records for provenance and future explicitly specified linguistic extensions.

These are controlled classification backbones for this project's DIL experiments. They do not reproduce the encoder/decoder fusion architecture of WhisperCARE. WhisperCARE's 30-second segmentation and matched-transcript preparation are documented separately in the speech preprocessing protocol.

## Model contract

`build_model(model_config, num_classes)` returns a `Classifier` with:

- `encoder` and `classifier` submodules;
- `forward(inputs)` producing `[batch, num_classes]` logits;
- `encode(inputs)` producing the pooled retrieval representation;
- `input_keys()` specifying the accepted tensor fields;
- `metadata()` recording requested and resolved Hugging Face revisions, modality, and dimensions.

Model configuration keys include `modality`, `name`, optional pinned `revision`, `cache_dir`, `local_files_only`, and `gradient_checkpointing`. Checkpointing uses the non-reentrant PyTorch implementation so LoRA gradients remain valid when embedding/base weights are frozen. A model with a completely frozen encoder keeps that encoder in evaluation mode during head-only training.

`pretrained=False` requires an explicit `random_config`; it is reserved for offline smoke tests. A failed download never silently substitutes random weights. Real pretrained Whisper loading temporarily instantiates the complete Hugging Face model container to load the published keys correctly, then retains only the encoder.

## Update policies

`configure_updates(model, update_config)` supports `full`, `classifier_only`, `encoder_lora_only`, and `lora_classifier`. The latter two inject LoRA with `W(x) + (alpha/r) B(A(dropout(x)))`, random A and zero B; its initial function therefore equals the common pre-adaptation model. Frozen base weights and biases remain unchanged.

```json
{
  "policy": "lora_classifier",
  "rank": 8,
  "alpha": 16,
  "dropout": 0.05,
  "placement": "late",
  "num_layers": 4
}
```

`placement` is `early`, `late`, or `all`. Early and late comparisons must use the same `num_layers`, rank, and target set. `all` must select all encoder layers. Default BERT targets are `query` and `value`; default Whisper targets are `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, and `fc2`, matching the target-module family used in the reference WhisperCARE configurations. Explicit `targets` can narrow this factor. All selected layers and actual trainable parameter counts are returned for logging. Head-only versus LoRA-only naturally differ in parameter count; report the difference rather than claiming a matched budget.

Policies must branch from the same first-domain-trained checkpoint for the module-selection study. The engine controls this stage boundary. `encoder_lora_only` freezes the already trained classifier; it must not freeze an untrained random head and then be presented as that ablation. The named parameter groups returned by `parameter_groups` are disjoint `encoder` and `head` lists for uniform or different-frequency optimization.

## Dataset contract

`load_records(data_config, split, domain, allow_test_labels=False)` reads prepared Parquet or segment CSV files. Text domains are `amazon`/`yelp`; speech domains are `en`/`zh`. It retains quality/provenance columns, injects common `example_id`, `domain`, and `split` fields, and validates portable audio paths. Speech test diagnoses are joined by both recording and participant identity only when `allow_test_labels=True`; the model-facing test CSV must not contain them. Text test labels are also hidden by default.

`make_collator(model_config, max_length=256)` returns a callable producing `inputs` (tensor dictionary), `labels` (tensor or `None`), and unchanged `records` for identity-aware evaluation and replay. BERT tokenizes and truncates at the configured maximum length. Audio is read from the prepared 16 kHz mono segments without silent resampling or truncation. The Whisper feature extractor pads unpadded tails to the model window and supplies a valid-frame mask. The encoder pooling excludes padded output frames after the convolutional stride of two.

The runtime-only `_audio_file` field resolves the portable `audio_path` against the local data root. If replay records are restored on another machine, resolve this field again against that machine's root. Do not use it as a model feature.

Implementation references: [Hugging Face BERT](https://huggingface.co/docs/transformers/model_doc/bert), [Hugging Face Whisper](https://huggingface.co/docs/transformers/model_doc/whisper), and [Hu et al., LoRA](https://arxiv.org/abs/2106.09685).
