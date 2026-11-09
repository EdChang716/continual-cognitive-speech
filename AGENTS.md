# Project conventions

- Keep data acquisition/preparation in `src/ccspeech/data`, configuration in `configs`, and thin entry points in `scripts`.
- Add model backbones, memory algorithms, training loops, and evaluation in separate subpackages when implemented. Share their data contracts; do not copy complete pipelines per method.
- Keep raw datasets, all participant-level metadata, predictions, checkpoints, embeddings, tokens, and downloaded bundles under ignored directories. Never add them to Git.
- Use the committed source/release manifests for dataset revisions and integrity. Preserve raw inputs and explicitly version changed protocols.
- Keep official speech-test subjects separate; split by participant before chunking. Test labels never enter adaptation, retrieval, replay, or hyperparameter selection.
- Use identical source splits across method comparisons and reversed domain orders.
- Add meaningful tests for leakage, data corruption, and changed behavior. Run `python -m pytest -q`; use `PYTHONPATH=src` in bundled runtimes if editable imports are unavailable.
- Avoid large model dependencies for the data-only CLI. Do not start GPU jobs or provision paid resources as a side effect of data preparation.
- Document assumptions and deviations from the original papers. Do not describe adapted two-domain results as reproducing the original five-domain MbPA++ scores.
