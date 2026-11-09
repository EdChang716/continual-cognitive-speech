# Amazon and Yelp data protocol

Phase 1 compares continual-learning memory and parameter-update mechanisms under a shared five-class label space. Amazon and Yelp are different review domains; both predict ratings from one to five stars. Run Amazon → Yelp and Yelp → Amazon using the same prepared records. A domain identifier is retained as evaluation metadata, not supplied as an extra classification input.

This is an adapted two-domain domain-incremental learning (DIL) experiment. It does not reproduce the numerical scores of MbPA++'s original five-dataset, 33-class benchmark. [MbPA++ §4.1](https://papers.neurips.cc/paper/9471-episodic-memory-in-lifelong-language-learning.pdf) used 115,000 training examples and 7,600 test examples per dataset, with shared Amazon/Yelp rating labels. We retain these sample counts and reserve an additional 10,000 validation examples per domain from the original training split. Exact class balancing and duplicate filtering are additional choices in this project.

## Pinned raw sources

| Domain | Training rows | Test rows | Fields |
| --- | ---: | ---: | --- |
| [Amazon Review Full mirror](https://huggingface.co/datasets/vgaraujov/amazon_review_full) | 3,000,000 | 650,000 | `label`, `title`, `content` |
| [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full) | 650,000 | 50,000 | `label`, `text` |

`configs/data/sources.json` pins repository revisions, source file sizes, SHA-256 digests, and expected row counts. Both pinned sources already encode labels as `0..4`; `0` means one star and `4` means five stars. Preparation rejects invalid or missing labels. It must not subtract one from these labels.

Raw files remain unchanged. Amazon input text is exactly `title + "\n" + content`, including empty title or body fields if the combined text is nonempty. Yelp input text is unchanged. No tokenization, truncation, embeddings, or model-specific preprocessing is performed at this stage.

## Rebuild

From the repository root, after installing the package and acquiring the raw data:

```bash
python -m ccspeech.data.text --data-root data --seed 42
```

The default prepared sizes **per domain** are:

| Split | Total | Per rating class | Source |
| --- | ---: | ---: | --- |
| Train | 115,000 | 23,000 | Original training split |
| Validation | 10,000 | 2,000 | Original training split, disjoint from selected training records |
| Test | 7,600 | 1,520 | Original test split |

For a pilot, use a separate data root containing the same raw sources, or explicitly rebuild with `--overwrite`:

```bash
python -m ccspeech.data.text --data-root data --seed 42 \
  --train-size 1000 --validation-size 500 --test-size 500 --overwrite
```

All requested counts must be positive multiples of five. Overwriting replaces the prepared text datasets and their report only after preparation succeeds; it does not modify raw sources. The pilot command above replaces any existing default preparation, so rerun the default command with `--overwrite` before the main experiments.

## Duplicate and sampling rules

1. Verify all four raw file sizes and hashes before reading records.
2. Form a duplicate key by applying Unicode NFKC normalization, case folding, and whitespace collapse, then computing SHA-256. This normalization is used only for duplicate detection. The model-input text retains its original form.
3. Scan **all official test records first**, in Amazon then Yelp order. Keep the first occurrence of each normalized text. This ensures a training example cannot duplicate any official test text, including test examples not selected for our 7,600-example evaluation subset.
4. Scan all training records in Amazon then Yelp order. Reject duplicate hashes already seen in either official test set or earlier training records. This also removes cross-domain duplicate texts. Duplicate ownership follows this fixed order for both experiment directions. Conflicting labels are counted; the first occurrence remains the owner.
5. Exclude whitespace-only normalized inputs and record their counts. Missing text fields, invalid labels, unexpected raw row counts, or insufficient eligible class counts stop preparation.
6. For each domain and class, select the smallest SHA-256 priorities derived from the protocol version, seed, source domain, source split, and zero-based source row. This is reproducible pseudorandom sampling over all eligible unique records, without loading the full corpus into memory.
7. For training-source selections, use a separate SHA-256 priority to assign 2,000 of each class to validation and the remaining 23,000 to training. Never form validation from the official test set.
8. Verify exact counts, per-class balance, and zero normalized-text overlap across **all six** prepared splits before publishing outputs.

Hash-based duplicate ownership operates on the entire raw corpus, not just the final sample. It therefore changes the population relative to the unfiltered benchmark; report these differences when comparing results. It detects normalized exact duplicates, not semantic paraphrases or all possible author/product overlap. The source files do not provide reviewer/product identifiers for grouping.

## Outputs and provenance

Prepared files are `data/processed/text/{amazon,yelp}/{train,validation,test}.parquet`. Every record has:

| Field | Meaning |
| --- | --- |
| `example_id` | Domain, original split, and zero-based original row |
| `domain` | `amazon` or `yelp` |
| `source_split` | Original `train` or `test` |
| `source_row` | Zero-based CSV record / parquet row, excluding CSV header |
| `text` | Model input before tokenization |
| `label` | Integer rating class `0..4` |
| `text_sha256` | SHA-256 of normalized duplicate-detection text |

`data/processed/text/index.csv` records selected source rows and assignments without review text. `data/reports/text.json` contains the source manifest and hash, revisions, seed, filtering rules, raw and eligible counts, empty/duplicate/conflicting-label counts, output hashes, class balance, and final overlap check. Preparation logs and reports do not contain review excerpts. Scratch SQLite files are held under `data/tmp` and removed on completion.

Prepared records are written in source-row order for deterministic artifacts. Training code must shuffle within each arriving domain using its experiment seed. All methods and both domain orders must reuse the same preparation report and index file. Replay and query-local adaptation may use only training records encountered so far; validation and test labels never enter the memory bank.

The original corpus construction is described by [Zhang, Zhao, and LeCun (2015)](https://arxiv.org/abs/1509.01626). Dataset access and redistribution terms remain separate from this repository's code license; consult the source pages.
