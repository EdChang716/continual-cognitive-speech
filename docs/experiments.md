# Execution and interpretation

## Controlled design

The main question is when historical label/logit replay and retrieved local updates preserve an earlier domain while learning a new one, and whether the same mechanism holds for cognitive speech. Amazon/Yelp share five rating labels, and English/Mandarin TAUKADIAL share two cognitive labels. No new classification outputs are added at the transition. This is two-stage DIL, not the original MbPA++ five-dataset mixed-task protocol, a class-incremental benchmark, or evidence of long-term deployment.

A full supervised source warmup trains the pretrained encoder and randomly initialized linear head, with source validation only used for checkpoint selection. Its initial pretrained encoder is frozen separately for retrieval. A common saved source checkpoint initializes every compatible method and update policy. The source reservoir is built during the first pass, retaining pre-update logits and fixed keys. Reversed domain orders reuse the same prepared participant/review splits. Both domains are evaluated after source learning and after adaptation; unseen-domain scores do not count as old-domain forgetting.

During target adaptation, ER adds labeled replay to each current batch. DER++ adds historical-logit MSE and a separately sampled label-replay batch. Sparse replay only, nearest MbPA++ and random-neighbor MbPA++ share the same sparse global updates; their inference differs. Each local query uses only stored training labels, receives temporary Adam updates, predicts, and restores the global model completely. No test-time query label enters retrieval or adaptation. Sequential fine-tuning discards the source reservoir and fixed retrieval encoder before target training, so its effective adaptation memory is zero.

The source warmup cost is shared setup and recorded separately. Joint training sees the domain union from the pretrained initialization. Separate-domain models train independently from that initialization; they are references, not equal-compute sequential competitors. Joint/separate appear for both orders to support order-aligned reports; these reference entries are scientifically redundant and need only be run for one order if saving compute.

## Defaults and method differences

| Setting | Text | Speech |
|---|---|---|
| Backbone | Pinned BERT-base-uncased | Pinned Whisper-medium **encoder only** |
| Current batch | 16 | 2 |
| Source / target epochs | 1 / 1 | 5 / 5 |
| Global learning rate | 3e-5 | 3e-5 |
| Core memory | 256, 1,024 reviews | 32, 128 segments |
| DER++ alpha / beta | 0.5 / 0.5 | 0.5 / 0.5 |
| Sparse replay frequency | Every 10,000 unique new examples | Every 100 unique new segments |
| Sparse replay batch | Up to 100 | Up to 100 |
| Local retrieval / steps | Up to 32 / 30 | Up to 32 / 30 |
| Local Adam lr / anchor | 0.001 / 0.001 | 0.001 / 0.001 |
| LoRA rank / alpha | 8 / 16 | 8 / 16 |
| Early / late placement | First / last 3 layers | First / last 6 layers |

Global optimization uses AdamW with separate head/encoder parameter groups and per-group gradient clipping; local adaptation uses Adam. These choices are held fixed across controlled comparisons and are not claimed as the original papers' complete optimization recipes.

The speech sparse frequency is intentionally changed because 10,000 exceeds a whole small speech cohort, producing no sparse updates. Frequency counts unique first-epoch arrivals; later epochs revisit already available current-domain data without adding new reservoir offers. Sampling is without replacement up to available capacity; a 32-slot bank cannot supply 100 distinct replay examples. Actual presentations and step counts are logged. Source/target epochs and retrieval settings are proposed defaults requiring validation, not values proven optimal for this task.

Whisper's decoder is excluded. Matched transcripts and diagnostic quality fields never become model inputs. Text uses BERT [CLS]; audio uses a masked mean of encoder features, with padded positions excluded from pooling. This is an audio-classification extension, not a reconstruction of WhisperCARE's encoder/decoder fusion classifier.

Module ablations start with a trained source head: full model, head only, encoder LoRA with head frozen, or LoRA plus head. BERT targets query/value projections; Whisper targets q/k/v/out attention projections and fc1/fc2, consistent with the chosen reference pipeline's targets. Early/late placement selects equally many layers and the same rank within each backbone; report actual trainable parameters. No comparison assumes equal parameter counts across BERT and Whisper.

The Nested Learning-inspired condition accumulates separate head/encoder gradients and updates them at different intervals. Compare 1/4 with uniform1/1 and uniform4/4; the latter matches encoder step count. Pending gradients are averaged and a final partial window is flushed. This is a mechanistic frequency ablation; HOPE, Titans, and a learned neural long-term memory are **not** implemented.

## Outputs and reproducibility

`runs/<phase>/<source-to-target>/<method-policy-seed-hash>/` contains `run.json` (configuration, exact prepared-data hashes, model revision, environment), `source.json`, validation prediction CSVs, `target/best.pt`, `target/last.pt`, and `result.json`. Reference runs use per-reference subdirectories. Shared source stages live under `runs/_source/<hash>/`; retain that folder for subsequent final-test/source-retention evaluation. Relocating the whole runs tree and the data root is supported. Keep filenames and relative layout intact.

`best.pt` is a selection checkpoint with model, memory, RNG, and metadata. `last.pt` additionally has full optimizer/pending-gradient state for exact resume. A nonblocking OS lock prevents two processes writing the same run. Run IDs exclude machine-specific data/cache paths and device but include scientific settings. Changed data or incompatible resume settings fail explicitly. A changed backbone/config creates a distinct run; random initialization is allowed only in explicit smoke configs, never as a fallback for download failures.

Fixed retrieval weights are reconstructed from the pinned original backbone and seed. Memory stores historical logits and fixed keys; its serialized records strip absolute audio paths, and restored paths must remain under the current data root. Evaluations restore training RNG and local-retrieval RNG, so validation frequency does not perturb subsequent training. Final-test is a separate command, joins locked speech labels only for scoring, and preserves the report as an immutable record.

## Compute and storage planning

No RunPod instance is provisioned automatically. Start with one pilot using the intended GPU and inspect actual CUDA peak memory, training/evaluation times, and local-adaptation presentations in reports. CUDA execution has not been hardware-tested during preparation. Float32 is the conservative default; bfloat16 can be selected on a supported CUDA GPU. Keys and local adaptation stay float32. A pilot is not a research result, and a short pilot does not prove full-run memory or timing.

The default text validation set has 20,000 reviews. At32neighbors×30steps, MbPA++ requires19.2million neighbor presentations for one pass, plus retrieval and query inference. Source-stage reporting, each target epoch's validation, and final reporting also incur work. The two text test domains add15,200queries per evaluation stage. Consider validation-set size/local-step budget as an explicit protocol revision if costs are impractical; keep those settings identical for relevant controls and do not choose them on test scores. The provided full suite does not silently subsample evaluation.

Checkpoints can dominate disk use. For P float32 model parameters, model weights alone require roughly4P bytes; Adam adds about8P for fully trainable parameters, with additional pending gradients and episodic memory. One source and each target can have both best and last checkpoints. LoRA reduces optimizer storage but this implementation still saves full model weights. Run families incrementally and archive completed runs to persistent private storage. Keep shared source best checkpoints and selected target best checkpoints for final evaluation; deleting last checkpoints forfeits interrupted-run resume for those stages. `training_stats.estimated_checkpoint_tensor_bytes` and `checkpoint_disk_bytes` expose actual estimates/sizes instead of a guaranteed fixed disk allowance.

## Evaluation

Both phases report accuracy, macro-F1, balanced accuracy, and AUROC. Five-way text AUROC is macro one-vs-rest; speech AUROC is binary MCI probability. Accuracy supports comparison of the adapted text methods, without claiming equality to published MbPA++ scores. Source-score change is after-minus-before accuracy; forgetting is the earlier learned-domain score minus its final score. With two stages it is one transition, and improvement can yield negative forgetting.

Speech averages overlapping windows within recordings, then recordings equally within people. Bootstrap participants, not windows; final speech reports include individual-method intervals and a paired interval for source-score change. Seed SD and participant uncertainty answer different questions and remain separate. Undefined AUROC yields null/status and reports the number of valid bootstrap draws. Text bootstrap is off by default because of its larger sample count; set `evaluation.bootstrap_text=true` explicitly if desired.

Integrity and engineering smoke checks do not establish clinical validity or algorithmic novelty. Inspect uncertain-speaker cases and conduct any quality-filter sensitivity analysis using a fixed, predeclared rule; preserve patient splits, and avoid making preprocessing choices from test labels.
