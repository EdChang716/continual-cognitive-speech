# Memory methods and evaluation contracts

The project compares fixed-label, two-domain adaptation under common source initialization and finite episodic memory. Its text and speech settings are adaptations of the cited algorithms. Their scores must not be presented as reproducing the original five-dataset MbPA++ benchmark or a full Nested Learning architecture.

## Replay and retrieval

`ReservoirMemory` admits only explicitly marked training records. Each arriving example is offered once; replay, validation, and test examples are never inserted. A memory entry contains a record reference, optional detached historical logits, and an optional fixed retrieval key, all off GPU. Source references count against the logical example capacity even though immutable raw files remain on disk. Report physical referenced input sizes separately from keys, logits, and serialized record sizes.

Admission, global replay sampling, and random local retrieval have separate deterministic random streams. Use `sample(n)` for training and `sample(n, purpose="local")` for the random-retrieval control. This prevents running additional evaluation queries from changing the future reservoir or training replay draws. Nearest retrieval uses unnormalized Euclidean distance and deterministic tie breaking.

The ER objective is current-batch mean cross-entropy plus replay-batch mean cross-entropy, with replay coefficient 1 in the supplied experiment driver. DER++ adds historical-logit mean-squared error and labeled replay using independent memory draws. Store each incoming batch's detached **pre-update** logits during the source and target training trajectories, rather than recomputing targets after the domain finishes. The elementwise MSE convention follows the [authors' Mammoth implementation](https://github.com/aimagelab/mammoth/blob/master/models/derpp.py); see [Buzzega et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf).

The original [MbPA++ paper, Sections 2–4](https://papers.neurips.cc/paper/9471-episodic-memory-in-lifelong-language-learning.pdf) uses a separate frozen pretrained key encoder, Euclidean retrieval, and equally weighted local examples. Its reported settings include 32 neighbors, 30 Adam local updates, anchor coefficient 0.001, and 100 replay examples per 10,000 new examples. Local learning rates are 0.001 or 0.005. Our finite reservoir, shared-label DIL task, audio encoder, and selective-update variants are explicit experimental changes. A model-specific configuration may change these settings; report its actual values and costs.

## Temporary local adaptation

`local_adapt_predict(model, query_inputs, neighbor_inputs, neighbor_labels, ...)` accepts exactly one query and **no query label**. Its objective is mean neighbor cross-entropy plus `anchor_strength * sum((parameter - global_parameter)^2)` over currently trainable parameters. Neighbors receive uniform weights; retrieval distance does not weight the loss.

Neighbor microbatches contribute their sample fraction to the mean loss. The anchor gradient is added exactly once per local step. A fresh local optimizer is discarded after prediction. A `finally` block restores trainable weights, original gradients, model buffers, and each module's training/evaluation state, including on failure. Frozen parameters remain frozen. Local predictions therefore do not update the global model or its optimizer state. Use a gradient-enabled outer context or `torch.no_grad`; do not wrap local adaptation in `torch.inference_mode`.

The full-update variant can require substantial memory because temporary parameter anchors and Adam moments are additional to the global model. `microbatch_size` controls activation memory without changing equal-neighbor weighting. Local steps and neighbors add inference cost and must be included in latency accounting.

Local adaptation enables dropout as part of training. The evaluation driver should snapshot and restore PyTorch RNG state around the complete evaluation pass so that its timing or frequency cannot alter subsequent global-training dropout. Temporary adaptation itself restores model state, while the driver owns evaluation-wide randomness.

## Update-frequency ablations

`TimescaleOptimizer(head_parameters, encoder_parameters, head_interval=..., slow_interval=...)` supports three comparisons: uniform `1/1`, fast-head/slow-encoder `1/k`, and uniform `k/k`. The last condition matches the slow encoder update count. Each group averages gradients over its own accumulation window. A final incomplete window is averaged over its actual number of batches and flushed. These are frequency ablations inspired by multiple timescales; they do not implement HOPE.

The training loop calls `zero_grad()`, computes a batch objective and `backward()`, then calls `step()`. Accumulated gradients are separate from `.grad`, so `zero_grad()` does not erase pending updates. Call `flush()` at the stage boundary before evaluation/checkpointing. Optimizer checkpoints include moments, pending accumulations, and update counters. Report examples, gradient batches, both actual optimizer-step counts, wall time, and trainable parameters rather than equating equal epoch counts with equal computation.

## Evaluation units

`classification_metrics` reports accuracy, macro-F1 over every configured class, balanced accuracy over observed classes, and AUROC. Binary AUROC uses class-1 probability; multiclass AUROC is macro one-versus-rest. When any class is missing, AUROC is null with an explicit status instead of a fabricated number.

For speech, `aggregate_speech_predictions` first averages window probabilities within each recording, then equally averages recordings within a participant. A long recording consequently does not gain more influence merely because it produced more overlapping windows. Duplicate segment predictions, conflicting participant labels, and inconsistent cohort metadata are errors. Both methods and domain orders use the same aggregation.

Bootstrap intervals resample evaluation units, so speech callers must pass the **aggregated participants**. The default uses ordinary participant resampling; undefined AUROC draws are excluded only from the AUROC interval and their valid count is reported. Stratified resampling is an explicitly recorded option. `continual_summary` uses the equal-domain mean of final accuracy and reports source-score change and forgetting from the stage-by-domain score matrix. An improvement can yield negative forgetting. Unseen-domain scores are not used as old-domain performance peaks.
