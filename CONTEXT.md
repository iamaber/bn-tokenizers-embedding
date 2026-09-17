# Domain context

## Core concepts

- **Tokenizer**: the immutable inference runtime that normalizes text, encodes it into token IDs, decodes IDs, and optionally samples Unigram segmentations.
- **Model**: the versioned JSON vocabulary and algorithm contract loaded by a Tokenizer. A model is either Unigram or BPE and uses the `nfc-whitespace-v1` normalization contract.
- **Unigram segmentation**: maximum-likelihood, forward/backward, or sampled segmentation over learned pieces. Encoding and sampling include byte fallback; training uses learned pieces only.
- **BPE segmentation**: ranked merge application over learned character pieces with byte fallback.
- **Training input**: positive weighted, normalized, space-free word counts plus the deterministic observed alphabet used by Unigram and BPE training.
- **Corpus split**: the provenance-preserving train, validation, and test assignment produced after grouping paired forms, duplicates, and near-duplicates.
- **Native handle**: the numeric lifetime token held by Python for a loaded Go Tokenizer through the C bridge.
- **Evaluation**: held-out measurement of round trips, sequence lengths, fallback, vocabulary use, and throughput.

## Current constraints

- Refactors must preserve the public Python API, token IDs, normalized round trips, seeded sampling behavior, model format, and corpus output.
- Exported C symbols and their observable ownership/error behavior remain stable while the native handle implementation is deepened behind them.
- The Go engine remains usable without Python or CGo; the Python package remains backed by the native Go runtime.
- No new runtime dependency or network requirement is introduced.

## Architecture decisions

- The native handle keeps `bntok_call`, `bntok_encode`, and `bntok_free` stable. JSON control and binary encoding remain adapters, while lifecycle, operation semantics, error translation, and ownership become one deep implementation.
- Unigram scoring belongs to one internal module with distinct Viterbi, expected-count, and sampling operations. Preserve score accumulation, tie order, RNG draw order, and the difference between training and inference fallback.
- Training-input preparation owns shared word validation, deterministic word ordering, and observed characters. Each trainer retains its configuration checks, vocabulary policy, and existing error messages.
- Corpus split preparation owns grouping, heldout propagation, filtering, deduplication, and accounting before serialization. Saved-file audits and official holdout audits stay independent of preparation, so they can detect mistakes in its output.
