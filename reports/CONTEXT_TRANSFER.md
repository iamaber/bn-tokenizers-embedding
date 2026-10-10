# Sparse context initialization — 9 October 2026

Both 32k artifacts now have executed initialization tables using exact original
vocabulary overlaps and sparsemax-weighted training-context anchors. The
implementation uses deterministic 512-dimensional hashed co-occurrence, a
symmetric window of two, positive PMI and normalized cosine similarity.
Controls/bytes and unobserved pieces retain the existing decomposition fallback.
Only the shared 8,000 audited training groups supply contexts.

Unigram copied 8,806 exact anchors, initialized 11,404 additional learned pieces
from contexts and retained decomposition for 11,530 unseen/no-context pieces.
Building its table took 2.75 seconds. Both artifacts record their actual build
cost, anchor/context coverage, sparse support, base/data/tokenizer/table hashes
and method limitations in [the manifests](context-initialization-v1/).

The real Unigram table passed finite normalized inference and saved-checkpoint
reload equality. Tests also verify sparsemax support/normalization and reject
changed tokenizer, base or table files before changing weights.

This is **FOCUS-inspired**, not a FOCUS reproduction: hashed PPMI contexts replace
its trained fastText auxiliary embeddings. Native raw pieces also differ in
word-boundary semantics. No retrieval gain is claimed before matched adaptation
and fresh evaluation. The [source review](ROUND2_SOURCE_RESEARCH.md) explains
why the available ZeTT XLM-R hypernetwork is not a drop-in for E5-small.
