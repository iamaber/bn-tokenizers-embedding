# Reviewed negatives and extended training

The unchanged, pinned E5 encoder mined 128 near-neighbour candidates from the
original 8,000 TRAIN-only alignment pairs: 64 English and 64 Banglish queries.
All candidates were explicitly judged by the implementation agent, as authorized
by the user. These are single-agent judgments, not independent human review.

The review identified 18 incorrect or uncertain positive pairs, 30 partially
relevant neighbours and four equivalent alternatives. Only a correct positive
with a grade-zero neighbour enters the negative bank: 79 examples, comprising
35 English and 44 Banglish queries. The 49 exclusions overlap those categories.
The complete text-free decisions and input bindings are in
[the review archive](../experiments/retrieval/hard_negative_review_v1/).

A new alignment version removes the 18 flagged positives from ordinary batches
as well. It retains 7,982 pairs: 2,080 English and 5,902 Banglish. The previous
alignment and experiment remain unchanged. Fidelity of the unsampled positive
pairs, and semantic false negatives within ordinary batches, remain unverified.

[The frozen training protocol](../experiments/retrieval/round2_training_protocol.json)
uses five arms: original tokenizer, Unigram mean/context initialization and BPE
mean/context initialization. Each has seeds 17, 29 and 43 under both budgets.
Equal-data exposure is 192 teacher representation steps and 384 contrastive
steps, three times the previous step counts. Equal-compute exposure is 360
charged seconds, including model loading and initialization; context-table build
cost is conservatively charged in full per run. Saving and input validation are
outside that budget. Completed optimizer steps can overrun the time boundary;
reports retain that measured overrun.

All batches contain eight English and eight Banglish queries. Every third
contrastive step adds the reviewed positive-versus-negative objective. An
English-query cosine retention term, weighted 0.2, accompanies contrastive
training. Matched seeds consume the same sample order across equal-data arms.
These objectives are hypotheses; fresh quality results must establish whether
they preserve English performance or improve Banglish retrieval.

Context initialization v2 uses precisely the revised alignment and is bound by
the protocol. The earlier v1 tables remain available for provenance. Original
and Unigram-context smoke runs executed two representation and three contrastive
steps, including 16 verified negative examples. Both produced finite normalized
vectors and bitwise-identical inference after save/reload. See
[the verification record](round2-smoke-verification.json).

Fresh benchmark authoring is still in progress. Its source passages, agent
questions, judgments and selection rule must freeze before any adapted quality
evaluation. The completed first experiment's selected unchanged E5 is preserved.
