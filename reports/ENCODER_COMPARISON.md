# Controlled encoder comparison — 9 October 2026

**Decision: retain unchanged multilingual E5-small.** Neither 32k tokenizer passed the frozen development quality or English-retention criteria in either budget. Reserved-test scoring followed that decision and did not change it.

## Measured domain retrieval quality

Adapted values are means over seeds 17, 29 and 43. Intervals jointly resample seeds and the same intent groups, using 2,000 paired bootstrap draws. Development has 42 independent intents; reserved test has 18.

| Budget / encoder | Development BL→BN nDCG@10 | Test BL→BN nDCG@10 | Test difference vs adapted original [95% interval] |
|---|---:|---:|---|
| Unchanged E5 | 0.8039 | 0.8712 | — |
| equal-data / original | 0.8106 | 0.8565 | — |
| equal-data / unigram-32000 | 0.6788 | 0.8257 | -0.0309 [-0.1546, +0.0768] |
| equal-data / bpe-32000 | 0.7035 | 0.8069 | -0.0497 [-0.1789, +0.0722] |
| equal-compute / original | 0.8069 | 0.8603 | — |
| equal-compute / unigram-32000 | 0.7036 | 0.8173 | -0.0430 [-0.1596, +0.0624] |
| equal-compute / bpe-32000 | 0.6972 | 0.7757 | -0.0846 [-0.2085, +0.0319] |

Development custom-minus-adapted-original nDCG differences were −0.1071 to −0.1319 under equal data and −0.1033 to −0.1097 under equal compute. English-query mean nDCG fell by 0.0833–0.0928. Every custom arm failed the +0.03 gain, positive paired interval and ≤0.01 English-loss gates; all passed the latency and memory gates. The reserved-test intervals include zero, so they do not establish a tokenizer gain or a precise loss estimate.

## Inference cost on the development workload

CPU, four threads, maximum 512 tokens, batch 16. Warm batch-1 latency includes tokenization, encoding and exact cached-document search. Adapted values are medians across three fresh processes; each times 60 requests (12 distinct queries × five repeats). Peak RSS includes model loading and evaluation, not training.

| Budget / encoder | Batch-1 latency (ms) | Peak process RSS (MiB) | Query tokens / 126 texts |
|---|---:|---:|---:|
| unchanged | 4.221 | 888.2 | 2,265 |
| equal-data/original | 4.197 | 884.0 | 2,265 |
| equal-data/unigram-32000 | 4.257 | 473.9 | 3,141 |
| equal-data/bpe-32000 | 4.233 | 489.6 | 3,095 |
| equal-compute/original | 4.246 | 883.5 | 2,265 |
| equal-compute/unigram-32000 | 4.281 | 473.5 | 3,141 |
| equal-compute/bpe-32000 | 4.274 | 488.9 | 3,095 |

No development or test query was truncated. These small-corpus timings measure this CPU workload; they do not establish production p95 latency, GPU behavior or large-index search performance.

## Training and provenance

The body is the pinned E5-small snapshot `614241f622f53c4eeff9890bdc4f31cfecc418b3`. All adapted arms use the same 8,000 clean training groups, teacher representation objective, symmetric contrastive objective and optimizer. Equal-data runs use 64 representation steps plus 128 contrastive steps, batch 16 / maximum 128 tokens. Equal-compute runs have 120 wall seconds including loading, initialization and both stages; final-step overruns are recorded. All 18 checkpoints were trained and reloaded.

Custom embeddings start from means of original subword embeddings, with explicit control/byte fallbacks. This is heuristic initialization, not a reproduction of FOCUS or ZeTT. The feasibility budgets are small; these results do not rule out benefits from longer adaptation, better transfer or different objectives.

The six-domain corpus is agent-authored and agent-judged at the user’s request. All 3,600 intent-passage pairs have explicit labels projected across the authored parallel forms. There is no independent human or language-review claim. The corpus is small and several metrics approach their ceiling. Original sentence-pilot review and all 19 encoder scores are complete; see the [sentence report](RETRIEVAL_REVIEW.md) for its pool-coverage limits.

## Evidence

- [Frozen development selection](encoder-domain-selection-v2.json) and [reserved-test confirmation](encoder-domain-test-confirmation-v2.json).
- [Development suite](encoder-evaluation-development-reviewed-v2.json), [test suite](encoder-evaluation-test-reviewed-v2.json) and [full per-model reports](encoder-evaluation/).
- [Equal-data matrix](encoder-equal-data-matrix-verified-v1.json), [equal-compute matrix](encoder-equal-compute-matrix-verified-v2.json) and [training reports](encoder-training/).
- [Reviewed benchmark](DOMAIN_BENCHMARK.md), [data audit](vashantor-expanded-repro-audit-v2.json) and [criteria](../experiments/retrieval/criteria.json).
