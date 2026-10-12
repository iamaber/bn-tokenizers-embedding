# Completed sentence review and failure audit — 9 October 2026

All 3,152 original E5/BM25/source-pair pooled judgments and all 300 source-pair decisions are complete. The user authorized agent judgments. These are **single-agent sentence-correspondence labels**, with no independent human-review claim.

The rubric assigns 2 for the same essential meaning, preserving negation, names and numbers; 1 for meaningfully related but incomplete correspondence; and 0 for a different or contradictory sentence. There are 255 grade-2, 226 grade-1 and 2,671 grade-0 judgments in the full original packet. The reviewed cohort retains 240 queries (236 Banglish, four mixed-script) and 1,996 candidate documents. Sixty queries are excluded for wrong or uninterpretable correspondence, including four with exposed personal contact/account identifiers. Every retained query has a grade-2 positive.

The [complete text-free annotation archive](../experiments/retrieval/pilot-annotation-complete-v1/) binds its 3,152 labels and 300 decisions to the exact frozen source packet. Restore reconstructs text locally from the original corpus, then strict import checks complete coverage, source identity, grades, reasons and provenance. Privacy-excluded groups are removed from every copied text artifact. The earlier draft archive is a historical partial snapshot. Raw corpus text remains local and ignored by Git. The [reviewed manifest](retrieval-reviewed-v1.json) records every exclusion and artifact hash.

## Baseline on fixed reviewed labels

| System / label view | Recall@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|
| BM25 / graded | 0.0104 | 0.0093 | 0.0092 |
| Unchanged E5 / graded | 0.7804 | 0.6667 | 0.6746 |
| BM25 / exact grade 2 | 0.0125 | 0.0093 | 0.0092 |
| Unchanged E5 / exact grade 2 | 0.7646 | 0.6629 | 0.6322 |

Graded Recall counts all positive labels, including partial matches, and divides by the number of known positives; it is not a binary hit rate. The exact sensitivity view treats grade 1 as zero. All 240 queries remain eligible in both views. Unknown candidates contribute zero to the numerical proxy but remain explicitly unjudged. The unchanged encoder misses every positive for 49 reviewed queries; filtering the cohort and adding alternatives changes what is being measured, so its reviewed score cannot be called a model improvement over the original 300-query score.

## What the 73 historical misses reveal

All 73 raw E5 misses have [individual agent observations](../experiments/retrieval/pilot-failure-judgments-v1.jsonl), with source/annotation hashes in the [audit report](retrieval-failure-review-v1.json):

- 15 belong to excluded source pairs.
- Nine already retrieve a reviewed positive, including partial alternatives. One has an incorrect provisional target and retrieves the corrected exact counterpart.
- 49 remain misses among retained queries.

Observed issues include negation and outcome contrasts, short context-dependent utterances, colloquial spelling, English insertions, and different intent/scope despite overlapping topic words. These are diagnostic observations, not proven causes inside the encoder. No model intervention isolated their mechanisms. Two historical provisional targets differ from the actual source counterparts; the full pooled review handles them rather than trusting group identity as a pair label.

## Encoder comparison and pool limits

All 19 systems were scored on the original 300-query pilot and the reviewed 240-query cohort. Under equal data, reviewed graded nDCG is 0.9213 for adapted original, 0.8373 for Unigram and 0.8374 for BPE. Under equal compute it is 0.9193, 0.8124 and 0.8317 respectively. Crossed paired seed/group intervals and exact-match sensitivity are in the [comparison report](encoder-sentence-comparison-v1.json).

This pool was built from the original E5 and BM25 results. Approximately 69–77% of adapted models' reviewed top-10 candidates were unjudged, versus 0.3% for unchanged E5 and zero for BM25. Therefore these adapted sentence-pilot scores are incomplete-label measurements and cannot establish comprehensive search relevance or definitive improvement. The expanded-pool analysis below addresses that top-10 coverage gap. The separate six-domain benchmark explicitly judges every candidate and supplies the frozen model-selection evidence. Sentence results were scored after the choice was frozen and cannot change it.

The measured sentence experiment suggests that alignment training with the original vocabulary is worth investigating further. The [fully judged domain comparison](ENCODER_COMPARISON.md) still supports retaining unchanged E5 for the declared domain objective.

## Completed expanded pool

The [text-free archive](../experiments/retrieval/expanded_pool_review_v1/) adds 6,711 agent judgments: 6,316 grade 0, 362 grade 1 and 33 grade 2. All 2,496 retained original labels are preserved, giving 9,207 judgments over the same 240 queries and 1,996 documents. These are agent judgments, with no independent human-review claim.

The [expanded analysis](encoder-expanded-pool-analysis-v1.json) rescores the same frozen rankings for all 19 systems, with zero unjudged top-10 pairs for every system. Graded nDCG gains for adapted original over unchanged E5 are about 0.266 under both budgets. Custom-tokenizer arms trail adapted original by 0.074–0.095; the paired 95% intervals are below zero. The exact grade-2 sensitivity view agrees with the direction of these comparisons.

Expanded relevance sets change ideal-ranking and recall denominators, so scores from different label versions are not directly comparable. Outside-pool relevance and annotation uncertainty remain unresolved. This secondary sentence analysis does not change the frozen domain choice of unchanged E5. See the [reproduction commands](../experiments/retrieval/README.md#expanded-pool-reproduction).
