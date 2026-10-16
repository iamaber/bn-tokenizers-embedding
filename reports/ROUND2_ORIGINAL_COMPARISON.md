# Longer original-tokenizer adaptation on fresh source-backed retrieval

**Decision: retain unchanged multilingual E5-small.** The revised training package recovers quality relative to the earlier adapted original-tokenizer controls, but fails the frozen gain-over-unchanged and English-retention gates on development. Reserved test was scored after this decision froze and does not change it.

## Completed work

The [blinded label audit](LABEL_AUDIT.md) covers all 395 added positives and 120 sampled negatives. It records 51 grade disagreements and provides an unfilled human-review packet. The labels remain preserved pending adjudication; these are agent reviews with no human-review claim.

The [fresh source-backed benchmark](SOURCE_BENCHMARK.md) contains 90 intents across six domains: 60 development and 30 reserved test, with Bengali, Banglish and English forms plus one Banglish spelling variant per intent. All 4,500 same-split intent pairs have explicit agent judgments, projected across the nine language directions. Source groups stay wholly in one split. The original/expanded corpus and actual training pairs/negatives have zero audited canonical or near-copy matches. Dataset v2 explicitly declares reviewed-label provenance; its texts and judgments are identical to v1.

All six longer original-tokenizer runs are trained, saved and verified under the [existing frozen training protocol](../experiments/retrieval/round2_training_protocol.json): three seeds in each budget. Equal-data runs finish 192 representation plus 384 contrastive steps. Equal-compute runs use 360 charged seconds, allowing a reported final-step overrun. Every batch has eight English and eight Banglish queries; every third contrastive batch includes reviewed negatives. All 13 encoders—unchanged E5, six earlier short controls and six longer runs—were evaluated in fresh sequential processes on both splits.

## Measured quality

Figures are nDCG@10 with equal weight per domain; adapted-family values average three seeds. English is the equal-direction mean over English queries targeting Bengali, Banglish and English documents. Banglish figures include canonical and spelling-variant queries.

| Model family | Development BL→BN | Development English | Reserved-test BL→BN | Reserved-test English |
|---|---:|---:|---:|---:|
| Unchanged E5 | 0.6835 | 0.9072 | 0.8874 | 0.9597 |
| Short original, equal data | 0.5291 | 0.8470 | 0.8547 | 0.9233 |
| Longer original, equal data | 0.6296 | 0.7883 | 0.8858 | 0.9114 |
| Short original, equal compute | 0.5164 | 0.8469 | 0.8390 | 0.9227 |
| Longer original, equal compute | 0.6210 | 0.7939 | 0.8654 | 0.9012 |

The longer recipe gains about 0.10 over short adaptation on development, with positive paired intervals in both budgets. It remains below unchanged E5 and produces substantially worse English retrieval. On reserved test, its gain over the short control is smaller; performance relative to unchanged E5 is uncertain, while the English regression remains evident.

| Split / budget | Longer minus short BL→BN, 95% interval | Longer minus unchanged BL→BN, 95% interval | English minus unchanged, 95% interval |
|---|---|---|---|
| Development / equal data | +0.1004 [0.0261, 0.1884] | −0.0540 [−0.1100, 0.0025] | −0.1189 [−0.1547, −0.0863] |
| Development / equal compute | +0.1046 [0.0357, 0.1676] | −0.0626 [−0.1179, −0.0083] | −0.1133 [−0.1541, −0.0788] |
| Reserved test / equal data | +0.0311 [−0.0001, 0.0838] | −0.0016 [−0.0552, 0.0572] | −0.0483 [−0.0975, −0.0115] |
| Reserved test / equal compute | +0.0264 [0.0014, 0.0487] | −0.0220 [−0.0900, 0.0431] | −0.0585 [−0.1101, −0.0184] |

Intervals use crossed paired resampling of three training seeds and six domain blocks. Spelling variants are not independent bootstrap observations. These intervals exclude annotation uncertainty and do not establish generalization to natural search traffic.

## Frozen gates and runtime

The [criteria](../experiments/retrieval/round2_original_criteria.json) require at least +0.03 BL→BN nDCG over short adaptation, a positive paired interval, gain over unchanged E5, English loss no greater than 0.01, and latency/memory at most 1.25× unchanged E5 in every seed under both budgets.

On development, both budgets pass the short-control gain and paired-interval gates but fail gain over unchanged E5 and English retention. Equal data passes latency and memory. Equal compute passes memory but one seed has a 1.263× query-latency ratio, just above the cap. The rejection therefore does not depend on that timing result.

Local timings vary substantially: reserved-test latency ratios are roughly 0.37–0.54× despite identical original-tokenizer architecture. These small CPU measurements are exploratory and do not establish a deployment speedup. All workload hashes match within each split, and memory ratios are close to 1.0. Larger workload validation remains separate from this experiment.

## Interpretation and limits

This contrast changes training exposure, English replay/retention, hard-negative use and the positive alignment data, which removes 18 previously identified errors. It cannot isolate the causal contribution of any one change. The two protocols also have different step/time budgets; the shared budget labels do not imply equal cost across the earlier and longer protocols.

The benchmark is a 90-intent agent-authored source-backed pilot, below the proposed 180-intent target. Passages are short extracts/summaries and translated forms, while communication guidance stands in for conversation. A small number of source pages creates topic shifts between development and test. Lexical exclusion cannot rule out semantic overlap or base-model pretraining exposure. No independent fluent-human review, natural user distribution, or long-document benchmark claim is made.

The result supports investigating the English tradeoff before increasing training further. It provides no basis for replacing unchanged E5 under the declared gates, and it leaves the first experiment's frozen decision intact.

## Evidence and reproduction

- [Verification record](round2-original-verification-v1.json): 55 tests pass; all 26 evaluation reports and six training reports match their saved checksums.
- [Training inventory](round2-original-training-v1.json) and [individual training reports](round2-original-training/).
- [Frozen inputs](round2-original-frozen-inputs-v1.json), [development selection](round2-original-selection-v1.json), and [reserved-test confirmation](round2-original-confirmation-v1.json).
- [Absolute-quality summary](round2-original-summary-v1.json) and [all 26 evaluation reports](round2-original-evaluation/).
- [Source benchmark and commands](SOURCE_BENCHMARK.md), [label audit](LABEL_AUDIT.md), and [overlap audit](source-benchmark-overlap-v1.json).

Rankings, scored query rows, checkpoint weights and raw corpus text remain local in checksum-bound ignored directories. The runner verifies identities and file hashes before evaluation, preserves frozen artifacts, and requires a saved development selection before entering reserved-test scoring. Re-verification of saved rankings and comparisons reproduces the frozen selection and confirmation exactly.
