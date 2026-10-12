# Retrieval improvement implementation — 9 October 2026

The requested scope is all five recommendations from the project review. A working
script is not evidence that human annotation, model training or final evaluation
has finished. This ledger records those separately. Existing frozen test text is
not used for development or model selection.

| Improvement | Completed outcome | Evidence / limits |
|---|---|---|
| 1. Reviewed labels and failure analysis | All 3,152 original pooled labels and 300 source-pair decisions completed; 240 retained queries / 1,996 documents; all 73 historical misses annotated | [Review report](RETRIEVAL_REVIEW.md), immutable text-free archive and reviewed manifest. User-authorized agent review; independent human review is not claimed. Outside-pool relevance stays unknown. |
| 2. Domain search proxy | 60 intents across six requested domains in Bengali/Banglish/English; 42 development / 18 reserved test; all 3,600 intent pairs judged and projected over nine directions | [Benchmark](DOMAIN_BENCHMARK.md). All 19 encoders scored on both splits after frozen selection. Authored proxy replaces the initial human-data dependency at the user's request; no natural-traffic claim. |
| 3. Controlled encoder comparison | Unchanged E5 and adapted original/32k Unigram/32k BPE; both budgets, three seeds; 18 actual checkpoints and 19 model evaluations on domain and sentence cohorts | [Comparison](ENCODER_COMPARISON.md), verified matrices and training/evaluation reports. Same data/objectives; heuristic initialization and small feasibility budgets. |
| 4. Natural-data expansion | Provenance-audited original Vashantor additions; expanded train has 586,181 records, including 22,765 accepted new-source train forms; shared 8,000-group alignment | Source locks and repeatable audit preserve original/project/benchmark holdouts and remove reserve-only corpus rows. No training overlap found by canonical/group checks; semantic overlap cannot be ruled out. |
| 5. Success criteria and uncertainty | Criteria frozen before adapted scoring; crossed paired seed/group intervals, English guardrail, repeated full-query latency and fresh-process RSS; reserved test bound to frozen decision | Both custom arms fail quality/English gates and pass latency/memory gates in both budgets. Choice remains unchanged E5; test and secondary sentence results cannot revise it. |

## Experimental decisions

- Start with the existing pinned multilingual E5-small snapshot, rather than
  changing model size while measuring tokenizer effects.
- Primary quality metric: group-weighted nDCG@10 for Banglish → Bengali retrieval.
  Secondary metrics: Recall@10 and MRR@10; report every query language/direction.
- Declared practical gain: at least +0.03 absolute nDCG@10 over the adapted
  original-tokenizer control, with the paired 95% interval above zero.
- English guardrail: at most 0.01 absolute nDCG@10 regression, assessed on reviewed
  English queries. Missing English evaluation means the guardrail is unverified.
- Deployment guardrails: no more than 1.25× query encoding latency or peak memory
  versus the adapted control on the same hardware and batch size.
- Report equal-data experiments first; equal-compute runs use a declared training
  time budget on the same device. Steps and token counts alone are not equal compute.
- Count initialization, representation adaptation and contrastive training toward
  total cost. Report embedding-table size separately from encoder body size.
- Use three fixed training seeds (17, 29, 43) for all declared arms; all were run. Freeze choices before running the final reviewed test.

These are project decisions, not thresholds established by the literature. They
were frozen before adapted development scoring and cannot be revised from reserved-test results.

## External dependencies

On 9 October the user explicitly requested that the implementation agent create
and judge the material, targeting business, banking, technology, science, general
conversation and QA. The implementation will therefore provide agent-authored,
agent-reviewed domain evaluation and label that provenance explicitly. It must not
claim independent human review or naturally collected user queries. Those remain
limits on generalization, not a reason to stop the user-authorized experiments.

This user decision supersedes the initial human-review dependency in items 1–2.
Complete the annotation, evaluation and matched experiments with the authorized
agent judgments; preserve rubric, source records, exclusions and disagreement
support so independent reviewers can audit them later. Do not mark the five-item
goal complete until the requested implemented and executed outcomes are verified.

## Completion

The user-authorized implemented and executed outcomes above are complete. Larger natural-search corpora, independent fluent review, longer training, vocabulary-size ablations and stronger transfer methods are follow-up research. They are not claimed as completed results. The original frozen project test was read only for integrity and never scored; the separately reserved domain test was scored only after the model decision froze.

Adapted-system sentence pooling is now complete; see the [second-experiment ledger](ROUND2_PLAN.md) and [expanded analysis](encoder-expanded-pool-analysis-v1.json).
