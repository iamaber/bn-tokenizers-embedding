# Retrieval improvement implementation — 9 October 2026

The requested scope is all five recommendations from the project review. A working
script is not evidence that human annotation, model training or final evaluation
has finished. This ledger records those separately. Existing frozen test text is
not used for development or model selection.

| Improvement | Required outcome | Current evidence / remaining work |
|---|---|---|
| 1. Reviewed labels and failure analysis | Independent reviewer calibration, adjudicated pairs and alternatives, immutable reviewed qrels, categorized E5 failures | Review packets/importer and 73-miss audit implemented and independently code-reviewed (4d5b71f). User authorized agent judgments; 30-group calibration and 40 full query pools (423 judgments) reviewed so far; finish the remaining pooled labels and 73 failure explanations. |
| 2. Natural search benchmark | Human-authored Bengali/Banglish/English queries against independent domain documents, source/intent holdout, relevance judgments | Agent-authored 60-intent/six-domain benchmark implemented, with 42 development / 18 reserved test intents and all nine language directions. E5 development baselines executed (7269fee); expand pooled judgments and score frozen test after selection. |
| 3. Controlled encoder comparison | Unchanged E5, adapted original tokenizer, adapted 32k Unigram and BPE; same texts/objective; compute and initialization costs measured | Pinned E5 input/vocabulary adapter, teacher representation adaptation, matched contrastive training and strict resumable matrix runner implemented and independently reviewed; Both full three-seed matrices executed: nine equal-data and nine equal-compute checkpoints; sample-order and artifact checks passed. Isolated domain evaluation and inference cost measurements underway. |
| 4. Natural-data expansion | Provenance-audited additions, held-out groups excluded from adaptation and contrastive data, rebuilt grouped splits | Audited original Vashantor v2; expanded grouped corpus has 586,181 train records, including 22,765 accepted natural Vashantor train forms. Repeatable audit excludes every original/project holdout and all 360 authored benchmark surfaces. Frozen shared alignment has 8,000 clean groups (4,800 BanglaTLit, 1,600 Tatoeba, 1,600 Vashantor), with English retention and hashed adaptation text. |
| 5. Success criteria and uncertainty | Criteria frozen before model selection; paired group differences, three seeds for promising candidates, English regression and repeat-based end-to-end cost | Paired group-difference comparison and frozen criteria implemented and reviewed (c66a92d). Three-seed training completed in both budgets. Crossed paired seed/group analysis, fresh-process RSS, five-repeat batch throughput and batch-1 latency, and reserved-test gate implemented and independently reviewed; domain scoring and frozen final decision underway. |

## Experimental decisions

- Start with the existing pinned multilingual E5-small snapshot, rather than
  changing model size while measuring tokenizer effects.
- Primary quality metric: group-weighted nDCG@10 for Banglish → Bengali retrieval.
  Secondary metrics: Recall@10 and MRR@10; report every query language/direction.
- Proposed practical gain: at least +0.03 absolute nDCG@10 over the adapted
  original-tokenizer control, with the paired 95% interval above zero.
- English guardrail: at most 0.01 absolute nDCG@10 regression, assessed on reviewed
  English queries. Missing English evaluation means the guardrail is unverified.
- Deployment guardrails: no more than 1.25× query encoding latency or peak memory
  versus the adapted control on the same hardware and batch size.
- Report equal-data experiments first; equal-compute runs use a declared training
  time budget on the same device. Steps and token counts alone are not equal compute.
- Count initialization, representation adaptation and contrastive training toward
  total cost. Report embedding-table size separately from encoder body size.
- Use three fixed training seeds (17, 29, 43) for candidates reaching the practical
  gain on development data. Freeze choices before running the final reviewed test.

These are project decisions, not thresholds established by the literature. They
may be revised before selection, with a versioned record and rationale.

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
