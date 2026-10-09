# Retrieval improvement implementation — 9 October 2026

The requested scope is all five recommendations from the project review. A working
script is not evidence that human annotation, model training or final evaluation
has finished. This ledger records those separately. Existing frozen test text is
not used for development or model selection.

| Improvement | Required outcome | Current evidence / remaining work |
|---|---|---|
| 1. Reviewed labels and failure analysis | Independent reviewer calibration, adjudicated pairs and alternatives, immutable reviewed qrels, categorized E5 failures | Existing 3,152 pooled judgments and 300 authoring rows are blank. Implement independent review packets, strict import, and the 73-miss analysis. Human reviewers still required. |
| 2. Natural search benchmark | Human-authored Bengali/Banglish/English queries against independent domain documents, source/intent holdout, relevance judgments | Target domain and authors/reviewers requested. Implement import, overlap checks and group splits; inspect external FIRE data availability. |
| 3. Controlled encoder comparison | Unchanged E5, adapted original tokenizer, adapted 32k Unigram and BPE; same texts/objective; compute and initialization costs measured | Pinned E5 and both tokenizers exist locally; training/input adapter and experiment runs are pending. Feasibility runs may use provisional development labels, with that limitation explicit. |
| 4. Natural-data expansion | Provenance-audited additions, held-out groups excluded from adaptation and contrastive data, rebuilt grouped splits | Audit BanglaDual before inclusion; build clean alignment pairs from existing training groups first. Natural versus synthetic provenance must be preserved. |
| 5. Success criteria and uncertainty | Criteria frozen before model selection; paired group differences, three seeds for promising candidates, English regression and repeat-based end-to-end cost | Implement comparison/decision tooling and measured runs. Final success cannot be claimed on the current unreviewed pilot. |

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
