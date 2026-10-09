# Agent-authored domain retrieval proxy — 9 October 2026

The user requested that the implementation agent create and judge material for
business, banking, technology, science, general conversation and QA. This benchmark
implements that request. It is **agent-authored and agent-reviewed**, not naturally
collected user queries or independently human-labeled data.

The committed [material](../experiments/retrieval/domain_material.jsonl) contains
60 distinct reference intents, ten per domain, each with Bengali, Banglish and
English answer passages and separately phrased questions. Operational policies,
bank rules and personal plans are fictional examples, not claims about real
institutions. Basic science and arithmetic passages serve as controlled reference
answers. No benchmark text is included in encoder training.

The builder reserves three intents per domain for test and seven for development,
using seed 17. All translated forms of an intent stay in one split. Each split has
nine language-direction datasets, including monolingual checks. Development has
42 queries/documents per direction; test has 18. Reusing an intent in multiple
directions does not create additional independent observations.

All 360 document/query surfaces were checked against the tokenizer's 563,476
training records using NFC/case/punctuation-insensitive canonical keys: no exact
canonical overlaps. This does not rule out semantic overlap or pretrained-model
contamination. The existing project's frozen test was read only for holdout
integrity checks, never scored or used for selection. Expanded training also
excludes all 360 benchmark surfaces. The new reserved domain test was scored
after the development choice was frozen; that choice remains unchanged E5.

## Complete agent intent review

The reviewed v2 corpus uses the committed
[3,600-label intent matrix](../experiments/retrieval/domain_intent_judgments_v1.jsonl),
bound to its source material and label checksums. The agent compared every
question intent with every answer passage: 60 own-intent answers receive grade 2,
and 3,540 other-intent answers receive grade 0. No alternative passage supplies
the requested specific facts. The shared intent labels are projected across the
authored parallel language forms; this is not independent language review.

Both language surfaces and the original 42/18 split are byte-identical to v1.
Every candidate has an explicit label: 15,876 query-document judgments across
nine development directions and 2,916 across nine test directions. The
[reviewed manifest](domain-benchmark-reviewed-v2.json) records coverage and
provenance. Unlike the small original sentence pilot's pooled judgments, this
controlled corpus has no unjudged candidates.

## Unchanged E5-small development baseline

Pinned revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, original tokenizer,
prefixes, masked mean pooling and cosine ranking; same CPU settings as the pilot.
Each direction uses its own corpus. The complete intent review preserved the
baseline relevance labels, so the following development results remain valid.

| Query → document | Recall@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|
| Banglish → Bengali | 1.0000 | 0.8039 | 0.7420 |
| Bengali → Banglish | 1.0000 | 0.8790 | 0.8405 |
| Banglish → English | 0.9762 | 0.9498 | 0.9405 |
| English → Banglish | 0.9762 | 0.9603 | 0.9558 |
| Bengali → English | 1.0000 | 0.9912 | 0.9881 |
| English → Bengali | 1.0000 | 0.9881 | 0.9841 |
| Banglish → Banglish | 1.0000 | 0.9712 | 0.9627 |
| Bengali → Bengali | 1.0000 | 1.0000 | 1.0000 |
| English → English | 1.0000 | 1.0000 | 1.0000 |

Raw reports are `domain-baseline-<direction>-v1.json`; local dataset manifests,
rankings, per-query scores and newly pooled alternatives are under
`data/processed/domain-benchmark-v1/`. Several metrics are near their ceiling:
this small corpus cannot establish broad real-search generalization or a useful
tokenizer effect by itself. No tokenizer-swap gain is established here.

## Encoder comparison and reproduction

All 18 adapted checkpoints and the unchanged encoder have now been evaluated on
development and reserved test, across all nine directions. See the
[comparison](ENCODER_COMPARISON.md) for quality, paired intervals, latency,
memory and the frozen decision. Custom tokenizers failed the declared development
quality and English-retention gates under both budgets.

```sh
python3.12 experiments/retrieval/domains.py \
  --judgments experiments/retrieval/domain_intent_judgments_v1.jsonl \
  --output data/processed/domain-benchmark-repro
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/retrieval/bin/python \
  experiments/retrieval/evaluate.py \
  --pilot data/processed/domain-benchmark-repro/development/bl-bn \
  --report reports/domain-baseline-repro.json \
  --model models/hf-cache/models--intfloat--multilingual-e5-small/snapshots/614241f622f53c4eeff9890bdc4f31cfecc418b3
```

Run the other development directories for the remaining directions. The immutable
selection and test-suite hashes allow auditing the ordering of the executed
experiments. Sentence-pilot review and scoring are complete and reported [separately](RETRIEVAL_REVIEW.md);
they cannot revise the domain choice after test scoring.
