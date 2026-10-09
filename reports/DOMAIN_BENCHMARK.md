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
contamination. The existing project's frozen test split was not opened. New test
scores remain uncomputed pending model selection.

## Unchanged E5-small development baseline

Pinned revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, original tokenizer,
prefixes, masked mean pooling and cosine ranking; same CPU settings as the pilot.
Each direction uses its own corpus and source-answer relevance labels.

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

## Reproduction

```sh
python3.12 experiments/retrieval/domains.py \
  --output data/processed/domain-benchmark-repro
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/retrieval/bin/python \
  experiments/retrieval/evaluate.py \
  --pilot data/processed/domain-benchmark-repro/development/bl-bn \
  --report reports/domain-baseline-repro.json \
  --model models/hf-cache/models--intfloat--multilingual-e5-small/snapshots/614241f622f53c4eeff9890bdc4f31cfecc418b3
```

Run the other development directories for the remaining directions. Freeze model
selection before scoring test. Keep both the much larger sentence pilot and this
controlled QA proxy in the encoder comparison; report their label limitations
separately. Pool adapted systems' alternatives before accepting final judgments.
