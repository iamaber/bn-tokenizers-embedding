# Expanded-label audit

A blinded second-pass agent review checked all 395 new positive pairs and a seeded uniform sample of 120 of the 6,316 new negatives. The 515-pair audit agrees exactly on 464 grades (90.10%). All 33 original grade-2 labels were corroborated. There are 51 disagreements: 32 partial-to-exact upgrades, 12 partial-to-zero changes and seven zero-to-partial changes.

These counts describe the deliberately stratified audit sample, not population-wide annotation accuracy. Agent disagreement does not establish that either grade is correct. Original labels are preserved; the 51 disagreements are recorded for adjudication. No independent human review or language certification is claimed.

The [agent judgments and disagreements](../experiments/retrieval/expanded_label_audit_v1/) are text-free. The [audit report](expanded-label-audit-v1.json) binds them to source checksums. A blank human review packet with all 515 pairs is local at `data/processed/expanded-label-audit-complete-v1/human-review-packet.jsonl`; it hides original and audit grades. Raw corpus text remains uncommitted.

Reproduce from the repository root with the original expanded dataset available. Use a fresh output directory:

```sh
.venv/retrieval/bin/python experiments/retrieval/label_audit.py \
  --archive experiments/retrieval/expanded_pool_review_v1 \
  --dataset data/processed/retrieval-reviewed-expanded-v3 \
  --reviews experiments/retrieval/expanded_label_audit_v1 \
  --output data/processed/expanded-label-audit-repro
```
