# Retrieval experiments

The requested five improvements are implemented and executed: reviewed sentence labels, a multilingual domain proxy, controlled encoder adaptation, audited natural-data expansion, and frozen quality/cost selection with paired uncertainty. See the [implementation ledger](../../reports/IMPROVEMENT_PLAN.md) and [measured comparison](../../reports/ENCODER_COMPARISON.md).

**Current choice: unchanged multilingual E5-small.** Both custom 32k tokenizers failed the declared domain quality and English-retention gates under both budgets. All 18 adapted checkpoints were trained, and 19 encoders were evaluated on development, reserved test and both sentence cohorts. This is a small feasibility experiment; it does not settle whether better vocabulary transfer or longer training can help.

## Datasets and review

The original `data/processed/retrieval-pilot-v1` remains unchanged: 300 BanglaTLit validation query groups, 2,000 Bengali documents and provisional source qrels. Its historical E5 Recall@10 is 0.7567. The reviewed version `data/processed/retrieval-reviewed-v1` has 240 queries and 1,996 documents, with all original 3,152 pooled labels and 300 pair/privacy decisions completed. Sixty query groups are excluded with reasons, including four privacy exclusions. All 73 historical misses have diagnostic notes. See the [review report](../../reports/RETRIEVAL_REVIEW.md).

The original pool covers E5/BM25/source candidates, not all 1,996 documents for every query. Unknown labels remain unknown. Numerical proxy scoring counts unknowns as zero and reports coverage. The sentence comparison includes a strict grade-2 sensitivity view because grade 1 means only partial correspondence. Corpus deduplication groups alone do not establish exact pairs or independent semantic intents.

The [domain material](domain_material.jsonl) has 60 intents, ten per requested domain: business, banking, technology, science, conversation and QA. Each has separately phrased questions and passages in Bengali, Banglish and English. Seed 17 reserves 42 development and 18 test intents; every language form stays in the same split. The complete 3,600-label shared intent matrix projects to all nine language directions, giving explicit labels for every candidate. Policies and personal plans are fictional. See the [benchmark report](../../reports/DOMAIN_BENCHMARK.md).

The user explicitly authorized agent authorship and judgments. All reviewed artifacts say `agent`; no human identity or independent human review is invented. Separate blank reviewer/calibration packets remain available for future independent review. Raw corpus text stays local and ignored by Git; committed sentence archives contain only IDs, labels, rationale, provenance and checksums.

## Natural training expansion

Only audited original Vashantor v2 training forms were added. The expanded grouped corpus has 586,181 train records, including 22,765 accepted Vashantor training forms. Every original project holdout, original Vashantor validation/test form and authored benchmark surface is excluded from train. The project test was read only for integrity, never scored or used for selection. Reserve-only benchmark rows are suppressed from all corpus outputs.

The frozen alignment sample has 8,000 clean training groups: 4,800 BanglaTLit, 1,600 Tatoeba and 1,600 Vashantor. Both forms must belong to the same train group; identities and duplicated canonical text are rejected. Source locks, expanded manifests and repeatable audits record provenance. The merged BanglaDual Wikipedia subset overlaps an existing upstream dump and uses generated Romanization, so it was not used as a new natural-data source.

## Encoder protocol

The base is `intfloat/multilingual-e5-small` revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. All arms share its encoder body, texts, objectives and optimizer. Original/custom control IDs are remapped explicitly; masked mean pooling and L2 normalization use `query: ` / `passage: ` prefixes.

The custom vocabulary starts with mean original-subword embeddings and explicit byte/control fallbacks. This is heuristic initialization, not FOCUS or ZeTT. Every adapted arm receives frozen-teacher representation distillation followed by symmetric in-batch contrastive training. The original-tokenizer control receives the same stages.

[Run protocol](run_protocol.json): seeds 17, 29 and 43, CPU/four threads, batch 16, training maximum 128 tokens. Equal data is 64 representation plus 128 contrastive steps. Equal compute is 120 wall seconds including loading, vocabulary initialization and both stages; a final optimizer-step overrun is reported. Saving is excluded. Same seeded data order is verified for equal-data arms. All checkpoint/report/model/data/code hashes are recorded and checked on resume.

[Criteria](criteria.json) were frozen before adapted domain scoring. A custom candidate must gain at least 0.03 absolute Banglish→Bengali nDCG over adapted original, have a positive paired interval, lose no more than 0.01 English-query mean nDCG, and stay within 1.25× latency/RSS in every seed under both budgets. Uncertainty jointly resamples training seeds and intent groups. The domain choice was frozen before reserved-test scoring; secondary sentence results cannot revise it.

Evaluation uses 512 tokens, batch 16 and fresh sequential processes. Five warm batch-throughput passes and 60 batch-1 requests include tokenization, encoding and exact cached-document search. RSS includes model loading and evaluation. These small CPU measurements do not establish production latency or large-index performance.

## Reproduction

Run from the repository root with the pinned environment in `requirements.txt`, the source-lock data, native tokenizer models and pinned E5 snapshot present. All evaluation runs offline. Outputs refuse to overwrite reviewed versions.

Restore the completed review against its checksum-bound original packet:

```sh
.venv/retrieval/bin/python experiments/retrieval/annotation_archive.py restore \
  --pilot data/processed/retrieval-pilot-v1 \
  --archive experiments/retrieval/pilot-annotation-complete-v1 \
  --output data/processed/retrieval-review-repro
.venv/retrieval/bin/python experiments/retrieval/review.py import \
  --pilot data/processed/retrieval-pilot-v1 \
  --judgments data/processed/retrieval-review-repro/reviewer-1.jsonl \
  --pair-review data/processed/retrieval-review-repro/pair-review.jsonl \
  --output data/processed/retrieval-reviewed-repro
.venv/retrieval/bin/python experiments/retrieval/domains.py \
  --judgments experiments/retrieval/domain_intent_judgments_v1.jsonl \
  --output data/processed/domain-benchmark-repro
```

`vashantor.py`, `audit_expansion.py` and `alignment.py` build/audit the expanded training inputs. `run_matrix.py` executes the fixed training protocol; the committed verified matrices and individual training reports document the completed runs. `run_evaluations.py` evaluates all checkpoints and validates resumed artifacts. `seed_analysis.py` freezes development selection; `confirm.py` checks reserved-test evidence against that exact selection. `sentence_analysis.py` reports raw/reviewed/strict correspondence, paired comparisons and unknown-candidate coverage.

The final suite reports are `reports/encoder-evaluation-{development-reviewed-v2,test-reviewed-v2,sentence-v1}.json`; full model reports are under `reports/encoder-evaluation/`. Rankings, scores, corpora and checkpoint weights stay local in their hash-bound directories. To regenerate model results, use a new output directory with the same verified matrices and datasets.

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/retrieval/bin/python \
  -m unittest discover -s experiments/retrieval -p 'test_*.py'
```

Future work includes independent fluent review, larger natural-search corpora, edition-specific external retrieval data, better vocabulary-transfer methods and longer adaptation budgets. These are limits and follow-up experiments, not results already established by this run.

## Expanded-pool reproduction

The [completed archive](expanded_pool_review_v1/) contains 6,711 additional agent judgments. Together with the 2,496 preserved labels, these cover every top-10 candidate from all 19 frozen encoders. The [expanded report](../../reports/encoder-expanded-pool-analysis-v1.json) includes graded and exact grade-2 comparisons without changing the frozen domain choice. Raw text and rankings remain local; the same source dataset and frozen evaluation outputs are required.

Run from the repository root. Use fresh output paths, since the commands refuse overwrites. Rebuild the packet, verify it against the archived checksums, import the archived judgments, then rescore:

```sh
.venv/retrieval/bin/python experiments/retrieval/expand_pool.py build \
  --source data/processed/retrieval-reviewed-v1 \
  --suite reports/encoder-evaluation-sentence-v1.json \
  --output data/processed/retrieval-expanded-pool-repro

.venv/retrieval/bin/python - <<'PYVERIFY'
import hashlib
import json
from pathlib import Path

archive = Path("experiments/retrieval/expanded_pool_review_v1")
pool = Path("data/processed/retrieval-expanded-pool-repro")
manifest = json.loads((archive / "manifest.json").read_text())
for path, key in (
    (pool / "packet.jsonl", "packet_sha256"),
    (pool / "manifest.json", "pool_manifest_sha256"),
    (archive / "pool-manifest.json", "pool_manifest_sha256"),
    (archive / "judgments.jsonl", "judgments_sha256"),
):
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[key]:
        raise ValueError(f"Archive checksum mismatch: {path}")
print("Packet, pool manifest and archived judgments verified")
PYVERIFY

.venv/retrieval/bin/python experiments/retrieval/expand_pool.py import \
  --pool data/processed/retrieval-expanded-pool-repro \
  --labels experiments/retrieval/expanded_pool_review_v1/judgments.jsonl \
  --output data/processed/retrieval-reviewed-expanded-repro

.venv/retrieval/bin/python experiments/retrieval/expanded_analysis.py \
  --dataset data/processed/retrieval-reviewed-expanded-repro \
  --suite reports/encoder-evaluation-sentence-v1.json \
  --selection reports/encoder-domain-selection-v2.json \
  --output data/processed/retrieval-expanded-analysis-repro.json
```

Reproduction paths differ from the saved report's paths; compare coverage and label-view results rather than whole-file hashes. The report records checksums for the analysis and its scoring, coverage, uncertainty and input-validation helpers. Labels outside the pool remain unknown, and the bootstrap does not include annotation uncertainty.
