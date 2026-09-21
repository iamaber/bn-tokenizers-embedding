# Cross-script retrieval pilot

This is the first executable research milestone: a **development-only paired-sentence retrieval proxy**, baseline scores, and pending human-review material. It is not a benchmark of natural search relevance or evidence that our tokenizer improves embeddings.

## Current artifacts

Prepared on 21 September 2026 from existing validation data:

- 300 queries from 300 distinct corpus groups: 292 Banglish and eight script-mixed.
- 2,000 Bengali candidate sentences, all from BanglaTLit to avoid easy source-domain distractors.
- One provisional positive per query, recovered from an exact original transliteration pair.
- Zero group/canonical-text overlap with tokenizer training. The frozen test was never opened.
- No synthetic spelling variants, human grades, or natural search questions were invented.

Local data lives in `data/processed/retrieval-pilot-v1/`, ignored by Git:

| File | Purpose |
|---|---|
| `documents.jsonl` | Candidate text, IDs, source groups and attribution |
| `queries.jsonl` | Query text, IDs, corpus group, language bucket |
| `qrels.jsonl` | Provisional source-pair positives, explicitly marked unreviewed |
| `manifest.json` | Selection seed, input/output hashes, split audit |
| `ranking-*.jsonl` | Top-10 IDs from each baseline |
| `review-pool.jsonl` | 3,152 shuffled candidate judgments; blank reviewer fields |
| `authoring.jsonl` | 300 rows for checking pairs/privacy and authoring natural query variants |

See [results and limitations](../../reports/RETRIEVAL_PILOT.md) and [raw results](../../reports/retrieval-pilot-v1.json).

## Method

`prepare.py` joins raw `train.csv` and `validation.csv` pairs to exact normalized texts in the processed validation split. Raw training-file membership alone does not make a pair eligible: both sides must be in the saved validation split and share its connected group. The raw test file is not read. Group IDs alone cannot establish pairing because corpus deduplication sometimes joins multiple original pairs.

Candidates have at least three whitespace words on both sides. Seeded sampling retains at most one query per connected group, avoids canonical duplicate queries/positives, and fills the candidate pool with canonically distinct Bengali validation sentences. Corpus groups are an approximation, not human-verified independent semantic intents. Audit checks use Unicode letter/mark/number canonical keys and saved group IDs; they do not rule out semantic paraphrase leakage or E5 pretraining overlap.

BM25 uses NFC, lowercasing, Unicode letters/marks/numbers, k1=1.5, b=0.75, binary query-term frequency, and no transliteration. It returns only positive-scoring documents, so zero-overlap queries do not receive arbitrary tied hits. This is deliberately a simple lexical floor.

E5 uses its **original tokenizer and pretrained weights** at revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Inputs use `query: ` / `passage: ` prefixes, attention-mask mean pooling, L2 normalization and exact cosine ranking, following the [official model card](https://huggingface.co/intfloat/multilingual-e5-small). Maximum length is 512 tokens, batch size 16, four CPU threads. No text is sent to an inference API. Model artifacts are downloaded once, then evaluation runs offline. No custom tokenizer was substituted into E5.

Recall@10, nDCG@10 and MRR@10 treat the source counterpart as relevant and all unjudged candidates as zero. This is an incomplete-label proxy: another candidate may also be valid. Results use equal corpus-group weight and 2,000 seeded group-bootstrap draws, with percentile intervals. Intervals reflect query sampling only; they do not account for uncertain labels, model seeds or corpus construction. Eight mixed-script examples cannot support a robust subgroup claim.

## Reproduce

Run from the repository root. Preparation and evaluation refuse to overwrite existing pilot/review directories so annotations are not silently lost. Use a new directory for each run.

```sh
python3.12 -m venv .venv/retrieval
.venv/retrieval/bin/python -m pip install -r experiments/retrieval/requirements.txt
.venv/retrieval/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("intfloat/multilingual-e5-small", revision="614241f622f53c4eeff9890bdc4f31cfecc418b3", cache_dir="models/hf-cache", allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"]))'
python3.12 experiments/retrieval/prepare.py --output data/processed/retrieval-pilot-repro
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/retrieval/bin/python \
  experiments/retrieval/evaluate.py \
  --pilot data/processed/retrieval-pilot-repro \
  --report reports/retrieval-pilot-repro.json \
  --model models/hf-cache/models--intfloat--multilingual-e5-small/snapshots/614241f622f53c4eeff9890bdc4f31cfecc418b3
python3.12 -m unittest discover -s experiments/retrieval -p 'test_*.py' -v
```

Omit `--model` to run only the standard-library BM25 baseline. Research dependencies are separate from package runtime requirements. Dependency and model-file hashes are recorded in the report; the reviewed model revision and original tokenizer must stay fixed for a comparable rerun.

## Human review: the next gate

Begin with 30 randomly selected query groups to calibrate the rubric, then review all 300. Use two fluent Bangla/Banglish readers independently, followed by adjudication. The JSONL files are local annotation material; no reviewers have been contacted and no judgments have been completed.

For `authoring.jsonl`, check whether the source pair conveys the same meaning and whether the text is appropriate for the study. Mark privacy concerns, unclear language, wrong transliteration, near-duplicate intent and unsuitable content for exclusion/review. Preserve source attribution and review the corpus licensing before redistribution. An exclusion must produce a new version and an explicit reason, not silently change the scored dataset.

For `review-pool.jsonl`, grade **semantic correspondence of sentences**:

- 2: same essential meaning, including negation, names and numbers.
- 1: meaningfully related but incomplete or partly different.
- 0: unrelated or contradicts the query.
- Leave null and explain if uncertain/unreadable; null is never a negative label.

Each reviewer should see only their own working copy. System identity, rank and the provisional source-positive flag are absent from the pool. Record independent grades before resolving disagreements. Do not assume retrieved non-counterparts are hard negatives; they may be alternate positives. Add candidate pooling from future systems and random candidates before claiming comprehensive relevance coverage. The current evaluator does not import manual grades; freeze a separately versioned adjudicated qrels file and update its manifest for that stage.

`natural_bengali_query`, `banglish_paraphrase` and `mixed_query` are blank authoring fields, not already validated evaluation queries. Any variants must remain in the same group. Do not copy a candidate sentence verbatim as a Bengali query and interpret self-matching as search accuracy.

## Research milestones after this pilot

1. **Reviewed sentence benchmark:** correct source pairs, judge alternatives, measure agreement, and expand naturally mixed-script coverage. Use this set for development only.
2. **Real search benchmark:** choose a target domain, collect independent documents and natural information needs, and obtain passage-level judgments. Reserve source/intent groups before model selection. The current sentence counterpart need not answer the user's question and cannot stand in for this stage.
3. **Matched encoder feasibility:** compare SentencePiece, original BPE and fused BPE using the same encoder architecture/data/objective; give each its own trained/adapted embedding table. Keep the unchanged E5 system as the practical reference. Define equal-data and equal-compute runs separately and record embedding parameter-count differences.
4. **Ablations and uncertainty:** language weighting, fusion and alignment training one at a time; at least three training seeds for promising candidates; paired group-bootstrap differences, not just separate score intervals. Caching is an inference optimization and has no ID-based accuracy effect.
5. **Final test:** freeze decisions before evaluating a fresh human-labeled source/intent holdout. Do not select models on this test or interpret today's pilot as a final result.

Training full encoders is intentionally downstream of the annotation gate: the repository still lacks a reviewed relevance set, a selected target domain and a concrete encoder compute budget.
